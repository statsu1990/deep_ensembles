import os

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, Concatenate, LeakyReLU, Lambda
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import regularizers
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
import numpy as np

import mixup_generator as mxgen


class BinaryClassifierCnnWithPartNormDist:
    """
    classify 2 label in cifar10
    """
    def __init__(self):
        return

    def built_model(self, input_shape=None):
        """
        model input: image shape(32, 32, 3)
        model output: probability of being label1, uncertainty score.
                      Range is [0,1] and [0,1], respectively. 
        """
        # constants
        if input_shape is None:
            # assume cifar10 image
            input_shape = (32, 32, 3)

        # model structure
        input_img = Input(input_shape)
        h = input_img

        h = Conv2D(32, (3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(32, (3, 3))(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)

        h = Conv2D(64, (3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(64, (3, 3))(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)

        oup_cnn = Flatten()(h)
        #oup_cnn = GlobalAveragePooling2D()(h)
        
        oup_cnn = Dense(32)(oup_cnn)
        oup_cnn = BatchNormalization()(oup_cnn)
        oup_cnn = Activation('relu')(oup_cnn)
        
        # expec
        #h_expec = Dense(16)(oup_cnn)
        #h_expec = BatchNormalization()(h_expec)
        #h_expec = Activation('relu')(h_expec)
        #h_expec = Dense(1)(h_expec)
        h_expec = Dense(1)(oup_cnn)
        h_expec = Activation('sigmoid')(h_expec)

        # var
        #h_var = Dense(16)(oup_cnn)
        #h_var = BatchNormalization()(h_var)
        #h_var = Activation('relu')(h_var)
        #h_var = Dense(1)(h_var)
        h_var = Dense(1)(oup_cnn)
        h_var = Activation('softplus')(h_var)
        h_var = Lambda(lambda x: x + 1e-6, output_shape=(1,))(h_var)

        oup = Concatenate(axis=-1)([h_expec, h_var])

        # model
        self.model = Model(inputs=input_img, outputs=oup)

        self.model.summary()

        return

    def part_norm_dist_log_likelihood(self, y_true, y_pred):
        """
        expec = y_pred[:,0]
        var = y_pred[:,1]

        -ln(L) = ln(I) + 0.5 * ln(var) + 0.5 * (x - expec)^2 / var

        """
        expec = y_pred[:,0:1]
        var = y_pred[:,1:2]

        loss_var = 0.5 * K.log(var)
        loss_l2 = 0.5 * K.square(y_true - expec) / var

        I = 0.5 * (tf.math.erf((1.0 - expec) / K.sqrt(2.0 * var)) - tf.math.erf((0.0 - expec) / K.sqrt(2.0 * var)))
        loss_I = K.log(I)

        loss_reg_var = K.sqrt(var) * 16.0 # regularization of var

        loss = loss_I + loss_var + loss_l2 + loss_reg_var

        return loss

    def train_model(self, x_train, y_train, x_test, y_test, epochs, batch_size, alpha=0.2):
        # compile
        self.model.compile(loss=self.part_norm_dist_log_likelihood, optimizer='nadam', metrics=['accuracy'])

        # datagen
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.1,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        datagen.fit(x_train)

         #mixup data augmentation
        datagen_mixup = mxgen.ImageMixupSequence(
                x=x_train, y=y_train, batch_size=batch_size,
                alpha=alpha,
                keras_ImageDataGenelator=datagen
                )

        #
        do_mixup = False
        if do_mixup:
            dgen = datagen_mixup
        else:
            dgen = datagen.flow(x_train, y_train, batch_size=batch_size)

        # fit
        self.model.fit_generator(dgen,
                            epochs=epochs,
                            steps_per_epoch=int(x_train.shape[0] / batch_size),
                            validation_data=(x_test, y_test),
                            )

        # score
        scores = self.model.evaluate(x_train, y_train, verbose=0)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])

        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        return

    def save_model(self, save_file_name):
        """
        save model
        """
        # dir
        dir_name = os.path.dirname(save_file_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        # save model
        self.model.save(save_file_name)
        print('Saved trained model at %s ' % save_file_name)
        
        # visualize
        plot_model(self.model, to_file=os.path.join(dir_name, 'model_structure.png'), show_shapes=True, show_layer_names=False)

        """
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # model file
        model_path = os.path.join(save_dir, 'trained_model.h5')
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)
        """
        
        return

    def load_model(self, model_file_path):
        """
        load model .h5 file
        """
        self.model = keras.models.load_model(model_file_path, custom_objects={'part_norm_dist_log_likelihood': self.part_norm_dist_log_likelihood})
        return
