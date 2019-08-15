from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class MixupSequence(Sequence):
    def __init__(self, x, y, batch_size=32, alpha=0.2):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.mix_num = 2

        #
        self.__sample_num = len(self.x)
        #
        self.__shuffuled_idxes = self.get_indexes()
        
        return

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x, batch_y = self.get_next_batch(idx)
        mixed_x, mixed_y = self.mixup(batch_x, batch_y)
        return mixed_x, mixed_y
    
    def on_epoch_end(self):
        self.__shuffuled_idxes = self.get_indexes()
        return

    def mixup(self, batch_x, batch_y):
        '''
        return mixuped_x, mixuped_y.
        batch_x = self.x[batch_idxs], batch_y = self.y[batch_idxs]
        batch_idxs = 
         [
          [idx(0), idx(1), ..., idx(batch_size)], # indexes of mixed no. 1
          [idx(0), idx(1), ..., idx(batch_size)], # indexes of mixed no. 2
         ].
        '''
        batch_size = batch_x.shape[1]

        #mixed_x[k,:,:,...] = batch_x[0,k,:,:,...] * mixup_rate[0,k] + batch_x[1,k,:,:,...] * mixup_rate[1,k] + ... + batch_x[mix_num,k,:,:,...] * mixup_rate[mix_num,k]
        #mixed_y[k,:,:,...] = batch_y[0,k,:,:,...] * mixup_rate[0,k] + batch_y[1,k,:,:,...] * mixup_rate[1,k] + ... + batch_y[mix_num,k,:,:,...] * mixup_rate[mix_num,k]
        
        mixup_rate = np.random.beta(self.alpha, self.alpha, size=batch_size)        
        casted_shape_x = [batch_size] + [1] * (len(batch_x.shape) - 2) # -2 = mixed No. and batch no.
        casted_shape_y = [batch_size] + [1] * (len(batch_y.shape) - 2) # -2 = mixed No. and batch no.
        mixup_rate_x = np.reshape(mixup_rate, casted_shape_x)
        mixup_rate_y = np.reshape(mixup_rate, casted_shape_y)

        mixuped_x = mixup_rate_x * batch_x[0] + (1.0 - mixup_rate_x) * batch_x[1]
        mixuped_y = mixup_rate_y * batch_y[0] + (1.0 - mixup_rate_y) * batch_y[1]

        return mixuped_x, mixuped_y

    def get_indexes(self):
        '''
        return indexes.
        indexes = 
         [
          [shuffled [0, 1,.., sample_num]], #indexes of mixed no. 1
          [shuffled [0, 1,.., sample_num]], #indexes of mixed no. 2
         ]
        '''
        indexes = np.ones((self.mix_num, self.__sample_num), dtype='int') * np.arange(self.__sample_num)
        for i in range(self.mix_num):
            np.random.shuffle(indexes[i,:])
        return indexes

    def get_next_batch(self, idx):
        batch_indxes = self.__shuffuled_idxes[:, idx*self.batch_size : (idx+1)*self.batch_size]
        return self.x[batch_indxes], self.y[batch_indxes]

class ImageMixupSequence(MixupSequence):
    def __init__(self, x, y, keras_ImageDataGenelator, batch_size = 32, alpha = 0.2):
        super().__init__(x, y, batch_size, alpha)

        #
        self.keras_img_genelator = keras_ImageDataGenelator
        self.keras_img_gens = []
        for i in range(self.mix_num):
            self.keras_img_gens.append(
                self.keras_img_genelator.flow(
                    x=self.x, y=self.y, 
                    batch_size=self.batch_size, 
                    shuffle=True, seed=None)
                )
        #
        self.__shuffuled_idxes = None
        
        return

    def on_epoch_end(self):
        pass

    def get_indexes(self):
        pass

    def get_next_batch(self, idx):
        batch_x, batch_y = [], []
        min_batch_size = -1
        for k_img_gen in self.keras_img_gens:
            temp_x, temp_y = k_img_gen.__next__()
            batch_x.append(temp_x)
            batch_y.append(temp_y)
            #
            if min_batch_size == -1 or min_batch_size > temp_x.shape[0]:
                min_batch_size = temp_x.shape[0]

        #Align batch sizes.
        #Thre is a possibility that the batch sizes may differ in case of multiple thread processing.
        for i in range(len(batch_x)):
            batch_x[i] = (batch_x[i])[:min_batch_size]
            batch_y[i] = (batch_y[i])[:min_batch_size]

        batch_x, batch_y = np.array(batch_x), np.array(batch_y)

        return batch_x, batch_y