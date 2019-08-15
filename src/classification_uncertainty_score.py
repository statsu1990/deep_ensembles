import os

import numpy as np
import scipy as sp
from scipy import stats as spst
import matplotlib.pyplot as plt

import cifar10_data
import classifier_cnn

def ensemble_part_norm():
    # ####################################################
    # make data
    # ####################################################

    # dog and cat images in cifar10
    # dog label = 1, cat label = 0
    dog_cat = cifar10_data.Cifar10_Dog_Cat()
    dog_cat.make_binary_data()


    # ####################################################
    # make cnn model classifing dog and cat
    # 犬と猫を分類するCNN作成
    # ####################################################
    ENSEMBLE_NUM = 10
    SAVED_MODEL_NAMES = [os.path.join(os.getcwd(),'saved_models_ensemble','trained_model' + str(i) + '.h5') for i in range(ENSEMBLE_NUM)]
    
    ## train model
    ## train acurracy 0.9227, loss 0.2634
    ## test  acurracy 0.8630, loss 0.3913
    cnns = []
    do_training = True
    if do_training:
        for i in range(ENSEMBLE_NUM):
            train_sample_num = len(dog_cat.x_train)
            if ENSEMBLE_NUM != 1:
                bagging_idx = np.random.choice(np.arange(train_sample_num), size=train_sample_num, replace=True)
            else:
                bagging_idx = np.arange(train_sample_num)

            cnn = classifier_cnn.BinaryClassifierCnnWithPartNormDist()
            cnn.built_model()
            cnn.train_model(dog_cat.x_train[bagging_idx], dog_cat.y_train[bagging_idx], 
                            dog_cat.x_test, dog_cat.y_test, 
                            epochs=100, batch_size=64, alpha=None)
            cnn.save_model(save_file_name=SAVED_MODEL_NAMES[i])
            cnns.append(cnn)

    ## load trained model
    if not do_training:
        for i in range(ENSEMBLE_NUM):
            cnn = classifier_cnn.BinaryClassifierCnnWithPartNormDist()
            cnn.load_model(SAVED_MODEL_NAMES[i])
            cnns.append(cnn)

    # ####################################################
    # predicted result
    # ####################################################
    def ensemble_expec_unc(_x):
        _expecs_pred = []
        _vars_pred = []

        for i in range(ENSEMBLE_NUM):
            _y_pred = cnns[i].model.predict(_x)

            _expec = (_y_pred[:,0])[:,np.newaxis]
            _var = (_y_pred[:,1])[:,np.newaxis]

            _expecs_pred.append(_expec)
            _vars_pred.append(_var)

        _expecs_pred = np.array(_expecs_pred)
        _vars_pred = np.array(_vars_pred)


        _I = 0.5 * (sp.special.erf((1.0 - _expecs_pred) / np.sqrt(2.0 * _vars_pred)) - sp.special.erf((0.0 - _expecs_pred) / np.sqrt(2.0 * _vars_pred)))
        _f1 = spst.norm.pdf(1.0, loc=_expecs_pred, scale=np.sqrt(_vars_pred))
        _f0 = spst.norm.pdf(0.0, loc=_expecs_pred, scale=np.sqrt(_vars_pred))

        _mean_x = np.average(_expecs_pred - _vars_pred / _I * (_f1 - _f0), axis=0)
        _mean_x2 = np.average(np.square(_expecs_pred) + _vars_pred - _vars_pred / _I * ((_expecs_pred + 1) * _f1 - _expecs_pred * _f0), axis=0)

        _ave_expec = np.average(_expecs_pred, axis=0)
        _ave_std = np.sqrt(_mean_x2 - np.square(_mean_x))

        return _ave_expec, _ave_std

    # train, test, y
    y_train_cnn, unc_train_cnn = ensemble_expec_unc(dog_cat.x_train)
    y_test_cnn, unc_test_cnn = ensemble_expec_unc(dog_cat.x_test)

    # another label
    label_dict = {
        0 : 'airplane',
        1 : 'automobile',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck',
    }
    ys_another_cnn = []
    uncs_another_cnn = []
    for key in label_dict.keys():
        _prob, _unc = ensemble_expec_unc(cifar10_data.Cifar10_1Label(label=key).x_train)
        ys_another_cnn.append(_prob.flatten())
        uncs_another_cnn.append(_unc.flatten())

    # save result dir
    SAVE_RESULT_DIR = os.path.join(os.getcwd(),'result_ensemble')
    if not os.path.isdir(SAVE_RESULT_DIR):
        os.makedirs(SAVE_RESULT_DIR)

    # ######################
    # accuracy
    # ######################
    # accuracy
    def calc_acc(_y, _pre_y):
        return 1 - np.average(np.logical_xor(_y > 0.5, _pre_y > 0.5))
    def print_calc_acc(_y_train, _pre_y_train, _y_test, _pre_y_test):
        print('  train acc, test acc : {0:.3f}, {1:.3f}'.format(
                calc_acc(_y_train, _pre_y_train), calc_acc(_y_test, _pre_y_test)))
    # normal cnn train acc, test acc : 0.923, 0.863
    print_calc_acc(dog_cat.y_train, y_train_cnn, dog_cat.y_test, y_test_cnn)


    # ###############################
    # histgram of std of predicted y
    # ###############################
    def plot_std_histgram(_data_list, _label_list, save_dir):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(_data_list, label=_label_list, density=True, stacked=False)
        ax.set_title('Normalized histgram of uncertainty coef')
        ax.set_xlabel('uncertainty coef')
        ax.set_ylabel('Normalized frequency')
        ax.legend()
        #plt.show()
        fig.savefig(save_dir)
        plt.clf()

    plot_std_histgram([unc_train_cnn, unc_test_cnn], ['train', 'test'], os.path.join(SAVE_RESULT_DIR, 'hist_unc.png'))
    plot_std_histgram(uncs_another_cnn, list(label_dict.values()), os.path.join(SAVE_RESULT_DIR, 'hist_unc_another.png'))


    # ###############################
    # uncertainty vs predicted y
    # ###############################
    max_unc = 0.0
    max_unc = np.maximum(max_unc, np.max(unc_train_cnn))
    max_unc = np.maximum(max_unc, np.max(unc_test_cnn))
    for unc_another_cnn in uncs_another_cnn:
        max_unc = np.maximum(max_unc, np.max(unc_another_cnn))

    min_unc = 0.0
    min_unc = np.minimum(min_unc, np.min(unc_train_cnn))
    min_unc = np.minimum(min_unc, np.min(unc_test_cnn))
    for unc_another_cnn in uncs_another_cnn:
        min_unc = np.minimum(min_unc, np.min(unc_another_cnn))

    def plot_unc_vs_predicted_y(_uncs, _pre_ys, _labels, save_file_name):
        _fig = plt.figure()
        _ax = _fig.add_subplot(111)

        for _unc, _pre_y, _label in zip(_uncs, _pre_ys, _labels):
            _ax.scatter(_unc, _pre_y, s=10, alpha=0.1, label=_label)
        _ax.set_xlim(min_unc, max_unc)
        _ax.set_ylim(0, 1.0)
        _ax.set_title('uncertainty coef vs predicted probability')
        _ax.set_xlabel('uncertainty coef')
        _ax.set_ylabel('predicted probability')
        _ax.grid(which='major',color='black',linestyle='-')
        _ax.legend()
        #plt.show()
        _fig.savefig(save_file_name)
        plt.clf()

    plot_unc_vs_predicted_y([unc_train_cnn], [y_train_cnn], ['train'], 
                            os.path.join(SAVE_RESULT_DIR, 'unc_vs_prob_train.png'))
    plot_unc_vs_predicted_y([unc_test_cnn], [y_test_cnn], ['test'], 
                            os.path.join(SAVE_RESULT_DIR, 'unc_vs_prob_test.png'))

    for _y_ano, _unc_ano, _label in zip(ys_another_cnn, uncs_another_cnn, list(label_dict.values())):
        plot_unc_vs_predicted_y([_unc_ano], [_y_ano], [_label], 
                            os.path.join(SAVE_RESULT_DIR, 'unc_vs_prob_' + _label + '.png'))

ensemble_part_norm()


