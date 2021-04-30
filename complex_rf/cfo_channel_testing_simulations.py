'''
Real life channel and CFO experiments are done in this code.


 - Physical Layer Channel Simulation
 - Physical Layer CFO Simulation
 - Channel Equalization
 - CFO Compensation
 - Channel Augmentation
 - CFO Augmentation
'''

import numpy as np
from timeit import default_timer as timer
import argparse
# from tqdm import trange, tqdm
import json
import os
# import matplotlib as mpl
import copy
from collections import OrderedDict as odict
# import matplotlib.pyplot as plt

from keras.regularizers import l2
from keras.models import Model, load_model
from keras.layers import Dense, Input, Activation, Conv1D, Dropout, GlobalAveragePooling1D, Lambda, Average
from keras import optimizers, regularizers, losses
from keras.utils import plot_model
from keras import backend as K
import keras

from cxnn.complexnn import ComplexDense, ComplexConv1D, utils, Modrelu
# from simulators import physical_layer_channel, physical_layer_cfo, cfo_compansator, equalize_channel, augment_with_channel_test, augment_with_cfo_test, get_residual
from experiment_setup import get_arguments


def test_experiments(architecture, config):

    sample_rate = config['sample_rate']
    outfile = './data/test_dataset_2.npz'

    np_dict = np.load(outfile)
    dict_wifi = {}
    dict_wifi['x_train'] = np_dict['x_train.npy']
    dict_wifi['y_train'] = np_dict['y_train.npy']
    dict_wifi['x_validation'] = np_dict['x_val.npy']
    dict_wifi['y_validation'] = np_dict['y_val.npy']
    dict_wifi['x_test'] = np_dict['x_test.npy']
    dict_wifi['y_test'] = np_dict['y_test.npy']


    x_test_orig = dict_wifi['x_test']
    y_test_orig = dict_wifi['y_test']

    data_format = '{}-new'.format(architecture)

    num_train = dict_wifi['x_train'].shape[0]
    num_test = dict_wifi['x_test'].shape[0]
    num_classes = dict_wifi['y_train'].shape[1]

    # Checkpoint path
    checkpoint = str('./checkpoints/' + data_format)

    # if augment_channel is False:
    #     num_aug_test = 0

    print(checkpoint)

    dict_wifi_no_aug = copy.deepcopy(dict_wifi)


    print("== BUILDING MODEL... ==")

    checkpoint_in = checkpoint

    if checkpoint_in is None:
        raise ValueError('Cannot test without a checkpoint')
        # data_input = Input(batch_shape=(batch_size, num_features, 2))
        # output, model_name = network_20_2(data_input, num_classes, weight_decay)
        # densenet = Model(data_input, output)

    checkpoint_in = checkpoint_in + '.h5'
    densenet = load_model(checkpoint_in,
                          custom_objects={'ComplexConv1D': ComplexConv1D,
                                          'GetAbs': utils.GetAbs,
                                          'Modrelu': Modrelu})

    batch_size = 100

    num_test_aug = dict_wifi['x_test'].shape[0]

    output_dict = odict(acc=odict(), comp=odict(), loss=odict())

    if num_test_aug != num_test:

        num_test_per_aug = num_test_aug // num_test

        embeddings = densenet.layers[-2].output

        model2 = Model(densenet.input, embeddings)

        logits_test = model2.predict(x=dict_wifi['x_test'],
                                     batch_size=batch_size,
                                     verbose=0)

        softmax_test = densenet.predict(x=dict_wifi['x_test'],
                                        batch_size=batch_size,
                                        verbose=0)

        layer_name = densenet.layers[-1].name
        weight, bias = densenet.get_layer(layer_name).get_weights()

        logits_test = logits_test.dot(weight) + bias

        logits_test_new = np.zeros((num_test, num_classes))
        softmax_test_new = np.zeros((num_test, num_classes))
        for i in range(num_test_per_aug):
            # list_x_test.append(x_test_aug[i*num_test:(i+1)*num_test])

            logits_test_new += logits_test[i*num_test:(i+1)*num_test]
            softmax_test_new += softmax_test[i*num_test:(i+1)*num_test]

        # Adding LLRs for num_channel_aug_test test augmentations
        label_pred_llr = logits_test_new.argmax(axis=1)
        label_act = dict_wifi['y_test'][:num_test].argmax(axis=1)
        ind_correct = np.where(label_pred_llr == label_act)[0]
        ind_wrong = np.where(label_pred_llr != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc_llr = 100.*ind_correct.size / num_test

        # Adding LLRs for num_channel_aug_test test augmentations
        label_pred_soft = softmax_test_new.argmax(axis=1)
        label_act = dict_wifi['y_test'][:num_test].argmax(axis=1)
        ind_correct = np.where(label_pred_soft == label_act)[0]
        ind_wrong = np.where(label_pred_soft != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc_soft = 100.*ind_correct.size / num_test

        # 1 test augmentation
        probs = densenet.predict(x=dict_wifi['x_test'][:num_test],
                                 batch_size=batch_size,
                                 verbose=0)
        label_pred = probs.argmax(axis=1)
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc = 100.*ind_correct.size / num_test

        # No test augmentations
        probs = densenet.predict(x=dict_wifi_no_aug['x_test'],
                                 batch_size=batch_size,
                                 verbose=0)
        label_pred = probs.argmax(axis=1)
        label_act = y_test_orig.argmax(axis=1)
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
        assert (num_test == ind_wrong.size + ind_correct.size), 'Major calculation mistake!'
        test_acc_no_aug = 100.*ind_correct.size / num_test

        # print("\n========================================")
        print('Test accuracy (0 aug): {:.2f}%'.format(test_acc_no_aug))
        print('Test accuracy (1 aug): {:.2f}%'.format(test_acc))
        print('Test accuracy ({} aug) llr: {:.2f}%'.format(num_test_per_aug, test_acc_llr))
        print('Test accuracy ({} aug) softmax avg: {:.2f}%'.format(num_test_per_aug, test_acc_soft))

        output_dict['acc']['test_zero_aug'] = test_acc_no_aug
        output_dict['acc']['test_one_aug'] = test_acc
        output_dict['acc']['test_many_aug'] = test_acc_llr
        output_dict['acc']['test_many_aug_soft_avg'] = test_acc_soft

    else:
        probs = densenet.predict(x=dict_wifi['x_test'],
                                 batch_size=batch_size,
                                 verbose=0)
        label_pred = probs.argmax(axis=1)
        label_act = y_test_orig.argmax(axis=1)
        ind_correct = np.where(label_pred == label_act)[0]
        ind_wrong = np.where(label_pred != label_act)[0]
        assert (dict_wifi['x_test'].shape[0] == ind_wrong.size +
                ind_correct.size), 'Major calculation mistake!'
        test_acc_no_aug = 100.*ind_correct.size / dict_wifi['x_test'].shape[0]

        # print("\n========================================")
        print('Test accuracy (no aug): {:.2f}%'.format(test_acc_no_aug))
        output_dict['acc']['test'] = test_acc_no_aug

    return output_dict, num_test_aug // num_test


if __name__ == '__main__':

    args = get_arguments()
    architecture = args.architecture
    n_val = 5
    with open("configs_test.json") as config_file:
        config = json.load(config_file, encoding='utf-8')

    test_output, total_aug_test = test_experiments(architecture, config)

