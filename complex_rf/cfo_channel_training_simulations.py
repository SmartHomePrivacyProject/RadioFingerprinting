'''
export path_to_config="/home/rfml/wifi-rebuttal/wifi-fingerprinting-journal/config_cfo_channel.json"
export path_to_data="/home/rfml/wifi-rebuttal/wifi-fingerprinting-journal/data"
'''

# import matplotlib as mpl
import os
import json
# from tqdm import trange, tqdm
import argparse
from timeit import default_timer as timer
import numpy as np
import os
# import matplotlib.pyplot as plt


# from simulators import signal_power_effect, plot_signals, physical_layer_channel, physical_layer_cfo, cfo_compansator, equalize_channel, augment_with_channel, augment_with_cfo, get_residual
from cxnn.train import train_20, train_200
# from preproc.fading_model import normalize, add_custom_fading_channel, add_freq_offset
# from preproc.preproc_wifi import basic_equalize_preamble, offset_compensate_preamble
from experiment_setup import get_arguments

'''
Real life channel and CFO experiments are done in this code.


 - Physical Layer Channel Simulation
 - Physical Layer CFO Simulation
 - Channel Equalization
 - CFO Compensation
 - Channel Augmentation
 - CFO Augmentation
'''

#

def multiple_day_fingerprint(architecture, config):

    epochs = config['epochs']

    num_aug_test = config['num_aug_test']

    outfile = './data/test_dataset.npz'

    np_dict = np.load(outfile)
    dict_wifi = {}
    # import pdb
    # pdb.set_trace()
    dict_wifi['x_train'] = np_dict['x_train.npy']
    dict_wifi['y_train'] = np_dict['y_train.npy']
    dict_wifi['x_validation'] = np_dict['x_val.npy']
    dict_wifi['y_validation'] = np_dict['y_val.npy']
    dict_wifi['x_test'] = np_dict['x_test.npy']
    dict_wifi['y_test'] = np_dict['y_test.npy']
    dict_wifi['num_classes'] = dict_wifi['y_test'].shape[1]

    data_format = '{}'.format(architecture)
    sample_rate = 20
    print(data_format)
    # --------------------------------------------------------------------------------------------
    # Train
    # --------------------------------------------------------------------------------------------

    # Checkpoint path
    if not os.path.exists("./checkpoints/"):
        os.mkdir("./checkpoints/")
    checkpoint = str('./checkpoints/' + data_format)

    print(checkpoint)

    if sample_rate == 20:
        train_output, model_name, summary = train_20(dict_wifi, checkpoint_in=None,
                                                     num_aug_test=num_aug_test,
                                                     checkpoint_out=checkpoint,
                                                     architecture=architecture,
                                                     epochs=epochs)
    elif sample_rate == 200:
        train_output, model_name, summary = train_200(dict_wifi, checkpoint_in=None,
                                                      num_aug_test=num_aug_test,
                                                      checkpoint_out=checkpoint,
                                                      architecture=architecture,
                                                      epochs=epochs)
    else:
        raise NotImplementedError

    # --------------------------------------------------------------------------------------------
    # Write in log file
    # --------------------------------------------------------------------------------------------

    # Write logs
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    with open('./logs/' + data_format + '.txt', 'a+') as f:
        f.write('\n\n-----------------------\n'+str(model_name)+'\n\n')

    return train_output


if __name__ == '__main__':

    # args = get_arguments()
    #
    # architecture = args.architecture
    architecture = 'reim2x'
    with open("./configs_train.json") as config_file:
        config = json.load(config_file, encoding='utf-8')

    log_name = 'cplx_test'

    train_output = multiple_day_fingerprint(architecture, config)

    with open("./logs/" + log_name + '.txt', 'a+') as f:

        for keys, dicts in train_output.items():
            f.write(str(keys)+':\n')
            for key, value in dicts.items():
                f.write('\t'+str(key)+': {:.2f}%'.format(value)+'\n')
