# eval_paper_authors for evaluating the best model (best according to time static and raw filter setting)
# this requires a trained model in the checkpoint dictionary. the name needs to be adjusted accordingly in dataset_checkpoint_dict
import os

import sys
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)


device = 2
singleormultistep = 'singlestep'

# checkpoint_pth = 'Checkpoints'

dataset_list = ['ICEWS18'] #'ICEWS14', 'YAGO', 'WIKI'
setting_list = [ 'raw', 'time', 'static']

dataset_checkpoint_dict = {}
dataset_checkpoint_dict['ICEWS14']  = 'checkpoints_2022_8_3_15_49_33'
dataset_checkpoint_dict['WIKI'] = 'checkpoints_2022_8_3_15_50_32'
dataset_checkpoint_dict['YAGO'] = 'checkpoints_2022_8_10_12_4_34'
dataset_checkpoint_dict['ICEWS18'] = 'checkpoints_2022_8_10_12_9_54' 

best_epochs_dict = {}
icews14 = {}
wiki = {}
yago = {}
icews18 = {}
icews14['time'] = 8
icews14['static'] = 7
icews14['raw'] = 8
wiki['time'] = 8
wiki['static'] = 8
wiki['raw'] = 6
yago['time'] = 8
yago['static'] = 8
yago['raw'] = 9
icews18['time'] = 8
icews18['static'] = 8
icews18['raw'] = 8

best_epochs_dict['ICEWS14'] = icews14
best_epochs_dict['WIKI'] = wiki
best_epochs_dict['YAGO'] = yago
best_epochs_dict['ICEWS18'] = icews18
for dataset in dataset_list:    
    for setting in setting_list:
        best_epoch = best_epochs_dict[dataset][setting]
        checkpoint_dir = dataset_checkpoint_dict[dataset]
        print("start evaluation on test set ", checkpoint_dir, dataset, best_epoch, setting)
        os.system("python eval.py --load_checkpoint {}/checkpoint_{}.pt --whole_or_seen {} --device {} --dataset {} --setting {} --singleormultistep {}".format(checkpoint_dir,
                                best_epoch, 'whole', device, dataset, setting, singleormultistep)) #modified eval_paper_authors: added dataset


    
    
