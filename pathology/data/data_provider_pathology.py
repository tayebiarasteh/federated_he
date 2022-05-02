"""
Created on May 2, 2022.
data_provider_pathology.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os
import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from config.serde import read_config






class data_loader_pathology():
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', site=None):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        site: str
            name of the client for federated learning
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        file_base_dir = self.params['file_path']

        if mode=='train':
            self.directory_path = os.path.join(file_base_dir, 'train')
        elif mode == 'valid':
            self.directory_path = os.path.join(file_base_dir, 'valid')
        elif mode == 'test':
            self.directory_path = self.params['test_file_path']


    def provide_data(self):
        """
        """
        xTrain, yTrain = np.load(os.path.join(self.directory_path, 'features.npy')), np.load(os.path.join(self.directory_path, 'labels.npy'))
        pdb.set_trace()

        # xTrain, xTest = xTrain / 255.0, xTest / 255.0

        # transform numpy to torch.Tensor
        xTrain, yTrain = map(torch.tensor, (xTrain.astype(np.float32), yTrain.astype(np.int_)))

        # convert torch.Tensor to a dataset
        yTrain = yTrain.type(torch.LongTensor)

        # yTest = yTest.type(torch.LongTensor)
        trainDs = torch.utils.data.TensorDataset(xTrain, yTrain)

        return trainDs
