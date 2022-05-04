"""
Created on May 2, 2022.
data_provider_pathology.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os
import torch
import pdb
import numpy as np
from torch.utils.data import Dataset

from config.serde import read_config




class data_loader_pathology():
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', site=None, benchmark='QUASAR_deployMSIH', fold=1):
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
        self.fold = fold
        self.mode = mode

        if self.mode =='train':
            self.directory_path = os.path.join(file_base_dir, 'train')
            if not site == None:
                self.directory_path = os.path.join(self.directory_path, site)
            else:
                self.directory_path = os.path.join(self.directory_path, 'central')
        elif self.mode == 'valid':
            self.directory_path = os.path.join(file_base_dir, 'valid')
        elif self.mode == 'test':
            self.directory_path = os.path.join(file_base_dir, 'test', benchmark)



    def provide_data(self):
        """
        """
        if self.mode == 'test':
            xfile, yfile = np.load(os.path.join(self.directory_path, 'features.npy')), np.load(os.path.join(self.directory_path, 'labels.npy'))
        else:
            xfile, yfile = np.load(os.path.join(self.directory_path, 'features_fold' + str(self.fold) + '.npy')), np.load(os.path.join(self.directory_path, 'labels_fold' + str(self.fold) + '.npy'))

        # transform numpy to torch.Tensor
        xfile, yfile = map(torch.tensor, (xfile.astype(np.float32), yfile.astype(np.int_)))

        # convert torch.Tensor to a dataset
        xfile = xfile.float()

        output_dataset = torch.utils.data.TensorDataset(xfile, yfile)

        return output_dataset
