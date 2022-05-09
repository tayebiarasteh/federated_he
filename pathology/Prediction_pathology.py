"""
Created on May 4, 2022.
Prediction_pathology.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os.path
import numpy as np
import torchmetrics
from tqdm import tqdm
import torch.nn.functional as F

from config.serde import read_config

epsilon = 1e-15



class Prediction:
    def __init__(self, cfg_path):
        """
        This class represents prediction (testing) process similar to the Training class.
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda()


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.
        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def setup_model(self, model, model_file_name=None, epoch=1):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)

        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], model_file_name)))
        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch25_" + model_file_name))
        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch" + str(epoch) + "_" + model_file_name))



    def predict(self, test_loader):
        """Evaluation with metrics epoch
        """
        self.model.eval()

        pred_array = []
        target_array = []

        for idx, (image, label) in enumerate(test_loader):

            with torch.no_grad():
                image, label = image.to(self.device), label.to(self.device)
                output = self.model(image)
                pred_array.append(output)
                target_array.append(label)

        pred_array = torch.cat(pred_array, dim=0)
        target_array = torch.cat(target_array, dim=0)

        return pred_array, target_array
