"""
Created on March 8, 2022.
Prediction_brats.py

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
import torchio as tio
import nibabel as nib

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


    def setup_model(self, model, model_file_name=None):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)

        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], model_file_name)))
        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch24_" + model_file_name))



    def evaluate_3D(self, test_loader):
        """Evaluation with metrics epoch
        Returns
        -------
        epoch_f1_score: float
            average test F1 score
        average_specifity: float
            average test specifity
        average_sensitivity: float
            average test sensitivity
        average_precision: float
            average test precision
        """
        self.model.eval()
        total_f1_score = []
        total_accuracy = []
        total_specifity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        for idx, (image, label) in enumerate(tqdm(test_loader)):
            label = label.long()
            image = image.float()
            image = image.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                output_sigmoided = F.sigmoid(output.permute(0, 2, 3, 4, 1))
                output_sigmoided = (output_sigmoided > 0.5).float()

            ############ Evaluation metric calculation ########
            # Metrics calculation (macro) over the whole set
            confusioner = torchmetrics.ConfusionMatrix(num_classes=label.shape[1], multilabel=True).to(self.device)
            confusion = confusioner(output_sigmoided.flatten(start_dim=0, end_dim=3),
                                    label.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=3))

            F1_disease = []
            accuracy_disease = []
            specifity_disease = []
            sensitivity_disease = []
            precision_disease = []

            for idx, disease in enumerate(confusion):
                TN = disease[0, 0]
                FP = disease[0, 1]
                FN = disease[1, 0]
                TP = disease[1, 1]
                F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
                accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
                specifity_disease.append(TN / (TN + FP + epsilon))
                sensitivity_disease.append(TP / (TP + FN + epsilon))
                precision_disease.append(TP / (TP + FP + epsilon))

            # Macro averaging
            total_f1_score.append(torch.stack(F1_disease))
            total_accuracy.append(torch.stack(accuracy_disease))
            total_specifity_score.append(torch.stack(specifity_disease))
            total_sensitivity_score.append(torch.stack(sensitivity_disease))
            total_precision_score.append(torch.stack(precision_disease))

        average_f1_score = torch.stack(total_f1_score).mean(0)
        average_accuracy = torch.stack(total_accuracy).mean(0)
        average_specifity = torch.stack(total_specifity_score).mean(0)
        average_sensitivity = torch.stack(total_sensitivity_score).mean(0)
        average_precision = torch.stack(total_precision_score).mean(0)

        return average_f1_score, average_accuracy, average_specifity, average_sensitivity, average_precision



    def evaluate_3D_tta(self, test_loader):
        """Evaluation with metrics epoch and applying test-time augmentation

        Returns
        -------
        epoch_f1_score: float
            average test F1 score

        average_specifity: float
            average test specifity

        average_sensitivity: float
            average test sensitivity

        average_precision: float
            average test precision
        """
        self.model.eval()
        total_f1_score = []
        total_accuracy = []
        total_specifity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        for idx, (image, label) in enumerate(tqdm(test_loader)):

            label = label.long()
            image = image.float()

            with torch.no_grad():

                output_normal = self.model(image.to(self.device))
                output_normal = output_normal.cpu()

                # augmentation
                transformed_image, transform = self.tta_performer(image, 'lateral_flip')
                transformed_image = transformed_image.to(self.device)
                output = self.model(transformed_image)
                output_back1 = transform(output[0].cpu())
                output_back1 = output_back1.unsqueeze(0)

                # augmentation
                transformed_image, transform = self.tta_performer(image, 'interior_flip')
                transformed_image = transformed_image.to(self.device)
                output = self.model(transformed_image)
                output_back5 = transform(output[0].cpu())
                output_back5 = output_back5.unsqueeze(0)

                # augmentation
                transformed_image, transform = self.tta_performer(image, 'AWGN')
                transformed_image = transformed_image.to(self.device)
                output_back2 = self.model(transformed_image)
                output_back2 = output_back2.cpu()

                # augmentation
                transformed_image, transform = self.tta_performer(image, 'gamma')
                transformed_image = transformed_image.to(self.device)
                output_back3 = self.model(transformed_image)
                output_back3 = output_back3.cpu()

                # # augmentation
                # transformed_image, transform = self.tta_performer(image, 'blur')
                # transformed_image = transformed_image.to(self.device)
                # output_back4 = self.model(transformed_image)
                # output_back4 = output_back4.cpu()

                # ensembling the predictions
                output = (output_normal + output_normal + output_back1 + output_back2 +
                          output_back3 ) / 5

                output = output.to(self.device)

                output_sigmoided = F.sigmoid(output.permute(0, 2, 3, 4, 1))
                output_sigmoided = (output_sigmoided > 0.5).float()

                label = label.to(self.device)

            ############ Evaluation metric calculation ########
            # Metrics calculation (macro) over the whole set
            confusioner = torchmetrics.ConfusionMatrix(num_classes=label.shape[1], multilabel=True).to(self.device)
            confusion = confusioner(output_sigmoided.flatten(start_dim=0, end_dim=3),
                                    label.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=3))

            F1_disease = []
            accuracy_disease = []
            specifity_disease = []
            sensitivity_disease = []
            precision_disease = []

            for idx, disease in enumerate(confusion):
                TN = disease[0, 0]
                FP = disease[0, 1]
                FN = disease[1, 0]
                TP = disease[1, 1]
                F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
                accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
                specifity_disease.append(TN / (TN + FP + epsilon))
                sensitivity_disease.append(TP / (TP + FN + epsilon))
                precision_disease.append(TP / (TP + FP + epsilon))

            # Macro averaging
            total_f1_score.append(torch.stack(F1_disease))
            total_accuracy.append(torch.stack(accuracy_disease))
            total_specifity_score.append(torch.stack(specifity_disease))
            total_sensitivity_score.append(torch.stack(sensitivity_disease))
            total_precision_score.append(torch.stack(precision_disease))

        average_f1_score = torch.stack(total_f1_score).mean(0)
        average_accuracy = torch.stack(total_accuracy).mean(0)
        average_specifity = torch.stack(total_specifity_score).mean(0)
        average_sensitivity = torch.stack(total_sensitivity_score).mean(0)
        average_precision = torch.stack(total_precision_score).mean(0)

        return average_f1_score, average_accuracy, average_specifity, average_sensitivity, average_precision



    def predict_3D(self, image):
        """Prediction of one signle image

        Returns
        -------
        """
        self.model.eval()

        image = image.float()
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            output_sigmoided = F.sigmoid(output)

        return output_sigmoided



    def predict_3D_tta(self, image):
        """Prediction of one signle image using test-time augmentation

        Returns
        -------
        """
        self.model.eval()

        image = image.float()

        with torch.no_grad():
            output_normal = self.model(image.to(self.device))
            output_normal = output_normal.cpu()

            # augmentation lateral flip
            transformed_image, transform = self.tta_performer(image, 'lateral_flip')
            transformed_image = transformed_image.to(self.device)
            output = self.model(transformed_image)
            output_back1 = transform(output[0].cpu())
            output_back1 = output_back1.unsqueeze(0)

            # augmentation interior flip
            transformed_image, transform = self.tta_performer(image, 'interior_flip')
            transformed_image = transformed_image.to(self.device)
            output = self.model(transformed_image)
            output_back2 = transform(output[0].cpu())
            output_back2 = output_back2.unsqueeze(0)

            # augmentation lateral & interior flip
            transformed_image, transform = self.tta_performer(image, 'lateral_flip')
            transformed_image = transformed_image.to(self.device)
            output = self.model(transformed_image)
            output_back5 = transform(output[0].cpu())
            output_back5 = output_back5.unsqueeze(0)
            transformed_image, transform = self.tta_performer(output_back5, 'interior_flip')
            transformed_image = transformed_image.to(self.device)
            output = self.model(transformed_image)
            output_back5 = transform(output[0].cpu())
            output_back5 = output_back5.unsqueeze(0)

            # augmentation
            transformed_image, transform = self.tta_performer(image, 'AWGN')
            transformed_image = transformed_image.to(self.device)
            output_back3 = self.model(transformed_image)
            output_back3 = output_back3.cpu()

            # augmentation
            transformed_image, transform = self.tta_performer(image, 'gamma')
            transformed_image = transformed_image.to(self.device)
            output_back4 = self.model(transformed_image)
            output_back4 = output_back4.cpu()

            # ensembling the predictions
            output = (output_normal + output_normal + output_back1 + output_back2 +
                      output_back3 + output_back4 + output_back5) / 7

            output = output.to(self.device)

            output_sigmoided = F.sigmoid(output)

        return output_sigmoided


    def tta_performer(self, image, transform_type):
        """applying test-time augmentation
        """

        if transform_type == 'lateral_flip':
            transform = tio.transforms.RandomFlip(axes='L', flip_probability=1)

        if transform_type == 'interior_flip':
            transform = tio.transforms.RandomFlip(axes='I', flip_probability=1)

        elif transform_type == 'AWGN':
            transform = tio.RandomNoise(mean=self.params['augmentation']['mu_AWGN'], std=self.params['augmentation']['sigma_AWGN'])

        elif transform_type == 'gamma':
            transform = tio.RandomGamma(log_gamma=(self.params['augmentation']['gamma_range'][0], self.params['augmentation']['gamma_range'][1]))

        elif transform_type == 'blur':
            transform = tio.RandomBlur(std=(self.params['augmentation']['gamma_range'][0], self.params['augmentation']['gamma_range'][1]))

        # normalized_img = nib.Nifti1Image(image[0,0].numpy(), np.eye(4))
        # nib.save(normalized_img, 'orggg.nii.gz')

        trans_img = transform(image[0])
        # normalized_img = nib.Nifti1Image(trans_img[0].numpy(), np.eye(4))
        # nib.save(normalized_img, 'tta_img.nii.gz')

        # transform = tio.RandomAffine(scales=(1.05, 1.05), translation=0, degrees=0, default_pad_value='minimum',
        #                              image_interpolation='nearest')
        # image = transform(trans_img)
        # normalized_img = nib.Nifti1Image(image[0].numpy(), np.eye(4))
        # nib.save(normalized_img, 'tta_img_back.nii.gz')

        # pdb.set_trace()

        return trans_img.unsqueeze(0), transform
