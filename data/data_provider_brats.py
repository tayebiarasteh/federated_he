"""
Created on March 4, 2022.
data_provider_brats.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os
import torch
import pdb
import pandas as pd
import numpy as np
from skimage.io import imsave
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage.interpolation import zoom

from config.serde import read_config






class data_loader_3D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', modality=2, multimodal=True, site=None, image_resize=True):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train

        modality: int
            modality of the MR sequence
            1: T1
            2: T1Gd
            3: T2
            4: T2-FLAIR

        site: str
            name of the client for federated learning

        image_resize: bool
            if we want to have image down sampling
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.modality = int(modality)
        self.multimodal = multimodal
        self.image_resize = image_resize
        org_df = pd.read_csv(os.path.join(self.file_base_dir, "brats20_master_list.csv"), sep=',')

        if mode=='train':
            self.subset_df = org_df[org_df['soroosh_split'] == 'train']
        elif mode == 'valid':
            self.subset_df = org_df[org_df['soroosh_split'] == 'valid']
        elif mode == 'test':
            self.subset_df = org_df[org_df['soroosh_split'] == 'test']

        if not site == None:
            self.subset_df = self.subset_df[self.subset_df['site'] == site]
        self.file_path_list = list(self.subset_df['pat_num'])



    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        path_pat = os.path.join(self.file_base_dir, 'pat' + str(self.file_path_list[idx]).zfill(3))
        path_file = os.path.join(path_pat, 'pat' + str(self.file_path_list[idx]).zfill(3) + '-mod' + str(self.modality) + '.nii.gz')
        img = nib.load(path_file).get_fdata()
        img = img.astype(np.float32)  # (d, h, w)

        label_path1 = path_file.replace('-mod' + str(self.modality), '-seg-label1')
        label1 = nib.load(label_path1).get_fdata()  # (d, h, w)
        label_path2 = path_file.replace('-mod' + str(self.modality), '-seg-label2')
        label2 = nib.load(label_path2).get_fdata()  # (d, h, w)
        label_path3 = path_file.replace('-mod' + str(self.modality), '-seg-label3')
        label3 = nib.load(label_path3).get_fdata()  # (d, h, w)

        if self.multimodal:
            path_file1 = path_file.replace('-mod' + str(self.modality), '-mod1')
            img1 = nib.load(path_file1).get_fdata()  # (d, h, w)
            # normalization
            normalized_img1 = self.irm_min_max_preprocess(img1.transpose(1, 2, 0))  # (h, w, d)
            normalized_img1 = normalized_img1.transpose(2, 0, 1)  # (d, h, w)

            path_file2 = path_file.replace('-mod' + str(self.modality), '-mod2')
            img2 = nib.load(path_file2).get_fdata()  # (d, h, w)
            # normalization
            normalized_img2 = self.irm_min_max_preprocess(img2.transpose(1, 2, 0))  # (h, w, d)
            normalized_img2 = normalized_img2.transpose(2, 0, 1)  # (d, h, w)

            path_file3 = path_file.replace('-mod' + str(self.modality), '-mod3')
            img3 = nib.load(path_file3).get_fdata()  # (d, h, w)
            # normalization
            normalized_img3 = self.irm_min_max_preprocess(img3.transpose(1, 2, 0))  # (h, w, d)
            normalized_img3 = normalized_img3.transpose(2, 0, 1)  # (d, h, w)

            path_file4 = path_file.replace('-mod' + str(self.modality), '-mod4')
            img4 = nib.load(path_file4).get_fdata()  # (d, h, w)
            # normalization
            normalized_img4 = self.irm_min_max_preprocess(img4.transpose(1, 2, 0))  # (h, w, d)
            normalized_img4 = normalized_img4.transpose(2, 0, 1)  # (d, h, w)

            # image resizing for memory issues
            if self.image_resize:
                normalized_img_resized1, label1 = self.resize_manual(normalized_img1, label1)
                normalized_img_resized2, label2 = self.resize_manual(normalized_img2, label2)
                normalized_img_resized3, label3 = self.resize_manual(normalized_img3, label3)
                normalized_img_resized4, _ = self.resize_manual(normalized_img4, label3)
            else:
                if normalized_img1.shape[0] % 8 > 0:
                    remainder_d = normalized_img1.shape[0] % 8
                else:
                    remainder_d = 8
                if normalized_img1.shape[1] % 8 > 0:
                    remainder_h = normalized_img1.shape[1] % 8
                else:
                    remainder_h = 8
                if normalized_img1.shape[2] % 8 > 0:
                    remainder_w = normalized_img1.shape[2] % 8
                else:
                    remainder_w = 8
                normalized_img_resized1 = np.pad(normalized_img1, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                normalized_img_resized2 = np.pad(normalized_img2, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                normalized_img_resized3 = np.pad(normalized_img3, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                normalized_img_resized4 = np.pad(normalized_img4, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                label1 = np.pad(label1, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                label2 = np.pad(label2, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                label3 = np.pad(label3, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')

            normalized_img_resized = np.stack((normalized_img_resized1, normalized_img_resized2,
                                               normalized_img_resized3, normalized_img_resized4))  # (c=4, d, h, w)
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (c=4, d, h, w)

        else:
            # normalization
            normalized_img = self.irm_min_max_preprocess(img.transpose(1, 2, 0)) # (h, w, d)
            normalized_img = normalized_img.transpose(2, 0, 1)  # (d, h, w)

            # image resizing for memory issues
            if self.image_resize:
                normalized_img_resized, label1 = self.resize_manual(normalized_img, label1)
                _, label2 = self.resize_manual(normalized_img, label2)
                _, label3 = self.resize_manual(normalized_img, label3)
            else:
                if normalized_img.shape[0] % 8 > 0:
                    remainder_d = normalized_img.shape[0] % 8
                else:
                    remainder_d = 8
                if normalized_img.shape[1] % 8 > 0:
                    remainder_h = normalized_img.shape[1] % 8
                else:
                    remainder_h = 8
                if normalized_img.shape[2] % 8 > 0:
                    remainder_w = normalized_img.shape[2] % 8
                else:
                    remainder_w = 8
                normalized_img_resized = np.pad(normalized_img, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                label1 = np.pad(label1, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                label2 = np.pad(label2, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                label3 = np.pad(label3, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')

                normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (d, h, w)
                normalized_img_resized = torch.unsqueeze(normalized_img_resized, 0)  # (c=1, d, h, w)

        label1 = torch.from_numpy(label1)  # (d, h, w)
        label2 = torch.from_numpy(label2)  # (d, h, w)
        label3 = torch.from_numpy(label3)  # (d, h, w)
        label = torch.stack((label1, label2, label3)) # (c=3, d, h, w)

        # normalized_img_resized = normalized_img_resized.half() # float16
        normalized_img_resized = normalized_img_resized.float() # float32
        label = label.int() # int32

        return normalized_img_resized, label



    def data_normalization_mean_std(self, image):
        """subtarcting mean and std for each individual patient and modality
        mean and std only over the tumor region

        Parameters
        ----------
        image: numpy array
            The raw input image
        Returns
        -------
        normalized_img: numpy array
            The normalized image
        """
        mean = image[image > 0].mean()
        std = image[image > 0].std()

        if self.outzero_normalization:
            image[image < 0] = -1000

        normalized_img = (image - mean) / std

        if self.outzero_normalization:
            normalized_img[normalized_img < -100] = 0

        return normalized_img



    def irm_min_max_preprocess(self, image, low_perc=1, high_perc=99):
        """Main pre-processing function used for the challenge (seems to work the best).
        Remove outliers voxels first, then min-max scale.
        Warnings
        --------
        This will not do it channel wise!!
        """
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)

        min_ = np.min(image)
        max_ = np.max(image)
        scale = max_ - min_
        image = (image - min_) / scale

        return image


    def resize_manual(self, img, gt):
        """Downsampling of the image and its label.
        Parameters
        ----------
        img: numpy array
            input image
        gt: numpy array
            input label
        Returns
        -------
        img: numpy array
            downsampled image
        gt: numpy array
            downsampled label
        """
        resize_ratio = np.divide(tuple(self.params['Network']['resize_shape']), img.shape)
        img = zoom(img, resize_ratio, order=2)
        gt = zoom(gt, resize_ratio, order=0)
        return img, gt






class data_loader_without_label_3D():
    """
    This is the dataloader based on our own implementation.
    """
    def __init__(self, cfg_path, mode='test', modality=2, multimodal=True, image_resize=True):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment
        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train

        modality: int
            modality of the MR sequence
            1: T1
            2: T1Gd
            3: T2
            4: T2-FLAIR
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.modality = int(modality)
        self.multimodal = multimodal
        self.image_resize = image_resize
        org_df = pd.read_csv(os.path.join(self.file_base_dir, "brats20_master_list.csv"), sep=',')

        if mode=='train':
            self.subset_df = org_df[org_df['soroosh_split'] == 'train']
        elif mode == 'valid':
            self.subset_df = org_df[org_df['soroosh_split'] == 'valid']
        elif mode == 'test':
            self.subset_df = org_df[org_df['soroosh_split'] == 'test']

        self.file_path_list = list(self.subset_df['pat_num'])



    def provide_test_without_label(self, file_path):
        """test data provider for prediction
        Returns
        ----------
        """
        img_nifti = nib.load(file_path)
        img = img_nifti.get_fdata()
        img = img.astype(np.float32)  # (d, h, w)

        if self.multimodal:
            path_file1 = file_path.replace('-mod' + str(self.modality), '-mod1')
            img1 = nib.load(path_file1).get_fdata()  # (d, h, w)
            # normalization
            normalized_img1 = self.irm_min_max_preprocess(img1.transpose(1, 2, 0))  # (h, w, d)
            normalized_img1 = normalized_img1.transpose(2, 0, 1)  # (d, h, w)

            path_file2 = file_path.replace('-mod' + str(self.modality), '-mod2')
            img2 = nib.load(path_file2).get_fdata()  # (d, h, w)
            # normalization
            normalized_img2 = self.irm_min_max_preprocess(img2.transpose(1, 2, 0))  # (h, w, d)
            normalized_img2 = normalized_img2.transpose(2, 0, 1)  # (d, h, w)

            path_file3 = file_path.replace('-mod' + str(self.modality), '-mod3')
            img3 = nib.load(path_file3).get_fdata()  # (d, h, w)
            # normalization
            normalized_img3 = self.irm_min_max_preprocess(img3.transpose(1, 2, 0))  # (h, w, d)
            normalized_img3 = normalized_img3.transpose(2, 0, 1)  # (d, h, w)

            path_file4 = file_path.replace('-mod' + str(self.modality), '-mod4')
            img4 = nib.load(path_file4).get_fdata()  # (d, h, w)
            # normalization
            normalized_img4 = self.irm_min_max_preprocess(img4.transpose(1, 2, 0))  # (h, w, d)
            normalized_img4 = normalized_img4.transpose(2, 0, 1)  # (d, h, w)

            # image resizing for memory issues
            if self.image_resize:
                normalized_img_resized1, _ = self.resize_manual(normalized_img1, normalized_img1)
                normalized_img_resized2, _ = self.resize_manual(normalized_img2, normalized_img2)
                normalized_img_resized3, _ = self.resize_manual(normalized_img3, normalized_img3)
                normalized_img_resized4, _ = self.resize_manual(normalized_img4, normalized_img4)
                img_resized, _ = self.resize_manual(img, img)

            else:
                if normalized_img1.shape[0] % 8 > 0:
                    remainder_d = normalized_img1.shape[0] % 8
                else:
                    remainder_d = 8
                if normalized_img1.shape[1] % 8 > 0:
                    remainder_h = normalized_img1.shape[1] % 8
                else:
                    remainder_h = 8
                if normalized_img1.shape[2] % 8 > 0:
                    remainder_w = normalized_img1.shape[2] % 8
                else:
                    remainder_w = 8
                normalized_img_resized1 = np.pad(normalized_img1, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                normalized_img_resized2 = np.pad(normalized_img2, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                normalized_img_resized3 = np.pad(normalized_img3, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                normalized_img_resized4 = np.pad(normalized_img4, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')
                img = np.pad(img, [(0, 8 - remainder_d), (0, 8 - remainder_h), (0, 8 - remainder_w)], mode='constant')

            normalized_img_resized = np.stack((normalized_img_resized1, normalized_img_resized2,
                                               normalized_img_resized3, normalized_img_resized4))  # (c=4, d, h, w)
            normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (c=4, d, h, w)

        else:
            # normalization
            normalized_img = self.irm_min_max_preprocess(img.transpose(1, 2, 0)) # (h, w, d)
            normalized_img = normalized_img.transpose(2, 0, 1)  # (d, h, w)

            # image resizing for memory issues
            if normalized_img.size > 2433600:  # 100*156*156 = 2433600
                normalized_img_resized, _ = self.resize_manual(normalized_img, normalized_img)
                img_resized, _ = self.resize_manual(img, img)

            else:
                normalized_img_resized = normalized_img
                img_resized = img

                normalized_img_resized = torch.from_numpy(normalized_img_resized)  # (d, h, w)
                normalized_img_resized = torch.unsqueeze(normalized_img_resized, 0)  # (c=1, d, h, w)

        scaling_ratio = img.shape * np.array(img_nifti.header.get_zooms()) / img_resized.shape
        scaling_ratio = scaling_ratio.astype(np.float32)

        normalized_img_resized = normalized_img_resized.unsqueeze(0)  # (n=1, c, d, h, w)

        return normalized_img_resized, img_nifti, img_resized, scaling_ratio



    def irm_min_max_preprocess(self, image, low_perc=1, high_perc=99):
        """Main pre-processing function used for the challenge (seems to work the best).
        Remove outliers voxels first, then min-max scale.
        Warnings
        --------
        This will not do it channel wise!!
        """
        non_zeros = image > 0
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)

        min_ = np.min(image)
        max_ = np.max(image)
        scale = max_ - min_
        image = (image - min_) / scale

        return image


    def data_normalization_mean_std(self, image):
        """subtarcting mean and std for each individual patient and modality
        mean and std only over the tumor region
        Parameters
        ----------
        image: numpy array
            The raw input image
        Returns
        -------
        normalized_img: numpy array
            The normalized image
        """
        mean = image[image > 0].mean()
        std = image[image > 0].std()

        if self.outzero_normalization:
            image[image < 0] = -1000

        normalized_img = (image - mean) / std

        if self.outzero_normalization:
            normalized_img[normalized_img < -100] = 0

        return normalized_img


    def resize_manual(self, img, gt):
        """Downsampling of the image and its label.
        Parameters
        ----------
        img: numpy array
            input image
        gt: numpy array
            input label
        Returns
        -------
        img: numpy array
            downsampled image
        gt: numpy array
            downsampled label
        """
        resize_ratio = np.divide(tuple(self.params['Network']['resize_shape']), img.shape)
        img = zoom(img, resize_ratio, order=2)
        gt = zoom(gt, resize_ratio, order=0)
        return img, gt