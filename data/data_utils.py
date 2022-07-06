"""
Created on March 7, 2022.
data_utils.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""


import numpy as np
import os
import pdb
import nibabel as nib
import pandas as pd
from tqdm import tqdm

import SimpleITK as sitk
import shutil
from multiprocessing import Pool
from collections import OrderedDict



def weight_creator(file_base_dir="/datasets/BraTS20/", label_num=2, modality=2):
    """Inverse class frequency weight creator based on the training data.
    Note that all the label files should have the same class numbers
    and the numbers should start from 0 and be integer.

    Parameters
    ----------
    label_num: int
        1: tumor core (smallest)
        2: whole tumor (biggest)
        3: Enhancing tumor

    modality: int
        modality of the MR sequence
        1: T1
        2: T1Gd
        3: T2
        4: T2-FLAIR

    Returns
    -------
    weight: list
        a list including the inverted class frequencies based on the training data
    """
    chosen_df = pd.read_csv(os.path.join(file_base_dir, "brats20_master_list.csv"), sep=',')
    chosen_df = chosen_df[chosen_df['soroosh_split'] == 'train']

    if int(modality) == 1:
        file_base_dir = os.path.join(file_base_dir, 'T1')
    elif int(modality) == 2:
        file_base_dir = os.path.join(file_base_dir, 'T1Gd')
    elif int(modality) == 3:
        file_base_dir = os.path.join(file_base_dir, 'T2')
    elif int(modality) == 4:
        file_base_dir = os.path.join(file_base_dir, 'T2-FLAIR')

    file_list = list(chosen_df['pat_num'])
    path_pat = os.path.join(file_base_dir, 'pat' + str(int(file_list[0])).zfill(3))
    label_path = os.path.join(path_pat, 'pat' + str(int(file_list[0])).zfill(3) + '-seg-label' + str(int(label_num)) + '.nii.gz')
    sums_array = np.zeros_like(np.unique(nib.load(label_path).get_fdata()))

    for idx in range(len(file_list)):

        path_pat = os.path.join(file_base_dir, 'pat' + str(int(file_list[idx])).zfill(3))
        label_path = os.path.join(path_pat, 'pat' + str(int(file_list[idx])).zfill(3) + '-seg-label' + str(int(label_num)) + '.nii.gz')
        label = nib.load(label_path).get_fdata()

        for classnum in range(len(np.unique(label))):
            sums_array[classnum] += (label == classnum).sum()

    total = sums_array.sum()

    tempweight = total / sums_array
    final_factor = sum(tempweight)
    weight = tempweight / final_factor
    print(weight)

    return weight




def mean_std_calculator(file_base_dir="/datasets/BraTS20/", modality=2):
    """

    Parameters
    ----------
    modality: int
        modality of the MR sequence
        1: T1
        2: T1Gd
        3: T2
        4: T2-FLAIR

    Returns
    -------
    weight: list
        a list including the inverted class frequencies based on the training data
    """
    chosen_df = pd.read_csv(os.path.join(file_base_dir, "brats20_master_list.csv"), sep=',')
    chosen_df = chosen_df[chosen_df['soroosh_split'] == 'train']

    if int(modality) == 1:
        file_base_dir = os.path.join(file_base_dir, 'T1')
    elif int(modality) == 2:
        file_base_dir = os.path.join(file_base_dir, 'T1Gd')
    elif int(modality) == 3:
        file_base_dir = os.path.join(file_base_dir, 'T2')
    elif int(modality) == 4:
        file_base_dir = os.path.join(file_base_dir, 'T2-FLAIR')

    file_list = list(chosen_df['pat_num'])
    stackk = np.array([])

    maxx = 0
    minn = 0
    for idx in tqdm(range(len(file_list))):

        path_pat = os.path.join(file_base_dir, 'pat' + str(int(file_list[idx])).zfill(3))
        img_path = os.path.join(path_pat, 'pat' + str(int(file_list[idx])).zfill(3) + '-mod' + str(modality) + '.nii.gz')
        img = nib.load(img_path).get_fdata()

        stackk = np.hstack((stackk, img[img > 0 ]))
        # stackk = np.hstack((stackk, img.flatten()))

        final_mean = stackk.mean()
        final_std = stackk.std()
        print(final_mean, final_std)
        # if maxx < img.max():
        #     maxx = img.max()
        # if minn > img.min():
        #     minn = img.min()
        # print(maxx, minn)

    return final_mean, final_std




