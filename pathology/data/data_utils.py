"""
Created on May 3, 2022.
data_utils.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""


import numpy as np
import os
import pdb
import pandas as pd
from tqdm import tqdm
import random




def center_concatenator():
    path = '/PATH'

    img_output_path = '/PATH/features.npy'
    label_output_path = '/PATH/labels.npy'

    belfast_img = np.load(os.path.join(path, 'belfast', 'features.npy'))
    belfast_label = np.load(os.path.join(path, 'belfast', 'labels.npy'))

    DACHS_img = np.load(os.path.join(path, 'DACHS', 'features.npy'))
    DACHS_label = np.load(os.path.join(path, 'DACHS', 'labels.npy'))

    TCGA_img = np.load(os.path.join(path, 'TCGA', 'features.npy'))
    TCGA_label = np.load(os.path.join(path, 'TCGA', 'labels.npy'))

    img = np.concatenate((belfast_img, DACHS_img, TCGA_img), axis=0)
    label = np.concatenate((belfast_label, DACHS_label, TCGA_label), axis=0)

    np.save(img_output_path, img)
    np.save(label_output_path, label)



def divider_train_valid(split_percentage=10):
    """
    """
    path = '/PATH/'
    dataset_name_list = ['belfast', 'DACHS', 'TCGA']
    fold_number = 5

    final_train_img_list = []
    final_train_label_list = []
    final_valid_img_list = []
    final_valid_label_list = []

    for dataset_name in dataset_name_list:

        img = np.load(os.path.join(path, 'train', dataset_name, 'original_full', 'features.npy'))
        label = np.load(os.path.join(path, 'train', dataset_name, 'original_full', 'labels.npy'))
        index_list = random.sample(range(0, label.shape[0]), label.shape[0] // split_percentage)
        valid_label = np.zeros(len(index_list))
        valid_img = np.zeros((len(index_list), 512))
        train_label = []
        train_img = []

        # new valid array
        for idx, number in enumerate(index_list):
            valid_img[idx] = img[number]
            valid_label[idx] = label[number]

        # new train array
        for number in range(len(label)):
            if not number in index_list:
                train_img.append(img[number])
                train_label.append(label[number])
        train_img = np.stack(train_img, 0)
        train_label = np.stack(train_label, 0)

        np.save(os.path.join(path, 'train', dataset_name, 'features_fold' + str(fold_number) + '.npy'), train_img)
        np.save(os.path.join(path, 'train', dataset_name, 'labels_fold' + str(fold_number) + '.npy'), train_label)

        os.makedirs(os.path.join(path, 'valid/fold' + str(fold_number)), exist_ok=True)
        np.save(os.path.join(path, 'valid/fold' + str(fold_number), dataset_name + '_features.npy'), valid_img)
        np.save(os.path.join(path, 'valid/fold' + str(fold_number), dataset_name + '_labels.npy'), valid_label)

        final_train_img_list.append(np.copy(train_img))
        final_train_label_list.append(np.copy(train_label))
        final_valid_img_list.append(np.copy(valid_img))
        final_valid_label_list.append(np.copy(valid_label))

    final_train_img = np.concatenate((final_train_img_list[0], final_train_img_list[1], final_train_img_list[2]), axis=0)
    final_train_label = np.concatenate((final_train_label_list[0], final_train_label_list[1], final_train_label_list[2]), axis=0)
    np.save(os.path.join(path, 'train/central', 'features_fold' + str(fold_number) + '.npy'), final_train_img)
    np.save(os.path.join(path, 'train/central', 'labels_fold' + str(fold_number) + '.npy'), final_train_label)

    final_valid_img = np.concatenate((final_valid_img_list[0], final_valid_img_list[1], final_valid_img_list[2]), axis=0)
    final_valid_label = np.concatenate((final_valid_label_list[0], final_valid_label_list[1], final_valid_label_list[2]), axis=0)
    np.save(os.path.join(path, 'valid', 'features_fold' + str(fold_number) + '.npy'), final_valid_img)
    np.save(os.path.join(path, 'valid', 'labels_fold' + str(fold_number) + '.npy'), final_valid_label)



def split_csv():

    path = '/PATH/YORKSHIR_deployMSIH/FULL_TEST_split.csv'
    df = pd.read_csv(path, sep=',')

    final_df_path = '/PATH/YORKSHIR_deployMSIH/org_split.csv'
    final_df = pd.DataFrame(columns=['pat_name', 'label', 'start_index', 'end_index'])

    patient_list = list(df['patientID'].unique())

    for patient in tqdm(patient_list):
        indexlist = df[df['patientID'] == patient].index
        assert indexlist[-1] == indexlist.max()
        assert indexlist[0] == indexlist.min()
        final_df = final_df.append(pd.DataFrame([[patient, df[df['patientID'] == patient]['y'].values[0], indexlist[0], indexlist[-1]]],
                     columns=['pat_name', 'label', 'start_index', 'end_index']))

    final_df.to_csv(final_df_path, sep=',', index=True)



if __name__=='__main__':
    # weight = center_concatenator()
    divider_train_valid()
    # split_csv()
