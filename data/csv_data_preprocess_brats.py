"""
Created on March 4, 2022.
csv_data_preprocess_brats.py

creating a master list for Brats dataset.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import nibabel as nib
import random
from math import ceil

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')




class csv_preprocess_brats():
    def __init__(self, cfg_path="/federated_he/config/config.yaml"):
        self.params = read_config(cfg_path)

    def hd5_to_nifti(self):
        """Converting HD5 data to nifti
        we have 4 MRI modalities and 3 labels per patient
        """

        # filee = "/PATH.h5"
        org_df = "/PATH/meta_data.csv"
        target_base = "/datasets/BraTS20/"
        base_dir = '/PATH'

        df = pd.read_csv(org_df, sep=',')
        patient_list = df['volume'].unique().tolist()


        for patient in tqdm(patient_list):
            df_pat = df[df['volume'] == patient]
            volume_mod1 = []
            volume_mod2 = []
            volume_mod3 = []
            volume_mod4 = []
            volume_mask1 = []
            volume_mask2 = []
            volume_mask3 = []

            for i in range(len(df_pat)):
                rel_path = df_pat[df_pat['slice'] == i]['slice_path'].values[0]
                path = os.path.join(base_dir, os.path.basename(rel_path))
                hf = h5py.File(path, 'r')
                volume_mod1.append(hf['image'][:, :, 0])
                volume_mod2.append(hf['image'][:, :, 1])
                volume_mod3.append(hf['image'][:, :, 2])
                volume_mod4.append(hf['image'][:, :, 3])
                volume_mask1.append(hf['mask'][:, :, 0])
                volume_mask2.append(hf['mask'][:, :, 1])
                volume_mask3.append(hf['mask'][:, :, 2])

            volume_mod1 = np.stack(volume_mod1) # (d, h, w)
            volume_mod2 = np.stack(volume_mod2) # (d, h, w)
            volume_mod3 = np.stack(volume_mod3) # (d, h, w)
            volume_mod4 = np.stack(volume_mod4) # (d, h, w)
            volume_mask1 = np.stack(volume_mask1) # (d, h, w)
            volume_mask2 = np.stack(volume_mask2) # (d, h, w)
            volume_mask3 = np.stack(volume_mask3) # (d, h, w)

            input_img = nib.Nifti1Image(volume_mod1, np.eye(4))
            os.makedirs(os.path.join(target_base, 'T1', 'pat' + str(patient).zfill(3)), exist_ok=True)
            nib.save(input_img, os.path.join(target_base, 'T1', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-mod1.nii.gz'))

            input_img = nib.Nifti1Image(volume_mod2, np.eye(4))
            os.makedirs(os.path.join(target_base, 'T1Gd', 'pat' + str(patient).zfill(3)), exist_ok=True)
            nib.save(input_img, os.path.join(target_base, 'T1Gd', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-mod2.nii.gz'))

            input_img = nib.Nifti1Image(volume_mod3, np.eye(4))
            os.makedirs(os.path.join(target_base, 'T2', 'pat' + str(patient).zfill(3)), exist_ok=True)
            nib.save(input_img, os.path.join(target_base, 'T2', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-mod3.nii.gz'))

            input_img = nib.Nifti1Image(volume_mod4, np.eye(4))
            os.makedirs(os.path.join(target_base, 'T2-FLAIR', 'pat' + str(patient).zfill(3)), exist_ok=True)
            nib.save(input_img, os.path.join(target_base, 'T2-FLAIR', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-mod4.nii.gz'))

            input_img = nib.Nifti1Image(volume_mask1, np.eye(4))
            nib.save(input_img, os.path.join(target_base, 'T1', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label1.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T1Gd', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label1.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T2', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label1.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T2-FLAIR', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label1.nii.gz'))

            input_img = nib.Nifti1Image(volume_mask2, np.eye(4))
            nib.save(input_img, os.path.join(target_base, 'T1', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label2.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T1Gd', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label2.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T2', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label2.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T2-FLAIR', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label2.nii.gz'))

            input_img = nib.Nifti1Image(volume_mask3, np.eye(4))
            nib.save(input_img, os.path.join(target_base, 'T1', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label3.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T1Gd', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label3.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T2', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label3.nii.gz'))
            nib.save(input_img, os.path.join(target_base, 'T2-FLAIR', 'pat' + str(patient).zfill(3), 'pat' + str(patient).zfill(3) + '-seg-label3.nii.gz'))


    def csv_divider_train_valid_test(self, num_clients=3):
        """

        Parameters
        ----------
        ratio: float
            ratio of dividing to train and valid/test
            0.1 means 10% valid, 10% test, 80% train

        num_clients: int
            number of federated clients for training
        """

        path = '/PATH/brats20_master_list.csv'
        output_df_path = '/PATH/' + str(
            num_clients) + '_clients/brats20_master_list.csv'
        os.makedirs(os.path.dirname(output_df_path), exist_ok=True)

        # initiating valid and train dfs
        final_train_data = pd.DataFrame(
            columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                     'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age', 'Survival_days',
                     'Extent_of_Resection', 'Grade'])
        final_valid_data = pd.DataFrame(
            columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                     'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age', 'Survival_days',
                     'Extent_of_Resection', 'Grade'])
        final_test_data = pd.DataFrame(
            columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                     'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age', 'Survival_days',
                     'Extent_of_Resection', 'Grade'])

        final_all_data = pd.DataFrame(
            columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                     'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age', 'Survival_days',
                     'Extent_of_Resection', 'Grade'])

        df = pd.read_csv(path, sep=',')

        subject_list = df['BraTS_2020_subject_ID'].unique().tolist()
        random.shuffle(subject_list)
        # val_num = ceil(len(subject_list) / (1 / ratio))

        test_subjects = subject_list[:35]
        valid_subjects = subject_list[35:69]
        train_subjects = subject_list[69:]

        # adding files to train
        for subject in train_subjects:
            selected_df = df[df['BraTS_2020_subject_ID'] == subject]
            tempp = pd.DataFrame([[selected_df['pat_num'].values[0], 'train', selected_df['site'].values[0],
                                   selected_df['BraTS_2020_subject_ID'].values[0],
                                   selected_df['BraTS_2017_subject_ID'].values[0],
                                   selected_df['BraTS_2018_subject_ID'].values[0],
                                   selected_df['TCGA_TCIA_subject_ID'].values[0],
                                   selected_df['BraTS_2019_subject_ID'].values[0], selected_df['Age'].values[0],
                                   selected_df['Survival_days'].values[0], selected_df['Extent_of_Resection'].values[0],
                                   selected_df['Grade'].values[0]]],
                                 columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID',
                                          'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID',
                                          'BraTS_2019_subject_ID', 'Age', 'Survival_days', 'Extent_of_Resection',
                                          'Grade'])
            final_train_data = final_train_data.append(tempp)

        # adding files to valid
        for subject in valid_subjects:
            selected_df = df[df['BraTS_2020_subject_ID'] == subject]
            tempp = pd.DataFrame([[selected_df['pat_num'].values[0], 'valid', 'site-valid',
                                   selected_df['BraTS_2020_subject_ID'].values[0],
                                   selected_df['BraTS_2017_subject_ID'].values[0],
                                   selected_df['BraTS_2018_subject_ID'].values[0],
                                   selected_df['TCGA_TCIA_subject_ID'].values[0],
                                   selected_df['BraTS_2019_subject_ID'].values[0], selected_df['Age'].values[0],
                                   selected_df['Survival_days'].values[0], selected_df['Extent_of_Resection'].values[0],
                                   selected_df['Grade'].values[0]]],
                                 columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID',
                                          'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID',
                                          'BraTS_2019_subject_ID', 'Age', 'Survival_days', 'Extent_of_Resection',
                                          'Grade'])
            final_valid_data = final_valid_data.append(tempp)

        # adding files to test
        for subject in test_subjects:
            selected_df = df[df['BraTS_2020_subject_ID'] == subject]
            tempp = pd.DataFrame([[selected_df['pat_num'].values[0], 'test', 'site-test',
                                   selected_df['BraTS_2020_subject_ID'].values[0],
                                   selected_df['BraTS_2017_subject_ID'].values[0],
                                   selected_df['BraTS_2018_subject_ID'].values[0],
                                   selected_df['TCGA_TCIA_subject_ID'].values[0],
                                   selected_df['BraTS_2019_subject_ID'].values[0], selected_df['Age'].values[0],
                                   selected_df['Survival_days'].values[0], selected_df['Extent_of_Resection'].values[0],
                                   selected_df['Grade'].values[0]]],
                                 columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID',
                                          'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID',
                                          'BraTS_2019_subject_ID', 'Age', 'Survival_days', 'Extent_of_Resection',
                                          'Grade'])
            final_test_data = final_test_data.append(tempp)

        per_client_images_num = ceil(len(final_train_data) / num_clients)

        subject_list = final_train_data['BraTS_2020_subject_ID'].unique().tolist()
        random.shuffle(subject_list)

        client_list = []
        for idx in range(num_clients):
            client_list.append(subject_list[idx * per_client_images_num:(idx + 1) * per_client_images_num])

        # initializing client dfs
        final_client_train_data_list = []
        for idx in range(num_clients):
            final_client_train_data_list.append(pd.DataFrame(
                columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                         'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age',
                         'Survival_days',
                         'Extent_of_Resection', 'Grade']))

        # adding files to clients
        for idx, client in enumerate(client_list):
            for subject in client:
                selected_df = final_train_data[final_train_data['BraTS_2020_subject_ID'] == subject]
                tempp = pd.DataFrame([[selected_df['pat_num'].values[0], 'train', 'site-' + str(idx + 1),
                                       selected_df['BraTS_2020_subject_ID'].values[0],
                                       selected_df['BraTS_2017_subject_ID'].values[0],
                                       selected_df['BraTS_2018_subject_ID'].values[0],
                                       selected_df['TCGA_TCIA_subject_ID'].values[0],
                                       selected_df['BraTS_2019_subject_ID'].values[0], selected_df['Age'].values[0],
                                       selected_df['Survival_days'].values[0],
                                       selected_df['Extent_of_Resection'].values[0], selected_df['Grade'].values[0]]],
                                     columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID',
                                              'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID',
                                              'BraTS_2019_subject_ID', 'Age', 'Survival_days', 'Extent_of_Resection',
                                              'Grade'])
                final_client_train_data_list[idx] = final_client_train_data_list[idx].append(tempp)

        # adding files to clients
        for idx, client in enumerate(final_client_train_data_list):
            final_all_data = final_all_data.append(client)

        final_all_data = final_all_data.append(final_valid_data)
        final_all_data = final_all_data.append(final_test_data)

        final_all_data = final_all_data.sort_values(['pat_num'])

        final_all_data.to_csv(output_df_path, sep=',', index=False)




    def csv_divider_train_valid(self, num_clients=3):
        """

        Parameters
        ----------
        ratio: float
            ratio of dividing to train and valid/test
            0.1 means 10% valid, 10% test, 80% train

        num_clients: int
            number of federated clients for training
        """

        path = '/PATH/brats20_master_list.csv'
        output_df_path = '/PATH/' + str(
            num_clients) + '_clients/brats20_master_list.csv'
        os.makedirs(os.path.dirname(output_df_path), exist_ok=True)

        # initiating valid and train dfs
        final_train_data = pd.DataFrame(
            columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                     'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age', 'Survival_days',
                     'Extent_of_Resection', 'Grade'])
        final_valid_data = pd.DataFrame(
            columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                     'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age', 'Survival_days',
                     'Extent_of_Resection', 'Grade'])
        final_all_data = pd.DataFrame(
            columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                     'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age', 'Survival_days',
                     'Extent_of_Resection', 'Grade'])

        df = pd.read_csv(path, sep=',')

        subject_list = df['BraTS_2020_subject_ID'].unique().tolist()
        random.shuffle(subject_list)
        # val_num = ceil(len(subject_list) / (1 / ratio))

        valid_subjects = subject_list[:39]
        train_subjects = subject_list[39:]

        # adding files to train
        for subject in train_subjects:
            selected_df = df[df['BraTS_2020_subject_ID'] == subject]
            tempp = pd.DataFrame([[selected_df['pat_num'].values[0], 'train', selected_df['site'].values[0],
                                   selected_df['BraTS_2020_subject_ID'].values[0],
                                   selected_df['BraTS_2017_subject_ID'].values[0],
                                   selected_df['BraTS_2018_subject_ID'].values[0],
                                   selected_df['TCGA_TCIA_subject_ID'].values[0],
                                   selected_df['BraTS_2019_subject_ID'].values[0], selected_df['Age'].values[0],
                                   selected_df['Survival_days'].values[0], selected_df['Extent_of_Resection'].values[0],
                                   selected_df['Grade'].values[0]]],
                                 columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID',
                                          'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID',
                                          'BraTS_2019_subject_ID', 'Age', 'Survival_days', 'Extent_of_Resection',
                                          'Grade'])
            final_train_data = final_train_data.append(tempp)

        # adding files to valid
        for subject in valid_subjects:
            selected_df = df[df['BraTS_2020_subject_ID'] == subject]
            tempp = pd.DataFrame([[selected_df['pat_num'].values[0], 'valid', 'site-valid',
                                   selected_df['BraTS_2020_subject_ID'].values[0],
                                   selected_df['BraTS_2017_subject_ID'].values[0],
                                   selected_df['BraTS_2018_subject_ID'].values[0],
                                   selected_df['TCGA_TCIA_subject_ID'].values[0],
                                   selected_df['BraTS_2019_subject_ID'].values[0], selected_df['Age'].values[0],
                                   selected_df['Survival_days'].values[0], selected_df['Extent_of_Resection'].values[0],
                                   selected_df['Grade'].values[0]]],
                                 columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID',
                                          'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID',
                                          'BraTS_2019_subject_ID', 'Age', 'Survival_days', 'Extent_of_Resection',
                                          'Grade'])
            final_valid_data = final_valid_data.append(tempp)

        per_client_images_num = ceil(len(final_train_data) / num_clients)

        subject_list = final_train_data['BraTS_2020_subject_ID'].unique().tolist()
        random.shuffle(subject_list)

        client_list = []
        for idx in range(num_clients):
            client_list.append(subject_list[idx * per_client_images_num:(idx + 1) * per_client_images_num])

        # initializing client dfs
        final_client_train_data_list = []
        for idx in range(num_clients):
            final_client_train_data_list.append(pd.DataFrame(
                columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID', 'BraTS_2017_subject_ID',
                         'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID', 'BraTS_2019_subject_ID', 'Age',
                         'Survival_days',
                         'Extent_of_Resection', 'Grade']))

        # adding files to clients
        for idx, client in enumerate(client_list):
            for subject in client:
                selected_df = final_train_data[final_train_data['BraTS_2020_subject_ID'] == subject]
                tempp = pd.DataFrame([[selected_df['pat_num'].values[0], 'train', 'site-' + str(idx + 1),
                                       selected_df['BraTS_2020_subject_ID'].values[0],
                                       selected_df['BraTS_2017_subject_ID'].values[0],
                                       selected_df['BraTS_2018_subject_ID'].values[0],
                                       selected_df['TCGA_TCIA_subject_ID'].values[0],
                                       selected_df['BraTS_2019_subject_ID'].values[0], selected_df['Age'].values[0],
                                       selected_df['Survival_days'].values[0],
                                       selected_df['Extent_of_Resection'].values[0], selected_df['Grade'].values[0]]],
                                     columns=['pat_num', 'soroosh_split', 'site', 'BraTS_2020_subject_ID',
                                              'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID', 'TCGA_TCIA_subject_ID',
                                              'BraTS_2019_subject_ID', 'Age', 'Survival_days', 'Extent_of_Resection',
                                              'Grade'])
                final_client_train_data_list[idx] = final_client_train_data_list[idx].append(tempp)

        # adding files to clients
        for idx, client in enumerate(final_client_train_data_list):
            final_all_data = final_all_data.append(client)

        final_all_data = final_all_data.append(final_valid_data)

        final_all_data = final_all_data.sort_values(['pat_num'])

        final_all_data.to_csv(output_df_path, sep=',', index=False)



class cropper():
    def __init__(self, cfg_path="/federated_he/config/config.yaml"):
        """
        Cropping the all the images and segmentations around the brain
        Parameters
        ----------
        cfg_path
        """
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        org_df = pd.read_csv(os.path.join(self.file_base_dir, "brats20_master_list.csv"), sep=',')
        # self.df = org_df[org_df['pat_num'] > 72]
        self.df = org_df[org_df['soroosh_split'] == 'test']
        # valid_df = org_df[org_df['soroosh_split'] == 'valid']
        # self.df = self.df.append(valid_df)
        # self.df = self.df.sort_values(['BraTS_2020_subject_ID'])



    def create_nonzero_mask(self, data):
        from scipy.ndimage import binary_fill_holes
        assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
        nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
        for c in range(data.shape[0]):
            this_mask = data[c] != 0
            nonzero_mask = nonzero_mask | this_mask
        nonzero_mask = binary_fill_holes(nonzero_mask)
        return nonzero_mask


    def get_bbox_from_mask(self, mask, outside_value=0):
        mask_voxel_coords = np.where(mask != outside_value)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


    def crop_to_bbox(self, image, bbox):
        assert len(image.shape) == 3, "only supports 3d images"
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        return image[resizer]


    def perform_cropping_old_data(self):

        for index, row in tqdm(self.df.iterrows()):
            path_pat = os.path.join(self.file_base_dir, 'pat' + str(row['pat_num']).zfill(3))
            path_file = os.path.join(path_pat, 'pat' + str(row['pat_num']).zfill(3) + '-mod1.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            nonzero_mask = self.create_nonzero_mask(data)
            bbox = self.get_bbox_from_mask(nonzero_mask, 0)
            # bbox = [[0, 148], [41, 190], [35, 220]]
            bbox[1:] = [[41, 190], [35, 220]]
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            x_input_nifti.affine[0, 3] = bbox[0][0] + 1
            x_input_nifti.affine[1, 3] = bbox[1][0] + 1
            x_input_nifti.affine[2, 3] = bbox[2][0] + 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            nib.save(resultt, path_file)

            # mod2
            path_file = path_file.replace('mod1', 'mod2')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            x_input_nifti.affine[0, 3] = bbox[0][0] + 1
            x_input_nifti.affine[1, 3] = bbox[1][0] + 1
            x_input_nifti.affine[2, 3] = bbox[2][0] + 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            nib.save(resultt, path_file)

            # mod3
            path_file = path_file.replace('mod2', 'mod3')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            x_input_nifti.affine[0, 3] = bbox[0][0] + 1
            x_input_nifti.affine[1, 3] = bbox[1][0] + 1
            x_input_nifti.affine[2, 3] = bbox[2][0] + 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            nib.save(resultt, path_file)

            # mod4
            path_file = path_file.replace('mod3', 'mod4')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            x_input_nifti.affine[0, 3] = bbox[0][0] + 1
            x_input_nifti.affine[1, 3] = bbox[1][0] + 1
            x_input_nifti.affine[2, 3] = bbox[2][0] + 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            nib.save(resultt, path_file)

            # seg-label1
            path_file = path_file.replace('mod4', 'seg-label1')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            x_input_nifti.affine[0, 3] = bbox[0][0] + 1
            x_input_nifti.affine[1, 3] = bbox[1][0] + 1
            x_input_nifti.affine[2, 3] = bbox[2][0] + 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            nib.save(resultt, path_file)

            # seg-label2
            path_file = path_file.replace('seg-label1', 'seg-label2')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            x_input_nifti.affine[0, 3] = bbox[0][0] + 1
            x_input_nifti.affine[1, 3] = bbox[1][0] + 1
            x_input_nifti.affine[2, 3] = bbox[2][0] + 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            nib.save(resultt, path_file)

            # seg-label3
            path_file = path_file.replace('seg-label2', 'seg-label3')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            x_input_nifti.affine[0, 3] = bbox[0][0] + 1
            x_input_nifti.affine[1, 3] = bbox[1][0] + 1
            x_input_nifti.affine[2, 3] = bbox[2][0] + 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            nib.save(resultt, path_file)


    def perform_cropping_new_data(self):

        # initializing the df for padding later to the original size
        final_df = pd.DataFrame(columns=['pat', 'left_h_first_dim', 'right_h_first_dim', 'left_w_second_dim', 'right_w_second_dim', 'left_d_third_dim', 'right_d_third_dim'])

        for index, row in tqdm(self.df.iterrows()):
            path_pat = os.path.join(self.file_base_dir, str(row['BraTS_2020_subject_ID']))
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_t1.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata() # (h, w, d)
            data = np.expand_dims(data, 0) # (1, h, w, d)

            nonzero_mask = self.create_nonzero_mask(data)
            bbox = self.get_bbox_from_mask(nonzero_mask, 0)
            final_df = final_df.append(pd.DataFrame([[str(row['BraTS_2020_subject_ID']), bbox[0][0], data.shape[1] - bbox[0][1],
                                                      bbox[1][0], data.shape[2] - bbox[1][1], bbox[2][0], data.shape[3] - bbox[2][1]]],
                                                    columns=['pat', 'left_h_first_dim', 'right_h_first_dim', 'left_w_second_dim',
                                                             'right_w_second_dim', 'left_d_third_dim', 'right_d_third_dim']))
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data) # (1, h, w, d)
            # x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/')), exist_ok=True)

            nib.save(resultt, path_file_output) # (h, w, d)

            # mod2
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_t1ce.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            # x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/')), exist_ok=True)
            nib.save(resultt, path_file_output) # (h, w, d)

            # mod3
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_t2.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            # x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/')), exist_ok=True)
            nib.save(resultt, path_file_output) # (h, w, d)

            # mod4
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_flair.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            # x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/')), exist_ok=True)
            nib.save(resultt, path_file_output) # (h, w, d)

            # # seg-label
            # path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_seg.nii.gz')
            # x_input_nifti = nib.load(path_file)
            # data = x_input_nifti.get_fdata()
            # data = np.expand_dims(data, 0)
            # cropped_data = []
            # for c in range(data.shape[0]):
            #     cropped = self.crop_to_bbox(data[c], bbox)
            #     cropped_data.append(cropped[None])
            # data = np.vstack(cropped_data)
            # x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            # resultt = nib.Nifti1Image(data[0].astype(np.uint8), affine=x_input_nifti.affine, header=x_input_nifti.header)
            # path_file_output = path_file.replace('/original/', '/cropped/')
            # os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/')), exist_ok=True)
            # nib.save(resultt, path_file_output) # (h, w, d)

        final_df.to_csv('/datasets/BraTS20/new_BraTS20/officialvalidation_padding_after_cropping.csv', sep=',', index=False)


    def perform_cropping_new_data_fullsize_training(self):

        # initializing the df for padding later to the original size
        final_df = pd.DataFrame(columns=['pat', 'left_h_first_dim', 'right_h_first_dim', 'left_w_second_dim', 'right_w_second_dim', 'left_d_third_dim', 'right_d_third_dim'])

        for index, row in tqdm(self.df.iterrows()):
            path_pat = os.path.join(self.file_base_dir, str(row['BraTS_2020_subject_ID']))
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_t1.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata() # (h, w, d)
            data = np.expand_dims(data, 0) # (1, h, w, d)

            nonzero_mask = self.create_nonzero_mask(data)
            bbox = self.get_bbox_from_mask(nonzero_mask, 0)
            final_df = final_df.append(pd.DataFrame([[str(row['BraTS_2020_subject_ID']), bbox[0][0], data.shape[1] - bbox[0][1],
                                                      bbox[1][0], data.shape[2] - bbox[1][1], bbox[2][0], data.shape[3] - bbox[2][1]]],
                                                    columns=['pat', 'left_h_first_dim', 'right_h_first_dim', 'left_w_second_dim',
                                                             'right_w_second_dim', 'left_d_third_dim', 'right_d_third_dim']))
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data) # (1, h, w, d)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/fullsize_label/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/fullsize_label/')), exist_ok=True)
            nib.save(resultt, path_file_output) # (h, w, d)

            # mod2
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_t1ce.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/fullsize_label/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/fullsize_label/')), exist_ok=True)
            nib.save(resultt, path_file_output) # (h, w, d)

            # mod3
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_t2.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/fullsize_label/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/fullsize_label/')), exist_ok=True)
            nib.save(resultt, path_file_output) # (h, w, d)

            # mod4
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_flair.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata()
            data = np.expand_dims(data, 0)
            cropped_data = []
            for c in range(data.shape[0]):
                cropped = self.crop_to_bbox(data[c], bbox)
                cropped_data.append(cropped[None])
            data = np.vstack(cropped_data)
            x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
            # x_input_nifti.affine[0, 3] -= bbox[0][0] - 1
            # x_input_nifti.affine[1, 3] -= bbox[1][0] - 1
            # x_input_nifti.affine[2, 3] -= bbox[2][0] - 1
            resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/fullsize_label/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/fullsize_label/')), exist_ok=True)
            nib.save(resultt, path_file_output) # (h, w, d)

            # full size seg-label
            path_file = os.path.join(path_pat, str(row['BraTS_2020_subject_ID']) + '_seg.nii.gz')
            x_input_nifti = nib.load(path_file)
            data = x_input_nifti.get_fdata() # (h, w, d)
            resultt = nib.Nifti1Image(data.astype(np.uint8), affine=x_input_nifti.affine, header=x_input_nifti.header)
            path_file_output = path_file.replace('/original/', '/cropped/fullsize_label/')
            os.makedirs(os.path.dirname(path_file.replace('/original/', '/cropped/fullsize_label/')), exist_ok=True)
            nib.save(resultt, path_file_output) # (h, w, d)


        final_df.to_csv('//datasets/BraTS20/new_BraTS20/officialtraining_padding_after_cropping.csv', sep=',', index=False)



if __name__ == '__main__':
    # handler = csv_preprocess_brats()
    # handler.csv_divider_train_valid_test(num_clients=5)
    # handler.csv_divider_train_valid(num_clients=5)
    crroppper = cropper()
    crroppper = crroppper.perform_cropping_new_data()
