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

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')




class csv_preprocess_brats():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/federated_he/config/config.yaml"):
        self.params = read_config(cfg_path)

    def hd5_to_nifti(self):
        """Converting HD5 data to nifti
        we have 4 MRI modalities and 3 labels per patient
        """

        # filee = "/home/soroosh/Downloads/BraTS2020_training_data/content/data/volume_1_slice_40.h5"
        org_df = "/home/soroosh/Downloads/BraTS2020_training_data/content/data/meta_data.csv"
        target_base = "/home/soroosh/Documents/datasets/BraTS20/"
        base_dir = '/home/soroosh/Downloads/BraTS2020_training_data/content/data'

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


class cropper():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/federated_he/config/config.yaml"):
        """
        Cropping the all the images and segmentations around the brain
        Parameters
        ----------
        cfg_path
        """
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        org_df = pd.read_csv(os.path.join(self.file_base_dir, "brats20_master_list.csv"), sep=',')
        self.df = org_df[org_df['pat_num'] > 72]



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


    def perform_cropping(self):

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



if __name__ == '__main__':
    # handler = csv_preprocess_mimic()
    # handler.hd5_to_nifti()
    crroppper = cropper()
    crroppper = crroppper.perform_cropping()
