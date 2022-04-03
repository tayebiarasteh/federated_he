"""
Created on March 7, 2022.
augmentation_brats.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""


import pdb
import torchio as tio
from random import random
import nibabel as nib
import numpy as np

from config.serde import read_config




def random_spatial_brats_augmentation(image, label, confg_path='/home/soroosh/Documents/Repositories/federated_he/config/config.yaml'):
    """Both image and the label should be augmented
    1. Random flip
    2, 3, 4. Random affine (zoom, rotation, shift)
        - scales (zoom): Tuple (a1, b1, a2, b2, a3, b3) defining the scaling ranges. a1 to b1 range for the first dimension
        0.1 means from 0.9 to 1.1 = (0.9, 1.1)
        - degrees (rotation): Tuple (a1, b1, a2, b2, a3, b3) defining the rotation ranges in degrees. a1 to b1 range for the first dimension
        - translation (shift): Tuple (a1, b1, a2, b2, a3, b3) defining the translation ranges in mm. a1 to b1 range for the first dimension
        - image_interpolation: 'nearest', 'linear', 'bspline', 'lanczos'. For the label we must choose 'nearest'.
        - default_pad_value: 'mean', 'minimum'. For the label we must choose 'minimum'.
    5. Random Elastic deformation
        - num_control_points: Number of control points along each dimension of the coarse grid (nx, ny, nz).
        Smaller numbers generate smoother deformations.
        The minimum number of control points is 4 as this transform uses cubic B-splines to interpolate displacement.
        - max_displacement (8 is good for brats): Maximum displacement along each dimension at each control point
        - locked_borders: If 0, all displacement vectors are kept.
        If 1, displacement of control points at the border of the coarse grid will be set to 0.
        If 2, displacement of control points at the border of the image (red dots in the image below) will also be set to 0.
    Compose
        first do flipping, then affine, then elastic
        for brats don't do flipping.
        elastic deformation important for brats
        rotation only one degree for brats
        intensity augmentations are more important for brats
    Parameters
    ----------
    image: torch tensor (n, c, d, h, w)
    label: torch tensor (n, c, d, h, w)
    confg_path: str

    Returns
    -------
    transformed_image.unsqueeze(0): torch tensor (n, c, d, h, w)
    transformed_label.unsqueeze(0): torch tensor (n, c, d, h, w)
    """
    params = read_config(confg_path)

    transform = tio.transforms.RandomFlip(axes='L', flip_probability=params['augmentation']['lateral_flip_prob'])
    image = transform(image)
    label = transform(label)

    transform = tio.transforms.RandomFlip(axes='I', flip_probability=params['augmentation']['interior_flip_prob'])
    image = transform(image)
    label = transform(label)

    if random() < params['augmentation']['elastic_prob']:
        transform = tio.RandomElasticDeformation(num_control_points=(params['augmentation']['eladf_control_points']),
                                                  max_displacement=(params['augmentation']['eladf_max_displacement']),
                                                  locked_borders=2, image_interpolation='nearest')
        # segmentation = nib.Nifti1Image(image.numpy()[1], affine=np.eye(4))
        # nib.save(segmentation, 'org.nii.gz')
        # segmentation = nib.Nifti1Image(transform(image).numpy()[1], affine=np.eye(4))
        # nib.save(segmentation, 'trans.nii.gz')
        image = transform(image)
        label = transform(label)
        return image, label

    if random() < params['augmentation']['zoom_prob']:
        transform = tio.RandomAffine(scales=(params['augmentation']['zoom_range'][0], params['augmentation']['zoom_range'][1]), default_pad_value='minimum',
                                     translation=(0,0,0), degrees=(0,0,0), image_interpolation='nearest')
        image = transform(image)
        label = transform(label)

    if random() < params['augmentation']['rotation_prob']:
        transform = tio.RandomAffine(degrees=(params['augmentation']['rotation_range']), default_pad_value='minimum',
                                     image_interpolation='nearest')
        image = transform(image)
        label = transform(label)

    if random() < params['augmentation']['shift_prob']:
        transform = tio.RandomAffine(translation=(params['augmentation']['shift_range']), default_pad_value='minimum',
                                     image_interpolation='nearest')
        image = transform(image)
        label = transform(label)

    assert len(label.unique()) == 2
    return image, label



def random_intensity_brats_augmentation(image, confg_path='/home/soroosh/Documents/Repositories/federated_he/config/config.yaml'):
    """Only image should be augmented
    """
    params = read_config(confg_path)

    # additive Gaussian noise (not needed for min max normalization; 20% prob for the mean std normalization)
    if random() < params['augmentation']['AWGN_prob']:
        transform = tio.RandomNoise(mean=params['augmentation']['mu_AWGN'], std=params['augmentation']['sigma_AWGN'])
        return transform(image)

    if random() < params['augmentation']['gamma_prob']:
        transform = tio.RandomGamma(log_gamma=(params['augmentation']['gamma_range'][0], params['augmentation']['gamma_range'][1]))
        return transform(image)

    if random() < params['augmentation']['motion_prob']:
        transform = tio.RandomMotion(degrees=10, translation=10, num_transforms=2)
        return transform(image)

    if random() < params['augmentation']['ghosting_prob']:
        transform = tio.RandomGhosting(num_ghosts=10, axes=(0, 1, 3), intensity=0.3)
        return transform(image)

    if random() < params['augmentation']['blurring_prob']:
        transform = tio.RandomBlur(std=(params['augmentation']['gamma_range'][0], params['augmentation']['gamma_range'][1]))
        return transform(image)

    return image





def random_augment(image, label, confg_path='/home/soroosh/Documents/Repositories/federated_he/config/config.yaml'):
    """
    Parameters
    ----------
    image: torch tensor (n, c, d, h, w)
    label: torch tensor (n, c, d, h, w)
    confg_path: str
    Returns
    -------
    transformed_image: torch tensor (n, c, d, h, w)
    transformed_label: torch tensor (n, c, d, h, w)
    """
    params = read_config(confg_path)

    if random() < params['augmentation']['general_spatial_probability']:
        transformed_image, transformed_label = random_spatial_brats_augmentation(image[0], label[0], confg_path)
        return transformed_image.unsqueeze(0), transformed_label.unsqueeze(0)

    elif random() < params['augmentation']['general_intensity_probability']:
        transformed_image = random_intensity_brats_augmentation(image[0], confg_path)
        return transformed_image.unsqueeze(0), label

    else:
        return image, label