"""
Created on March 4, 2022.
main_3D_brats.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
import nibabel as nib
from math import floor
from sklearn import metrics

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_brats import Training
from Prediction_brats import Prediction
from data.data_provider_brats import data_loader_3D, data_loader_without_label_3D, data_loader_3D_multiclass, data_loader_without_label_3D_multiclass, evaluation_fullsize_loader_3D
from models.UNet3D import UNet3D
from models.EDiceLoss_loss import EDiceLoss
from models.generalizeddice import GeneralizedDiceLoss

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15




def main_train_central_3D(global_config_path="federated_he/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', modality=2):
    """Main function for training + validation for directly 3d-wise

        Parameters
        ----------
        global_config_path: str
            always global_config_path="federated_he/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.

        modality: int
            modality of the MR sequence
            1: T1
            2: T1Gd
            3: T2
            4: T2-FLAIR
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    model = UNet3D(n_out_classes=3, firstdim=48) # for multiclass
    image_downsample = params['Network']['image_downsample']
    # weight = torch.Tensor(params['class_weights'])
    weight = None

    loss_function = EDiceLoss # for multi label
    # loss_function = GeneralizedDiceLoss # for multiclass
    # loss_function = CrossEntropyLoss # for multiclass
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    train_dataset = data_loader_3D(cfg_path=cfg_path, mode='train', image_downsample=image_downsample)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=2)
    if valid:
        valid_dataset = data_loader_3D(cfg_path=cfg_path, mode='valid', image_downsample=image_downsample)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=True, drop_last=True, shuffle=False, num_workers=2)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume, augment=augment)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, image_downsample=image_downsample)




def main_train_federated_3D(global_config_path="federated_he/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', modality=2, HE=False, num_clients=3, precision_fractional=15):
    """Main function for training + validation for directly 3d-wise

        Parameters
        ----------
        global_config_path: str
            always global_config_path="federated_he/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.

        modality: int
            modality of the MR sequence
            1: T1
            2: T1Gd
            3: T2
            4: T2-FLAIR

        HE: bool
            if we want to have homomorphic encryption

        num_clients: int
            number of training federated clients we want
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    model = UNet3D(n_out_classes=3, firstdim=48) # for multiclass
    # weight = torch.Tensor(params['class_weights'])
    weight = None

    loss_function = EDiceLoss # for multi label
    # loss_function = GeneralizedDiceLoss # for multiclass
    # loss_function = CrossEntropyLoss # for multiclass
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    num_workers = floor(16 / (num_clients + 1))
    image_downsample = params['Network']['image_downsample']

    train_loader = []
    for num in range(num_clients):
        train_dataset_client = data_loader_3D(cfg_path=cfg_path, mode='train', site='site-' + str(num + 1), image_downsample=image_downsample)
        train_loader.append(torch.utils.data.DataLoader(dataset=train_dataset_client, batch_size=params['Network']['batch_size'],
                                                           pin_memory=True, drop_last=True, shuffle=False, num_workers=num_workers))
    if valid:
        valid_dataset = data_loader_3D(cfg_path=cfg_path, mode='valid', image_downsample=image_downsample)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=True, drop_last=True, shuffle=False, num_workers=num_workers)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume, augment=augment)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function, weight=weight)
    trainer.training_setup_federated(train_loader, valid_loader=valid_loader, HE=HE, precision_fractional=precision_fractional, image_downsample=image_downsample)




def main_evaluate_3D(global_config_path="federated_he/config/config.yaml",
                    experiment_name='name', modality=2, tta=False):
    """Evaluation (for local models) for all the images using the labels and calculating metrics.

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = UNet3D(n_out_classes=3, firstdim=48)
    image_downsample = params['Network']['image_downsample']

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)

    # Generate test set
    test_dataset = data_loader_3D(cfg_path=cfg_path, mode='valid', image_downsample=image_downsample)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=False, num_workers=5)

    if tta:
        test_F1, test_accuracy, test_specifity, test_sensitivity, test_precision = predictor.evaluate_3D_tta(test_loader)
    else:
        test_F1, test_accuracy, test_specifity, test_sensitivity, test_precision = predictor.evaluate_3D(test_loader)

    ### evaluation metrics
    print(f'\n\t Average Dice score (whole tumor): {test_F1.mean().item() * 100:.2f}% | accuracy: {test_accuracy.mean().item() * 100:.2f}%'
          f' | specifity WT: {test_specifity.mean().item() * 100:.2f}%'
          f' | recall (sensitivity) WT: {test_sensitivity.mean().item() * 100:.2f}% | precision WT: {test_precision.mean().item() * 100:.2f}%\n')

    print('Individual Dice scores:')
    print(f'Dice label 1 (necrotic tumor core): {test_F1[0].item() * 100:.2f}%')
    print(f'Dice label 2 (peritumoral edematous/invaded tissue): {test_F1[1].item() * 100:.2f}%\n')
    print(f'Dice label 4, i.e., enhancing tumor (ET): {test_F1[2].item() * 100:.2f}%')
    print(f'Dice average 1 and 4, i.e., tumor core (TC): {(test_F1[0].item() + test_F1[2].item())/2 * 100:.2f}%')
    print(f'Dice average all 1, 2, 4, i.e., whole tumor (WT): {test_F1.mean().item() * 100:.2f}%\n')
    print('------------------------------------------------------'
          '----------------------------------')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name}\n\n' \
          f' test-time augmentation: {str(tta)} | Number of test images: {str(len(test_loader))}\n' \
          f'\n  Average Dice score (whole tumor): {test_F1.mean().item() * 100:.2f}% | accuracy: {test_accuracy.mean().item() * 100:.2f}% ' \
          f' | specifity WT: {test_specifity.mean().item() * 100:.2f}%' \
          f' | recall (sensitivity) WT: {test_sensitivity.mean().item() * 100:.2f}% | precision WT: {test_precision.mean().item() * 100:.2f}%\n\n' \
          f'  Dice label 1 (necrotic tumor core): {test_F1[0].item() * 100:.2f}% | ' \
          f'Dice label 2 (peritumoral edematous/invaded tissue): {test_F1[1].item() * 100:.2f}%\n\n' \
          f'- Dice label 4, i.e., enhancing tumor (ET): {test_F1[2].item() * 100:.2f}%\n' \
          f'- Dice average 1 and 4, i.e., tumor core (TC): {(test_F1[0].item() + test_F1[2].item())/2 * 100:.2f}%\n' \
          f'- Dice average all 1, 2, 4, i.e., whole tumor (WT): {test_F1.mean().item() * 100:.2f}%\n\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results', 'a') as f:
        f.write(msg)



def main_evaluate_3D_multilabelbased_multiclass_output(global_config_path="federated_he/config/config.yaml",
                    experiment_name='name', modality=2, tta=False):
    """Evaluation (for local models) for all the images using the labels and calculating metrics.

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = UNet3D(n_out_classes=3, firstdim=48)
    image_downsample = params['Network']['image_downsample']

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)

    # Generate test set
    test_dataset = evaluation_fullsize_loader_3D(cfg_path=cfg_path, mode='valid', image_downsample=image_downsample)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=False, num_workers=5)
    accuracy = 0
    f1_score = np.zeros(3)

    for idx, (image, org_label, file_name) in enumerate(tqdm(test_loader)):

        padding_df_pat = test_dataset.padding_df[test_dataset.padding_df['pat'] == file_name[0]]

        if tta:
            output_sigmoided = predictor.predict_3D_tta(image)
        else:
            output_sigmoided = predictor.predict_3D(image)
        output_sigmoided_classified = (output_sigmoided > 0.5).float()
        output_sigmoided_classified = output_sigmoided_classified.cpu().detach().numpy()
        output_sigmoided_classified = output_sigmoided_classified.transpose(0, 1, 3, 4, 2) # (n, c, h, w, d)
        output_sigmoided = output_sigmoided.cpu().detach().numpy()
        output_sigmoided = output_sigmoided.transpose(0, 1, 3, 4, 2) # (n, c, h, w, d)

        label1 = output_sigmoided_classified[0, 0].copy()
        label2 = output_sigmoided_classified[0, 1].copy()
        label4 = output_sigmoided_classified[0, 2].copy()
        label1 = np.where(label1 == 1, 1, 0)  # (h, w, d)
        label2 = np.where(label2 == 1, 2, 0)  # (h, w, d)
        label4 = np.where(label4 == 1, 4, 0)  # (h, w, d)

        label = label1 + label2 + label4 # (h, w, d)

        temp = np.where(label == 3)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 5)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 6)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 7)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        # padding the cropped image back to the original size
        label = np.pad(label, [(padding_df_pat['left_h_first_dim'].values[0], padding_df_pat['right_h_first_dim'].values[0]),
                               (padding_df_pat['left_w_second_dim'].values[0], padding_df_pat['right_w_second_dim'].values[0]),
                               (padding_df_pat['left_d_third_dim'].values[0], padding_df_pat['right_d_third_dim'].values[0])],
                       mode='constant')

        label = label.astype(np.uint8)  # (h, w, d)

        org_label = org_label.numpy()  # (h, w, d)
        org_label = org_label.astype(np.uint8)  # (h, w, d)

        ############ Evaluation metric calculation ########

        accuracy += metrics.accuracy_score(org_label.flatten(), label.flatten())
        f1_score_single = metrics.f1_score(org_label.flatten(), label.flatten(), labels=[1, 2, 4], average=None)
        f1_score += f1_score_single
        print(f1_score_single, '| sum 1:', org_label[org_label==1].sum(), label[label==1].sum(),
              '| sum 2:', org_label[org_label==2].sum(), label[label==2].sum(), '| sum 3:',
              org_label[org_label==4].sum(), label[label==4].sum())

    test_F1 = f1_score / len(test_loader)
    test_accuracy = accuracy / len(test_loader)

    ### evaluation metrics
    print(f'\n\t Average Dice score (whole tumor): {test_F1.mean() * 100:.2f}% | accuracy: {test_accuracy * 100:.2f}%\n')

    print('Individual Dice scores:')
    print(f'Dice label 1 (necrotic tumor core): {test_F1[0] * 100:.2f}%')
    print(f'Dice label 2 (peritumoral edematous/invaded tissue): {test_F1[1] * 100:.2f}%\n')
    print(f'Dice label 4, i.e., enhancing tumor (ET): {test_F1[2] * 100:.2f}%')
    print(f'Dice average 1 and 4, i.e., tumor core (TC): {(test_F1[0] + test_F1[2])/2 * 100:.2f}%')
    print(f'Dice average all 1, 2, 4, i.e., whole tumor (WT): {test_F1.mean() * 100:.2f}%\n')
    print('------------------------------------------------------'
          '----------------------------------')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name}\n\n' \
          f' test-time augmentation: {str(tta)} | Number of test images: {str(len(test_loader))}\n' \
          f'\n  Average Dice score (whole tumor): {test_F1.mean() * 100:.2f}% | accuracy: {test_accuracy * 100:.2f}%\n\n' \
          f'  Dice label 1 (necrotic tumor core): {test_F1[0] * 100:.2f}% | ' \
          f'Dice label 2 (peritumoral edematous/invaded tissue): {test_F1[1] * 100:.2f}%\n\n' \
          f'- Dice label 4, i.e., enhancing tumor (ET): {test_F1[2] * 100:.2f}%\n' \
          f'- Dice average 1 and 4, i.e., tumor core (TC): {(test_F1[0] + test_F1[2])/2 * 100:.2f}%\n' \
          f'- Dice average all 1, 2, 4, i.e., whole tumor (WT): {test_F1.mean() * 100:.2f}%\n\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results', 'a') as f:
        f.write(msg)



def main_predict_3D_multilabel_output(global_config_path="/federated_he/config/config.yaml",
                    experiment_name='name', modality=2, tta=False):
    """Prediction without evaluation for all the images.

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = UNet3D(n_out_classes=3)
    image_downsample = params['Network']['image_downsample']

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)

    # Generate test set
    test_dataset = data_loader_without_label_3D(cfg_path=cfg_path, mode='test', image_downsample=image_downsample)

    for idx in tqdm(range(len(test_dataset.file_path_list))):
        path_pat = os.path.join(test_dataset.file_base_dir, 'pat' + str(test_dataset.file_path_list[idx]).zfill(3))
        path_file = os.path.join(path_pat, 'pat' + str(test_dataset.file_path_list[idx]).zfill(3) + '-mod' + str(
            test_dataset.modality) + '.nii.gz')

        x_input, x_input_nifti, img_resized, scaling_ratio = test_dataset.provide_test_without_label(file_path=path_file)

        if tta:
            output_sigmoided = predictor.predict_3D_tta(x_input) # (d,h,w)
        else:
            output_sigmoided = predictor.predict_3D(x_input) # (d,h,w)

        output_sigmoided_classified = (output_sigmoided > 0.5).float()
        output_sigmoided_classified = output_sigmoided_classified.cpu().detach().numpy()

        x_input_nifti.header['pixdim'][1:4] = scaling_ratio
        x_input_nifti.header['dim'][1:4] = np.array(img_resized.shape)
        x_input_nifti.affine[0, 0] = scaling_ratio[0]
        x_input_nifti.affine[1, 1] = scaling_ratio[1]
        x_input_nifti.affine[2, 2] = scaling_ratio[2]

        # segmentation = nib.Nifti1Image(output_sigmoided_classified[0,0], affine=x_input_nifti.affine, header=x_input_nifti.header)
        # nib.save(segmentation, os.path.join(params['target_dir'], params['output_data_path'], os.path.basename(path_file).replace('.nii.gz', '-downsampled' + str(test_dataset.label_num) + '-label' + '.nii.gz')))
        segmentation = nib.Nifti1Image(output_sigmoided_classified[0,0], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(segmentation, os.path.join(params['target_dir'], params['output_data_path'], os.path.basename(path_file).replace('.nii.gz', '-downsampled1-label.nii.gz')))
        segmentation = nib.Nifti1Image(output_sigmoided_classified[0,1], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(segmentation, os.path.join(params['target_dir'], params['output_data_path'], os.path.basename(path_file).replace('.nii.gz', '-downsampled2-label.nii.gz')))
        segmentation = nib.Nifti1Image(output_sigmoided_classified[0,2], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(segmentation, os.path.join(params['target_dir'], params['output_data_path'], os.path.basename(path_file).replace('.nii.gz', '-downsampled4-label.nii.gz')))
        input_img = nib.Nifti1Image(img_resized, affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(input_img, os.path.join(params['target_dir'], params['output_data_path'], os.path.basename(path_file).replace('.nii.gz', '-downsampled-image.nii.gz')))
        pdb.set_trace()



def main_predict_3D_multilabelbased_multiclass_output(global_config_path="/federated_he/config/config.yaml",
                    experiment_name='name', modality=2, tta=False):
    """Prediction without evaluation for all the images.

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = UNet3D(n_out_classes=3, firstdim=48)
    image_downsample = params['Network']['image_downsample']

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)

    # Generate test set
    test_dataset = data_loader_without_label_3D_multiclass(cfg_path=cfg_path, mode='test', image_downsample=image_downsample)

    for idx in tqdm(range(len(test_dataset.file_path_list))):

        path_pat = os.path.join(test_dataset.file_base_dir, str(test_dataset.file_path_list[idx]))
        padding_df_pat = test_dataset.padding_df[test_dataset.padding_df['pat'] == os.path.basename(path_pat)]
        x_input, x_input_nifti, img_resized, scaling_ratio = test_dataset.provide_test_without_label(file_path=path_pat) # (d,h,w) all

        if tta:
            output_sigmoided = predictor.predict_3D_tta(x_input) # (d,h,w)
        else:
            output_sigmoided = predictor.predict_3D(x_input) # (d,h,w)

        output_sigmoided_classified = (output_sigmoided > 0.5).float()
        output_sigmoided_classified = output_sigmoided_classified.cpu().detach().numpy()
        output_sigmoided_classified = output_sigmoided_classified.transpose(0, 1, 3, 4, 2) # (n, c, h, w, d)
        output_sigmoided = output_sigmoided.cpu().detach().numpy()
        output_sigmoided = output_sigmoided.transpose(0, 1, 3, 4, 2) # (n, c, h, w, d)

        label1 = output_sigmoided_classified[0,0].copy()
        label2 = output_sigmoided_classified[0,1].copy()
        label4 = output_sigmoided_classified[0,2].copy()
        label1 = np.where(label1 == 1, 1, 0) # (h, w, d)
        label2 = np.where(label2 == 1, 2, 0) # (h, w, d)
        label4 = np.where(label4 == 1, 4, 0) # (h, w, d)

        label = label1 + label2 + label4 # (h, w, d)

        temp = np.where(label == 3)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 5)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 6)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 7)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        # padding the cropped image back to the original size
        label = np.pad(label, [(padding_df_pat['left_h_first_dim'].values[0], padding_df_pat['right_h_first_dim'].values[0]),
                               (padding_df_pat['left_w_second_dim'].values[0], padding_df_pat['right_w_second_dim'].values[0]),
                               (padding_df_pat['left_d_third_dim'].values[0], padding_df_pat['right_d_third_dim'].values[0])],
                       mode='constant')

        label = label.astype(np.uint8)

        segmentation = nib.Nifti1Image(label, affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(segmentation, os.path.join(params['target_dir'], params['output_data_path'], os.path.basename(path_pat) + '.nii.gz'))
        # nib.save(segmentation, os.path.join(params['target_dir'], params['output_data_path'], os.path.basename(path_pat) + '-label.nii.gz'))




def main_predict_3D_crossvalid_multilabelbased_multiclass_output(global_config_path="/federated_he/config/config.yaml",
                    experiment_name1='name', experiment_name2='name', experiment_name3='name', experiment_name4='name', experiment_name5='name', modality=2, tta=False):
    """Prediction without evaluation for all the images.
    5-fold cross validation

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name1, global_config_path)
    cfg_path = params['cfg_path']
    model = UNet3D(n_out_classes=3, firstdim=48)
    image_downsample = params['Network']['image_downsample']

    # Initialize predictions
    predictor1 = Prediction(cfg_path)
    predictor1.setup_model(model=model)

    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2)
    model2 = UNet3D(n_out_classes=3, firstdim=48)
    predictor2.setup_model(model=model2)

    params3 = open_experiment(experiment_name3, global_config_path)
    cfg_path3 = params3['cfg_path']
    predictor3 = Prediction(cfg_path3)
    model3 = UNet3D(n_out_classes=3, firstdim=48)
    predictor3.setup_model(model=model3)

    params4 = open_experiment(experiment_name4, global_config_path)
    cfg_path4 = params4['cfg_path']
    predictor4 = Prediction(cfg_path4)
    model4 = UNet3D(n_out_classes=3, firstdim=48)
    predictor4.setup_model(model=model4)

    params5 = open_experiment(experiment_name5, global_config_path)
    cfg_path5 = params5['cfg_path']
    predictor5 = Prediction(cfg_path5)
    model5 = UNet3D(n_out_classes=3, firstdim=48)
    predictor5.setup_model(model=model5)

    # Generate test set
    test_dataset = data_loader_without_label_3D_multiclass(cfg_path=cfg_path, mode='test', image_downsample=image_downsample)

    for idx in tqdm(range(len(test_dataset.file_path_list))):

        path_pat = os.path.join(test_dataset.file_base_dir, str(test_dataset.file_path_list[idx]))
        padding_df_pat = test_dataset.padding_df[test_dataset.padding_df['pat'] == os.path.basename(path_pat)]
        x_input, x_input_nifti, img_resized, scaling_ratio = test_dataset.provide_test_without_label(file_path=path_pat) # (d,h,w) all

        if tta:
            output_sigmoided1 = predictor1.predict_3D_tta(x_input) # (d,h,w)
            output_sigmoided2 = predictor2.predict_3D_tta(x_input) # (d,h,w)
            output_sigmoided3 = predictor3.predict_3D_tta(x_input) # (d,h,w)
            output_sigmoided4 = predictor4.predict_3D_tta(x_input) # (d,h,w)
            output_sigmoided5 = predictor5.predict_3D_tta(x_input) # (d,h,w)
            output_sigmoided = (output_sigmoided1 + output_sigmoided2 + output_sigmoided3 + output_sigmoided4 + output_sigmoided5) / 5
        else:
            output_sigmoided1 = predictor1.predict_3D(x_input) # (d,h,w)
            output_sigmoided2 = predictor2.predict_3D(x_input) # (d,h,w)
            output_sigmoided3 = predictor3.predict_3D(x_input) # (d,h,w)
            output_sigmoided4 = predictor4.predict_3D(x_input) # (d,h,w)
            output_sigmoided5 = predictor5.predict_3D(x_input) # (d,h,w)
            output_sigmoided = (output_sigmoided1 + output_sigmoided2 + output_sigmoided3 + output_sigmoided4 + output_sigmoided5) / 5

        output_sigmoided_classified = (output_sigmoided > 0.5).float()

        output_sigmoided_classified = output_sigmoided_classified.cpu().detach().numpy()
        output_sigmoided_classified = output_sigmoided_classified.transpose(0, 1, 3, 4, 2) # (n, c, h, w, d)
        output_sigmoided = output_sigmoided.cpu().detach().numpy()
        output_sigmoided = output_sigmoided.transpose(0, 1, 3, 4, 2) # (n, c, h, w, d)

        label1 = output_sigmoided_classified[0,0].copy()
        label2 = output_sigmoided_classified[0,1].copy()
        label4 = output_sigmoided_classified[0,2].copy()
        label1 = np.where(label1 == 1, 1, 0) # (h, w, d)
        label2 = np.where(label2 == 1, 2, 0) # (h, w, d)
        label4 = np.where(label4 == 1, 4, 0) # (h, w, d)

        label = label1 + label2 + label4 # (h, w, d)

        temp = np.where(label == 3)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 5)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 6)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        temp = np.where(label == 7)
        if len(temp[0]) > 0:
            for voxel in range(len(temp[0])):
                label_value = output_sigmoided[0, :, temp[0][voxel], temp[1][voxel], temp[2][voxel]].argmax() + 1
                if label_value > 2:
                    label_value = 4
                label[temp[0][voxel], temp[1][voxel], temp[2][voxel]] = label_value

        # padding the cropped image back to the original size
        label = np.pad(label, [(padding_df_pat['left_h_first_dim'].values[0], padding_df_pat['right_h_first_dim'].values[0]),
                               (padding_df_pat['left_w_second_dim'].values[0], padding_df_pat['right_w_second_dim'].values[0]),
                               (padding_df_pat['left_d_third_dim'].values[0], padding_df_pat['right_d_third_dim'].values[0])],
                       mode='constant')

        label = label.astype(np.uint8)

        segmentation = nib.Nifti1Image(label, affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(segmentation, os.path.join(params5['target_dir'], params5['output_data_path'], 'ensembled', os.path.basename(path_pat) + '.nii.gz'))
        # nib.save(segmentation, os.path.join(params['target_dir'], params['output_data_path'], os.path.basename(path_pat) + '-label.nii.gz'))




if __name__ == '__main__':
    main_train_federated_3D(global_config_path="/federated_he/config/config.yaml",
                  valid=True, resume=False, augment=True, experiment_name='tempp', HE=True, num_clients=2, precision_fractional=16)