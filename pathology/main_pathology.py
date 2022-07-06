"""
Created on May 2, 2022.
main_pathology.py

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
import pandas as pd
import torch.nn.functional as F
import torchmetrics
from sklearn import metrics
from math import floor

from config.serde import open_experiment, create_experiment, delete_experiment
from Train_Valid_pathology import Training
from data.data_provider_pathology import data_loader_pathology
from models.MNISTNET import mnistNet
from Prediction_pathology import Prediction

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15




def main_train_pathology(global_config_path="/federated_he/pathology/config/config.yaml", valid=False,
                  resume=False, experiment_name='name', train_site='belfast', fold=1):
    """Main function for training + validation for directly 3d-wise

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/federated_he/pathology/config/config.yaml"

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    model = mnistNet()
    loss_function = CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    trainset_class = data_loader_pathology(params["cfg_path"], mode='train', site=train_site, fold=fold)
    train_dataset = trainset_class.provide_data()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)

    if valid:
        validset_class = data_loader_pathology(params["cfg_path"], mode='valid', fold=fold)
        valid_dataset = validset_class.provide_data()
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=True, drop_last=True, shuffle=False, num_workers=2)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume)

    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function)
    trainer.training_init(train_loader=train_loader, valid_loader=valid_loader)



def main_train_federated_3D(global_config_path="/federated_he/pathology/config/config.yaml", valid=False,
                  resume=False, experiment_name='name', HE=False, num_clients=3, precision_fractional=15, fold=1):
    """

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/federated_he/pathology/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.

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
    model = mnistNet()
    loss_function = CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    num_workers = floor(16 / (num_clients + 1))
    train_loader = []

    # belfast
    trainset_class = data_loader_pathology(params["cfg_path"], mode='train', site='belfast', fold=fold)
    train_dataset = trainset_class.provide_data()
    train_loader.append(torch.utils.data.DataLoader(train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers))
    # DACHS
    trainset_class = data_loader_pathology(params["cfg_path"], mode='train', site='DACHS', fold=fold)
    train_dataset = trainset_class.provide_data()
    train_loader.append(torch.utils.data.DataLoader(train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers))
    # TCGA
    trainset_class = data_loader_pathology(params["cfg_path"], mode='train', site='TCGA', fold=fold)
    train_dataset = trainset_class.provide_data()
    train_loader.append(torch.utils.data.DataLoader(train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers))

    if valid:
        validset_class = data_loader_pathology(params["cfg_path"], mode='valid', fold=fold)
        valid_dataset = validset_class.provide_data()
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=True, drop_last=True, shuffle=False, num_workers=num_workers)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume)

    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function)

    trainer.training_setup_federated(train_loader=train_loader, valid_loader=valid_loader, HE=HE, precision_fractional=precision_fractional)





def main_test_pathology(global_config_path="/federated_he/pathology/config/config.yaml",
                    experiment_name='name', benchmark='YORKSHIR_deployMSIH'):
    """Evaluation (for local models) for all the images using the labels and calculating metrics.
    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = mnistNet()

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)

    if benchmark == 'QUASAR_deployMSIH':
        batch_size = 231
    elif benchmark == 'YORKSHIR_deployMSIH':
        batch_size = 39

    # Generate test set
    testset_class = data_loader_pathology(params["cfg_path"], mode='test', benchmark=benchmark)
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    pred_array, target_array = predictor.predict(test_loader)

    patient_wise_pred_list = []
    patient_wise_label_list = []

    split_df_path = os.path.join(params['file_path'], 'test', benchmark, 'org_split.csv')
    df = pd.read_csv(split_df_path, sep=',')

    for index, row in tqdm(df.iterrows()):
        temp = pred_array[row['start_index']: row['end_index'] + 1]
        temp_label = target_array[row['start_index']: row['end_index'] + 1]
        patient_wise_pred_list.append(temp.mean(0))
        patient_wise_label_list.append(temp_label[0])

    patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
    targets = torch.stack(patient_wise_label_list, 0)
    output_sigmoided = F.sigmoid(patient_wise_pred_list)
    max_preds = output_sigmoided.argmax(dim=1)

    pdb.set_trace()
    index_list = np.random.choice(max_preds.shape[0], max_preds.shape[0])

    new_targets = torch.zeros_like(targets)
    new_max_preds = torch.zeros_like(max_preds)
    for i, index in enumerate(index_list):
        new_targets[i] = targets[index]
        new_max_preds[i] = max_preds[index]

    pdb.set_trace()

    ### evaluation metrics
    accuracy_calculator = torchmetrics.Accuracy()
    total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

    fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
    aucc = metrics.auc(fpr, tpr)

    print(f'\n Benchmark name: {benchmark}')
    print(f'\n\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')


def main_test_pathology_bootstrap(global_config_path="/federated_he/pathology/config/config.yaml",
                    experiment_name='name', benchmark='YORKSHIR_deployMSIH'):
    """Evaluation (for local models) for all the images using the labels and calculating metrics.
    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = mnistNet()

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)

    if benchmark == 'QUASAR_deployMSIH':
        batch_size = 231
    elif benchmark == 'YORKSHIR_deployMSIH':
        batch_size = 39

    # Generate test set
    testset_class = data_loader_pathology(params["cfg_path"], mode='test', benchmark=benchmark)
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    pred_array, target_array = predictor.predict(test_loader)

    patient_wise_pred_list = []
    patient_wise_label_list = []

    split_df_path = os.path.join(params['file_path'], 'test', benchmark, 'org_split.csv')
    df = pd.read_csv(split_df_path, sep=',')

    for index, row in tqdm(df.iterrows()):
        temp = pred_array[row['start_index']: row['end_index'] + 1]
        temp_label = target_array[row['start_index']: row['end_index'] + 1]
        patient_wise_pred_list.append(temp.mean(0))
        patient_wise_label_list.append(temp_label[0])

    patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
    targets = torch.stack(patient_wise_label_list, 0)
    output_sigmoided = F.sigmoid(patient_wise_pred_list)
    max_preds = output_sigmoided.argmax(dim=1)

    AUC_list, accuracy_list = bootstrapper(max_preds, targets)
    print(f'\n Benchmark name: {benchmark}')
    print(f'\n\t Accuracy: {accuracy_list.mean() * 100:.2f} +- {accuracy_list.std() * 100:.2f}% | AUROC: {AUC_list.mean() * 100:.2f} +- {AUC_list.std() * 100:.2f}%\n')



def bootstrapper(max_preds, targets):

    AUC_list = []
    accuracy_list = []

    for _ in range(1000):

        index_list = np.random.choice(max_preds.shape[0], max_preds.shape[0])
        new_targets = torch.zeros_like(targets)
        new_max_preds = torch.zeros_like(max_preds)
        for i, index in enumerate(index_list):
            new_targets[i] = targets[index]
            new_max_preds[i] = max_preds[index]

        ### evaluation metrics
        accuracy_calculator = torchmetrics.Accuracy()
        total_accuracy = accuracy_calculator(new_max_preds.cpu(), new_targets.cpu()).item()

        fpr, tpr, thresholds = metrics.roc_curve(new_targets.cpu().numpy(), new_max_preds.cpu().numpy(), pos_label=1)
        aucc = metrics.auc(fpr, tpr)
        AUC_list.append(aucc)
        accuracy_list.append(total_accuracy)

    AUC_list = np.stack(AUC_list)
    accuracy_list = np.stack(accuracy_list)

    return AUC_list, accuracy_list





def main_test_pathology_all_epochs(global_config_path="/federated_he/pathology/config/config.yaml",
                    experiment_name='name', benchmark='YORKSHIR_deployMSIH'):
    """Evaluation (for local models) for all the images using the labels and calculating metrics.
    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = mnistNet()

    # Initialize prediction
    predictor = Prediction(cfg_path)

    if benchmark == 'QUASAR_deployMSIH':
        batch_size = 231
    elif benchmark == 'YORKSHIR_deployMSIH':
        batch_size = 39

    # Generate test set
    testset_class = data_loader_pathology(params["cfg_path"], mode='test', benchmark=benchmark)
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: {benchmark}\n')
    best_auc = 0
    best_acu_epoch = 0

    for i in range(params['num_epochs']):
        predictor.setup_model(model=model, epoch= int(i + 1))

        pred_array, target_array = predictor.predict(test_loader)

        patient_wise_pred_list = []
        patient_wise_label_list = []

        split_df_path = os.path.join(params['file_path'], 'test', benchmark, 'org_split.csv')
        df = pd.read_csv(split_df_path, sep=',')

        for index, row in df.iterrows():
            temp = pred_array[row['start_index']: row['end_index'] + 1]
            temp_label = target_array[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
        targets = torch.stack(patient_wise_label_list, 0)
        output_sigmoided = F.sigmoid(patient_wise_pred_list)
        max_preds = output_sigmoided.argmax(dim=1)

        ### evaluation metrics
        accuracy_calculator = torchmetrics.Accuracy()
        total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

        fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
        aucc = metrics.auc(fpr, tpr)

        if aucc > best_auc:
            best_auc = aucc
            best_acu_epoch = i + 1

        print(f'epoch {i + 1}:')
        print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')

        # saving the training and validation stats
        msg = f'----------------------------------------------------------------------------------------\n' \
              f'epoch {i + 1}:' \
              f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'

        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results', 'a') as f:
            f.write(msg)

    print('------------------------------------------------')
    print('------------------------------------------------')
    print('------------------------------------------------')
    print(f'\t epoch: {best_acu_epoch} | final best AUROC: {best_auc * 100:.2f}%')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name} | Benchmark name: {benchmark}\n\n' \
          f'\t epoch: {best_acu_epoch} | AUROC: {best_auc * 100:.2f}%\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/final_test_result', 'a') as f:
        f.write(msg)


def main_test_pathology_allbenchmarks_all_epochs(global_config_path="/federated_he/pathology/config/config.yaml",
                    experiment_name='name'):
    """
    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = mnistNet()

    # Initialize prediction
    predictor = Prediction(cfg_path)

    ################### first benchmark: QUASAR ##########################

    benchmark = 'QUASAR_deployMSIH'
    batch_size = 231

    # Generate test set
    testset_class = data_loader_pathology(params["cfg_path"], mode='test', benchmark=benchmark)
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: {benchmark}\n')
    best_auc = 0
    best_acu_epoch = 0

    for i in range(params['num_epochs']):
        predictor.setup_model(model=model, epoch= int(i + 1))

        pred_array, target_array = predictor.predict(test_loader)

        patient_wise_pred_list = []
        patient_wise_label_list = []

        split_df_path = os.path.join(params['file_path'], 'test', benchmark, 'org_split.csv')
        df = pd.read_csv(split_df_path, sep=',')

        for index, row in df.iterrows():
            temp = pred_array[row['start_index']: row['end_index'] + 1]
            temp_label = target_array[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
        targets = torch.stack(patient_wise_label_list, 0)
        output_sigmoided = F.sigmoid(patient_wise_pred_list)
        max_preds = output_sigmoided.argmax(dim=1)

        ### evaluation metrics
        accuracy_calculator = torchmetrics.Accuracy()
        total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

        fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
        aucc = metrics.auc(fpr, tpr)

        if aucc > best_auc:
            best_auc = aucc
            best_acu_epoch = i + 1

        print(f'epoch {i + 1}:')
        print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')

        # saving the training and validation stats
        msg = f'----------------------------------------------------------------------------------------\n' \
              f'epoch {i + 1}:' \
              f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'

        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/QUASAR_deployMSIH_test_results', 'a') as f:
            f.write(msg)

    print('------------------------------------------------')
    print('------------------------------------------------')
    print('------------------------------------------------')
    print(f'\t epoch: {best_acu_epoch} | final best AUROC: {best_auc * 100:.2f}%')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name} | Benchmark name: {benchmark}\n\n' \
          f'\t epoch: {best_acu_epoch} | AUROC: {best_auc * 100:.2f}%\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/QUASAR_deployMSIH_final_test_result', 'a') as f:
        f.write(msg)

    ################### second benchmark: YORKSHIR ##########################

    benchmark = 'YORKSHIR_deployMSIH'
    batch_size = 39

    # Generate test set
    testset_class = data_loader_pathology(params["cfg_path"], mode='test', benchmark=benchmark)
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: {benchmark}\n')
    best_auc = 0
    best_acu_epoch = 0

    for i in range(params['num_epochs']):
        predictor.setup_model(model=model, epoch=int(i + 1))

        pred_array, target_array = predictor.predict(test_loader)

        patient_wise_pred_list = []
        patient_wise_label_list = []

        split_df_path = os.path.join(params['file_path'], 'test', benchmark, 'org_split.csv')
        df = pd.read_csv(split_df_path, sep=',')

        for index, row in df.iterrows():
            temp = pred_array[row['start_index']: row['end_index'] + 1]
            temp_label = target_array[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
        targets = torch.stack(patient_wise_label_list, 0)
        output_sigmoided = F.sigmoid(patient_wise_pred_list)
        max_preds = output_sigmoided.argmax(dim=1)

        ### evaluation metrics
        accuracy_calculator = torchmetrics.Accuracy()
        total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

        fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
        aucc = metrics.auc(fpr, tpr)

        if aucc > best_auc:
            best_auc = aucc
            best_acu_epoch = i + 1

        print(f'epoch {i + 1}:')
        print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')

        # saving the training and validation stats
        msg = f'----------------------------------------------------------------------------------------\n' \
              f'epoch {i + 1}:' \
              f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'

        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/YORKSHIR_deployMSIH_test_results',
                  'a') as f:
            f.write(msg)

    print('------------------------------------------------')
    print('------------------------------------------------')
    print('------------------------------------------------')
    print(f'\t epoch: {best_acu_epoch} | final best AUROC: {best_auc * 100:.2f}%')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name} | Benchmark name: {benchmark}\n\n' \
          f'\t epoch: {best_acu_epoch} | AUROC: {best_auc * 100:.2f}%\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/YORKSHIR_deployMSIH_final_test_result',
              'a') as f:
        f.write(msg)

    ################### third benchmark: QUASAR + YORKSHIR ##########################

    # Generate QUASAR test set
    testset_class_QUASAR = data_loader_pathology(params["cfg_path"], mode='test', benchmark='QUASAR_deployMSIH')
    test_dataset_QUASAR = testset_class_QUASAR.provide_data()
    test_loader_QUASAR = torch.utils.data.DataLoader(test_dataset_QUASAR, batch_size=231,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    # Generate YORKSHIR test set
    testset_class_YORKSHIR = data_loader_pathology(params["cfg_path"], mode='test', benchmark='YORKSHIR_deployMSIH')
    test_dataset_YORKSHIR = testset_class_YORKSHIR.provide_data()
    test_loader_YORKSHIR = torch.utils.data.DataLoader(test_dataset_YORKSHIR, batch_size=39,
                                              pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: QUASAR + YORKSHIR\n')
    best_auc = 0
    best_acu_epoch = 0

    for i in range(params['num_epochs']):
        predictor.setup_model(model=model, epoch=int(i + 1))

        pred_array_QUASAR, target_array_QUASAR = predictor.predict(test_loader_QUASAR)
        pred_array_YORKSHIR, target_array_YORKSHIR = predictor.predict(test_loader_YORKSHIR)

        patient_wise_pred_list = []
        patient_wise_label_list = []

        split_df_path_QUASAR = os.path.join(params['file_path'], 'test', 'QUASAR_deployMSIH', 'org_split.csv')
        df_QUASAR = pd.read_csv(split_df_path_QUASAR, sep=',')
        for index, row in df_QUASAR.iterrows():
            temp = pred_array_QUASAR[row['start_index']: row['end_index'] + 1]
            temp_label = target_array_QUASAR[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        split_df_path_YORKSHIR = os.path.join(params['file_path'], 'test', 'YORKSHIR_deployMSIH', 'org_split.csv')
        df_YORKSHIR = pd.read_csv(split_df_path_YORKSHIR, sep=',')
        for index, row in df_YORKSHIR.iterrows():
            temp = pred_array_YORKSHIR[row['start_index']: row['end_index'] + 1]
            temp_label = target_array_YORKSHIR[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
        targets = torch.stack(patient_wise_label_list, 0)
        output_sigmoided = F.sigmoid(patient_wise_pred_list)
        max_preds = output_sigmoided.argmax(dim=1)

        ### evaluation metrics
        accuracy_calculator = torchmetrics.Accuracy()
        total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

        fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
        aucc = metrics.auc(fpr, tpr)

        if aucc > best_auc:
            best_auc = aucc
            best_acu_epoch = i + 1

        print(f'epoch {i + 1}:')
        print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')

        # saving the training and validation stats
        msg = f'----------------------------------------------------------------------------------------\n' \
              f'epoch {i + 1}:' \
              f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'

        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/QUASAR_YORKSHIR_test_results',
                  'a') as f:
            f.write(msg)

    print('------------------------------------------------')
    print('------------------------------------------------')
    print('------------------------------------------------')
    print(f'\t epoch: {best_acu_epoch} | final best AUROC: {best_auc * 100:.2f}%')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name} | Benchmark name: QUASAR + YORKSHIR\n\n' \
          f'\t epoch: {best_acu_epoch} | AUROC: {best_auc * 100:.2f}%\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/QUASAR_YORKSHIR_final_test_result',
              'a') as f:
        f.write(msg)



def main_test_pathology_crossvalid_allbenchmarks_all_epochs(global_config_path="/federated_he/pathology/config/config.yaml",
                    experiment_name1='name', experiment_name2='name', experiment_name3='name', experiment_name4='name', experiment_name5='name', tta=False):

    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']
    model1 = mnistNet()

    # Initialize predictions
    predictor1 = Prediction(cfg_path1)
    predictor1.setup_model(model=model1)

    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2)
    model2 = mnistNet()
    predictor2.setup_model(model=model2)

    params3 = open_experiment(experiment_name3, global_config_path)
    cfg_path3 = params3['cfg_path']
    predictor3 = Prediction(cfg_path3)
    model3 = mnistNet()
    predictor3.setup_model(model=model3)

    params4 = open_experiment(experiment_name4, global_config_path)
    cfg_path4 = params4['cfg_path']
    predictor4 = Prediction(cfg_path4)
    model4 = mnistNet()
    predictor4.setup_model(model=model4)

    params5 = open_experiment(experiment_name5, global_config_path)
    cfg_path5 = params5['cfg_path']
    predictor5 = Prediction(cfg_path5)
    model5 = mnistNet()
    predictor5.setup_model(model=model5)

    os.makedirs(os.path.join(params5['target_dir'], params5['stat_log_path'], 'ensembled'), exist_ok=True)

    ################### first benchmark: QUASAR ##########################

    # Generate test set
    testset_class = data_loader_pathology(params1["cfg_path"], mode='test', benchmark='QUASAR_deployMSIH')
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=231,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: QUASAR_deployMSIH\n')
    best_auc = 0
    best_acu_epoch = 0

    for i in range(params1['num_epochs']):
        predictor1.setup_model(model=model1, epoch= int(i + 1))
        predictor2.setup_model(model=model2, epoch= int(i + 1))
        predictor3.setup_model(model=model3, epoch= int(i + 1))
        predictor4.setup_model(model=model4, epoch= int(i + 1))
        predictor5.setup_model(model=model5, epoch= int(i + 1))

        pred_array1, target_array = predictor1.predict(test_loader)
        pred_array2, _ = predictor2.predict(test_loader)
        pred_array3, _ = predictor3.predict(test_loader)
        pred_array4, _ = predictor4.predict(test_loader)
        pred_array5, _ = predictor5.predict(test_loader)
        pred_array = (pred_array1 + pred_array2 + pred_array3 + pred_array4 + pred_array5) / 5

        patient_wise_pred_list = []
        patient_wise_label_list = []

        split_df_path = os.path.join(params1['file_path'], 'test', 'QUASAR_deployMSIH', 'org_split.csv')
        df = pd.read_csv(split_df_path, sep=',')

        for index, row in df.iterrows():
            temp = pred_array[row['start_index']: row['end_index'] + 1]
            temp_label = target_array[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
        targets = torch.stack(patient_wise_label_list, 0)
        output_sigmoided = F.sigmoid(patient_wise_pred_list)
        max_preds = output_sigmoided.argmax(dim=1)

        ### evaluation metrics
        accuracy_calculator = torchmetrics.Accuracy()
        total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

        fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
        aucc = metrics.auc(fpr, tpr)

        if aucc > best_auc:
            best_auc = aucc
            best_acu_epoch = i + 1

        print(f'epoch {i + 1}:')
        print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')

        # saving the training and validation stats
        msg = f'----------------------------------------------------------------------------------------\n' \
              f'epoch {i + 1}:' \
              f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'

        with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_deployMSIH_test_results', 'a') as f:
            f.write(msg)

    print('------------------------------------------------')
    print('------------------------------------------------')
    print('------------------------------------------------')
    print(f'\t epoch: {best_acu_epoch} | final best AUROC: {best_auc * 100:.2f}%')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name1} | Benchmark name: QUASAR\n\n' \
          f'\t epoch: {best_acu_epoch} | AUROC: {best_auc * 100:.2f}%\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_deployMSIH_final_test_result', 'a') as f:
        f.write(msg)

    ################### second benchmark: YORKSHIR ##########################

    # Generate test set
    testset_class = data_loader_pathology(params1["cfg_path"], mode='test', benchmark='YORKSHIR_deployMSIH')
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=39,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: YORKSHIR_deployMSIH\n')
    best_auc = 0
    best_acu_epoch = 0

    for i in range(params1['num_epochs']):
        predictor1.setup_model(model=model1, epoch= int(i + 1))
        predictor2.setup_model(model=model2, epoch= int(i + 1))
        predictor3.setup_model(model=model3, epoch= int(i + 1))
        predictor4.setup_model(model=model4, epoch= int(i + 1))
        predictor5.setup_model(model=model5, epoch= int(i + 1))

        pred_array1, target_array = predictor1.predict(test_loader)
        pred_array2, _ = predictor2.predict(test_loader)
        pred_array3, _ = predictor3.predict(test_loader)
        pred_array4, _ = predictor4.predict(test_loader)
        pred_array5, _ = predictor5.predict(test_loader)
        pred_array = (pred_array1 + pred_array2 + pred_array3 + pred_array4 + pred_array5) / 5

        patient_wise_pred_list = []
        patient_wise_label_list = []

        split_df_path = os.path.join(params1['file_path'], 'test', 'YORKSHIR_deployMSIH', 'org_split.csv')
        df = pd.read_csv(split_df_path, sep=',')

        for index, row in df.iterrows():
            temp = pred_array[row['start_index']: row['end_index'] + 1]
            temp_label = target_array[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
        targets = torch.stack(patient_wise_label_list, 0)
        output_sigmoided = F.sigmoid(patient_wise_pred_list)
        max_preds = output_sigmoided.argmax(dim=1)

        ### evaluation metrics
        accuracy_calculator = torchmetrics.Accuracy()
        total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

        fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
        aucc = metrics.auc(fpr, tpr)

        if aucc > best_auc:
            best_auc = aucc
            best_acu_epoch = i + 1

        print(f'epoch {i + 1}:')
        print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')

        # saving the training and validation stats
        msg = f'----------------------------------------------------------------------------------------\n' \
              f'epoch {i + 1}:' \
              f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'

        with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/YORKSHIR_deployMSIH_test_results', 'a') as f:
            f.write(msg)

    print('------------------------------------------------')
    print('------------------------------------------------')
    print('------------------------------------------------')
    print(f'\t epoch: {best_acu_epoch} | final best AUROC: {best_auc * 100:.2f}%')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name1} | Benchmark name: YORKSHIR\n\n' \
          f'\t epoch: {best_acu_epoch} | AUROC: {best_auc * 100:.2f}%\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/YORKSHIR_deployMSIH_final_test_result', 'a') as f:
        f.write(msg)


    ################### third benchmark: QUASAR + YORKSHIR ##########################

    # Generate QUASAR test set
    testset_class_QUASAR = data_loader_pathology(params1["cfg_path"], mode='test', benchmark='QUASAR_deployMSIH')
    test_dataset_QUASAR = testset_class_QUASAR.provide_data()
    test_loader_QUASAR = torch.utils.data.DataLoader(test_dataset_QUASAR, batch_size=231,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    # Generate YORKSHIR test set
    testset_class_YORKSHIR = data_loader_pathology(params1["cfg_path"], mode='test', benchmark='YORKSHIR_deployMSIH')
    test_dataset_YORKSHIR = testset_class_YORKSHIR.provide_data()
    test_loader_YORKSHIR = torch.utils.data.DataLoader(test_dataset_YORKSHIR, batch_size=39,
                                              pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: QUASAR + YORKSHIR\n')
    best_auc = 0
    best_acu_epoch = 0

    for i in range(params1['num_epochs']):
        predictor1.setup_model(model=model1, epoch= int(i + 1))
        predictor2.setup_model(model=model2, epoch= int(i + 1))
        predictor3.setup_model(model=model3, epoch= int(i + 1))
        predictor4.setup_model(model=model4, epoch= int(i + 1))
        predictor5.setup_model(model=model5, epoch= int(i + 1))

        pred_array1_QUASAR, target_array_QUASAR = predictor1.predict(test_loader_QUASAR)
        pred_array1_YORKSHIR, target_array_YORKSHIR = predictor1.predict(test_loader_YORKSHIR)

        pred_array2_QUASAR, _ = predictor2.predict(test_loader_QUASAR)
        pred_array2_YORKSHIR, _ = predictor2.predict(test_loader_YORKSHIR)

        pred_array3_QUASAR, _ = predictor3.predict(test_loader_QUASAR)
        pred_array3_YORKSHIR, _ = predictor3.predict(test_loader_YORKSHIR)

        pred_array4_QUASAR, _ = predictor4.predict(test_loader_QUASAR)
        pred_array4_YORKSHIR, _ = predictor4.predict(test_loader_YORKSHIR)

        pred_array5_QUASAR, _ = predictor5.predict(test_loader_QUASAR)
        pred_array5_YORKSHIR, _ = predictor5.predict(test_loader_YORKSHIR)

        pred_array_QUASAR = (pred_array1_QUASAR + pred_array2_QUASAR + pred_array3_QUASAR + pred_array4_QUASAR + pred_array5_QUASAR) / 5
        pred_array_YORKSHIR = (pred_array1_YORKSHIR + pred_array2_YORKSHIR + pred_array3_YORKSHIR + pred_array4_YORKSHIR + pred_array5_YORKSHIR) / 5

        patient_wise_pred_list = []
        patient_wise_label_list = []

        split_df_path_QUASAR = os.path.join(params1['file_path'], 'test', 'QUASAR_deployMSIH', 'org_split.csv')
        df_QUASAR = pd.read_csv(split_df_path_QUASAR, sep=',')
        for index, row in df_QUASAR.iterrows():
            temp = pred_array_QUASAR[row['start_index']: row['end_index'] + 1]
            temp_label = target_array_QUASAR[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        split_df_path_YORKSHIR = os.path.join(params1['file_path'], 'test', 'YORKSHIR_deployMSIH', 'org_split.csv')
        df_YORKSHIR = pd.read_csv(split_df_path_YORKSHIR, sep=',')
        for index, row in df_YORKSHIR.iterrows():
            temp = pred_array_YORKSHIR[row['start_index']: row['end_index'] + 1]
            temp_label = target_array_YORKSHIR[row['start_index']: row['end_index'] + 1]
            patient_wise_pred_list.append(temp.mean(0))
            patient_wise_label_list.append(temp_label[0])

        patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
        targets = torch.stack(patient_wise_label_list, 0)
        output_sigmoided = F.sigmoid(patient_wise_pred_list)
        max_preds = output_sigmoided.argmax(dim=1)

        ### evaluation metrics
        accuracy_calculator = torchmetrics.Accuracy()
        total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

        fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
        aucc = metrics.auc(fpr, tpr)

        if aucc > best_auc:
            best_auc = aucc
            best_acu_epoch = i + 1

        print(f'epoch {i + 1}:')
        print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')

        # saving the training and validation stats
        msg = f'----------------------------------------------------------------------------------------\n' \
              f'epoch {i + 1}:' \
              f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'

        with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_YORKSHIR_test_results', 'a') as f:
            f.write(msg)

    print('------------------------------------------------')
    print('------------------------------------------------')
    print('------------------------------------------------')
    print(f'\t epoch: {best_acu_epoch} | final best AUROC: {best_auc * 100:.2f}%')

    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f' Experiment name: {experiment_name1} | Benchmark name: QUASAR + YORKSHIR\n\n' \
          f'\t epoch: {best_acu_epoch} | AUROC: {best_auc * 100:.2f}%\n' \
          f'----------------------------------------------------------------------------------------\n'

    with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_YORKSHIR_final_test_result', 'a') as f:
        f.write(msg)



def main_test_pathology_crossvalid_allbenchmarks_single_epoch(global_config_path="/federated_he/pathology/config/config.yaml",
                    experiment_name1='name', experiment_name2='name', experiment_name3='name', experiment_name4='name', experiment_name5='name', tta=False):

    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']
    model1 = mnistNet()

    # Initialize predictions
    predictor1 = Prediction(cfg_path1)
    predictor1.setup_model(model=model1)

    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2)
    model2 = mnistNet()
    predictor2.setup_model(model=model2)

    params3 = open_experiment(experiment_name3, global_config_path)
    cfg_path3 = params3['cfg_path']
    predictor3 = Prediction(cfg_path3)
    model3 = mnistNet()
    predictor3.setup_model(model=model3)

    params4 = open_experiment(experiment_name4, global_config_path)
    cfg_path4 = params4['cfg_path']
    predictor4 = Prediction(cfg_path4)
    model4 = mnistNet()
    predictor4.setup_model(model=model4)

    params5 = open_experiment(experiment_name5, global_config_path)
    cfg_path5 = params5['cfg_path']
    predictor5 = Prediction(cfg_path5)
    model5 = mnistNet()
    predictor5.setup_model(model=model5)

    os.makedirs(os.path.join(params5['target_dir'], params5['stat_log_path'], 'ensembled'), exist_ok=True)

    ################### first benchmark: QUASAR ##########################

    # Generate test set
    testset_class = data_loader_pathology(params1["cfg_path"], mode='test', benchmark='QUASAR_deployMSIH')
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=231,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: QUASAR_deployMSIH\n')

    predictor1.setup_model(model=model1)
    predictor2.setup_model(model=model2)
    predictor3.setup_model(model=model3)
    predictor4.setup_model(model=model4)
    predictor5.setup_model(model=model5)

    pred_array1, target_array = predictor1.predict(test_loader)
    pred_array2, _ = predictor2.predict(test_loader)
    pred_array3, _ = predictor3.predict(test_loader)
    pred_array4, _ = predictor4.predict(test_loader)
    pred_array5, _ = predictor5.predict(test_loader)
    pred_array = (pred_array1 + pred_array2 + pred_array3 + pred_array4 + pred_array5) / 5

    patient_wise_pred_list = []
    patient_wise_label_list = []

    split_df_path = os.path.join(params1['file_path'], 'test', 'QUASAR_deployMSIH', 'org_split.csv')
    df = pd.read_csv(split_df_path, sep=',')

    for index, row in df.iterrows():
        temp = pred_array[row['start_index']: row['end_index'] + 1]
        temp_label = target_array[row['start_index']: row['end_index'] + 1]
        patient_wise_pred_list.append(temp.mean(0))
        patient_wise_label_list.append(temp_label[0])

    patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
    targets = torch.stack(patient_wise_label_list, 0)
    output_sigmoided = F.sigmoid(patient_wise_pred_list)
    max_preds = output_sigmoided.argmax(dim=1)

    ### evaluation metrics
    # accuracy_calculator = torchmetrics.Accuracy()
    # total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()
    #
    # fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
    # aucc = metrics.auc(fpr, tpr)
    #
    # print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')
    #
    # # saving the training and validation stats
    # msg = f'----------------------------------------------------------------------------------------\n' \
    #       f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'
    #
    # with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_deployMSIH_singleepoch_test_result', 'a') as f:
    #     f.write(msg)

    AUC_list, accuracy_list = bootstrapper(max_preds, targets)

    print(f'\t Accuracy: {accuracy_list.mean() * 100:.2f} +- {accuracy_list.std() * 100:.2f}% | AUROC: {AUC_list.mean() * 100:.2f} +- {AUC_list.std() * 100:.2f}%\n')
    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t Accuracy: {accuracy_list.mean() * 100:.2f} +- {accuracy_list.std() * 100:.2f}% | AUROC: {AUC_list.mean() * 100:.2f} +- {AUC_list.std() * 100:.2f}%\n'
    with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_deployMSIH_singleepoch_test_result', 'a') as f:
        f.write(msg)

    auc_df = pd.DataFrame({'AUC': AUC_list})
    auc_df.to_csv(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_deployMSIH_singleepoch_test_result.csv', sep=',', index=False)

    ################### second benchmark: YORKSHIR ##########################

    # Generate test set
    testset_class = data_loader_pathology(params1["cfg_path"], mode='test', benchmark='YORKSHIR_deployMSIH')
    test_dataset = testset_class.provide_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=39,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: YORKSHIR_deployMSIH\n')

    predictor1.setup_model(model=model1)
    predictor2.setup_model(model=model2)
    predictor3.setup_model(model=model3)
    predictor4.setup_model(model=model4)
    predictor5.setup_model(model=model5)

    pred_array1, target_array = predictor1.predict(test_loader)
    pred_array2, _ = predictor2.predict(test_loader)
    pred_array3, _ = predictor3.predict(test_loader)
    pred_array4, _ = predictor4.predict(test_loader)
    pred_array5, _ = predictor5.predict(test_loader)
    pred_array = (pred_array1 + pred_array2 + pred_array3 + pred_array4 + pred_array5) / 5

    patient_wise_pred_list = []
    patient_wise_label_list = []

    split_df_path = os.path.join(params1['file_path'], 'test', 'YORKSHIR_deployMSIH', 'org_split.csv')
    df = pd.read_csv(split_df_path, sep=',')

    for index, row in df.iterrows():
        temp = pred_array[row['start_index']: row['end_index'] + 1]
        temp_label = target_array[row['start_index']: row['end_index'] + 1]
        patient_wise_pred_list.append(temp.mean(0))
        patient_wise_label_list.append(temp_label[0])

    patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
    targets = torch.stack(patient_wise_label_list, 0)
    output_sigmoided = F.sigmoid(patient_wise_pred_list)
    max_preds = output_sigmoided.argmax(dim=1)

    ### evaluation metrics
    # accuracy_calculator = torchmetrics.Accuracy()
    # total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()
    #
    # fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
    # aucc = metrics.auc(fpr, tpr)
    #
    # print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')
    #
    # # saving the training and validation stats
    # msg = f'----------------------------------------------------------------------------------------\n' \
    #       f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'
    #
    # with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/YORKSHIR_deployMSIH_singleepoch_test_results', 'a') as f:
    #     f.write(msg)

    AUC_list, accuracy_list = bootstrapper(max_preds, targets)

    print(f'\t Accuracy: {accuracy_list.mean() * 100:.2f} +- {accuracy_list.std() * 100:.2f}% | AUROC: {AUC_list.mean() * 100:.2f} +- {AUC_list.std() * 100:.2f}%\n')
    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t Accuracy: {accuracy_list.mean() * 100:.2f} +- {accuracy_list.std() * 100:.2f}% | AUROC: {AUC_list.mean() * 100:.2f} +- {AUC_list.std() * 100:.2f}%\n'
    with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/YORKSHIR_deployMSIH_singleepoch_test_results', 'a') as f:
        f.write(msg)

    auc_df = pd.DataFrame({'AUC': AUC_list})
    auc_df.to_csv(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/YORKSHIR_deployMSIH_singleepoch_test_results.csv', sep=',', index=False)
    ################### third benchmark: QUASAR + YORKSHIR ##########################

    # Generate QUASAR test set
    testset_class_QUASAR = data_loader_pathology(params1["cfg_path"], mode='test', benchmark='QUASAR_deployMSIH')
    test_dataset_QUASAR = testset_class_QUASAR.provide_data()
    test_loader_QUASAR = torch.utils.data.DataLoader(test_dataset_QUASAR, batch_size=231,
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    # Generate YORKSHIR test set
    testset_class_YORKSHIR = data_loader_pathology(params1["cfg_path"], mode='test', benchmark='YORKSHIR_deployMSIH')
    test_dataset_YORKSHIR = testset_class_YORKSHIR.provide_data()
    test_loader_YORKSHIR = torch.utils.data.DataLoader(test_dataset_YORKSHIR, batch_size=39,
                                              pin_memory=True, drop_last=False, shuffle=False, num_workers=10)

    print(f'\nBenchmark name: QUASAR + YORKSHIR\n')

    predictor1.setup_model(model=model1)
    predictor2.setup_model(model=model2)
    predictor3.setup_model(model=model3)
    predictor4.setup_model(model=model4)
    predictor5.setup_model(model=model5)

    pred_array1_QUASAR, target_array_QUASAR = predictor1.predict(test_loader_QUASAR)
    pred_array1_YORKSHIR, target_array_YORKSHIR = predictor1.predict(test_loader_YORKSHIR)

    pred_array2_QUASAR, _ = predictor2.predict(test_loader_QUASAR)
    pred_array2_YORKSHIR, _ = predictor2.predict(test_loader_YORKSHIR)

    pred_array3_QUASAR, _ = predictor3.predict(test_loader_QUASAR)
    pred_array3_YORKSHIR, _ = predictor3.predict(test_loader_YORKSHIR)

    pred_array4_QUASAR, _ = predictor4.predict(test_loader_QUASAR)
    pred_array4_YORKSHIR, _ = predictor4.predict(test_loader_YORKSHIR)

    pred_array5_QUASAR, _ = predictor5.predict(test_loader_QUASAR)
    pred_array5_YORKSHIR, _ = predictor5.predict(test_loader_YORKSHIR)

    pred_array_QUASAR = (pred_array1_QUASAR + pred_array2_QUASAR + pred_array3_QUASAR + pred_array4_QUASAR + pred_array5_QUASAR) / 5
    pred_array_YORKSHIR = (pred_array1_YORKSHIR + pred_array2_YORKSHIR + pred_array3_YORKSHIR + pred_array4_YORKSHIR + pred_array5_YORKSHIR) / 5

    patient_wise_pred_list = []
    patient_wise_label_list = []

    split_df_path_QUASAR = os.path.join(params1['file_path'], 'test', 'QUASAR_deployMSIH', 'org_split.csv')
    df_QUASAR = pd.read_csv(split_df_path_QUASAR, sep=',')
    for index, row in df_QUASAR.iterrows():
        temp = pred_array_QUASAR[row['start_index']: row['end_index'] + 1]
        temp_label = target_array_QUASAR[row['start_index']: row['end_index'] + 1]
        patient_wise_pred_list.append(temp.mean(0))
        patient_wise_label_list.append(temp_label[0])

    split_df_path_YORKSHIR = os.path.join(params1['file_path'], 'test', 'YORKSHIR_deployMSIH', 'org_split.csv')
    df_YORKSHIR = pd.read_csv(split_df_path_YORKSHIR, sep=',')
    for index, row in df_YORKSHIR.iterrows():
        temp = pred_array_YORKSHIR[row['start_index']: row['end_index'] + 1]
        temp_label = target_array_YORKSHIR[row['start_index']: row['end_index'] + 1]
        patient_wise_pred_list.append(temp.mean(0))
        patient_wise_label_list.append(temp_label[0])

    patient_wise_pred_list = torch.stack(patient_wise_pred_list, 0)
    targets = torch.stack(patient_wise_label_list, 0)
    output_sigmoided = F.sigmoid(patient_wise_pred_list)
    max_preds = output_sigmoided.argmax(dim=1)

    ### evaluation metrics
    # accuracy_calculator = torchmetrics.Accuracy()
    # total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()
    #
    # fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
    # aucc = metrics.auc(fpr, tpr)
    #
    # print(f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n')
    #
    # # saving the training and validation stats
    # msg = f'----------------------------------------------------------------------------------------\n' \
    #       f'\t Accuracy: {total_accuracy * 100:.2f}% | AUROC: {aucc * 100:.2f}%\n'
    #
    # with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_YORKSHIR_singleepoch_test_results', 'a') as f:
    #     f.write(msg)

    AUC_list, accuracy_list = bootstrapper(max_preds, targets)

    print(f'\t Accuracy: {accuracy_list.mean() * 100:.2f} +- {accuracy_list.std() * 100:.2f}% | AUROC: {AUC_list.mean() * 100:.2f} +- {AUC_list.std() * 100:.2f}%\n')
    # saving the training and validation stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t Accuracy: {accuracy_list.mean() * 100:.2f} +- {accuracy_list.std() * 100:.2f}% | AUROC: {AUC_list.mean() * 100:.2f} +- {AUC_list.std() * 100:.2f}%\n'
    with open(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_YORKSHIR_singleepoch_test_results', 'a') as f:
        f.write(msg)

    auc_df = pd.DataFrame({'AUC': AUC_list})
    auc_df.to_csv(os.path.join(params5['target_dir'], params5['stat_log_path']) + '/ensembled/QUASAR_YORKSHIR_singleepoch_test_results.csv', sep=',', index=False)



def pvalue_out_of_bootstrap(biggerdf_path, smallerdf_path):

    biggerdf = pd.read_csv(biggerdf_path, sep=',')
    smallerdf = pd.read_csv(smallerdf_path, sep=',')

    first_array = np.array(biggerdf['AUC'])
    smaller_array = np.array(smallerdf['AUC'])

    counter = first_array > smaller_array

    ratio = (len(counter) - counter.sum()) / len(counter)

    print('\n\tp-value:', ratio)






if __name__ == '__main__':
    main_train_pathology(global_config_path="/federated_he/pathology/config/config.yaml",
                  resume=False, valid=True, experiment_name='central_lr1e4_batch124_fold1', train_site='central', fold=1)