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

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_pathology import Training
from data.data_provider_pathology import data_loader_pathology
from models.MNISTNET import mnistNet
from Prediction_pathology import Prediction

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15




def main_train_pathology(global_config_path="/home/soroosh/Documents/Repositories/federated_he/pathology/config/config.yaml", valid=False,
                  resume=False, experiment_name='name'):
    """Main function for training + validation for directly 3d-wise

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/federated_he/pathology/config/config.yaml"

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

    trainset_class = data_loader_pathology(params["cfg_path"], mode='train')
    train_dataset = trainset_class.provide_data()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)

    if valid:
        validset_class = data_loader_pathology(params["cfg_path"], mode='valid')
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



def main_test_pathology(global_config_path="/home/soroosh/Documents/Repositories/federated_he/pathology/config/config.yaml",
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

    accuracy_calculator = torchmetrics.Accuracy()
    total_accuracy = accuracy_calculator(max_preds.cpu(), targets.cpu()).item()

    fpr, tpr, thresholds = metrics.roc_curve(targets.cpu().numpy(), max_preds.cpu().numpy(), pos_label=1)
    aucc = metrics.auc(fpr, tpr)

    pdb.set_trace()




if __name__ == '__main__':
    # delete_experiment(experiment_name='tempppnohe', global_config_path="/home/soroosh/Documents/Repositories/federated_he/pathology/config/config.yaml")
    # main_train_pathology(global_config_path="/home/soroosh/Documents/Repositories/federated_he/pathology/config/config.yaml",
    #               resume=False, valid=False, experiment_name='tempppnohe')
    main_test_pathology(global_config_path="/home/soroosh/Documents/Repositories/federated_he/pathology/config/config.yaml",
                  experiment_name='tempppnohe', benchmark='YORKSHIR_deployMSIH')
