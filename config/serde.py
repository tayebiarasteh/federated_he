"""
Created on November 10, 2019
functions for writing/reading data to/from disk

@modified_by: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""
import yaml
import numpy as np
import os
import warnings
import shutil
import pdb




def read_config(config_path):
    """Reads config file in yaml format into a dictionary

    Parameters
    ----------
    config_path: str
        Path to the config file in yaml format

    Returns
    -------
    config dictionary
    """

    with open(config_path, 'rb') as yaml_file:
        return yaml.safe_load(yaml_file)


def write_config(params, cfg_path, sort_keys=False):
    with open(cfg_path, 'w') as f:
        yaml.dump(params, f)


def create_experiment(experiment_name, global_config_path):
    params = read_config(global_config_path)
    params['experiment_name'] = experiment_name
    create_experiment_folders(params)
    cfg_file_name = params['experiment_name'] + '_config.yaml'
    cfg_path = os.path.join(os.path.join(params['target_dir'], params['network_output_path']), cfg_file_name)
    params['cfg_path'] = cfg_path
    write_config(params, cfg_path)
    return params


def create_experiment_folders(params):
    try:
        path_keynames = ["network_output_path", "tb_logs_path", "stat_log_path", "output_data_path"]
        for key in path_keynames:
            params[key] = os.path.join(params['experiment_name'], params[key])
            os.makedirs(os.path.join(params['target_dir'], params[key]))
    except:
        raise Exception("Experiment already exist. Please try a different experiment name")


def open_experiment(experiment_name, global_config_path):
    """Open Existing Experiments
    """
    default_params = read_config(global_config_path)
    cfg_file_name = experiment_name + '_config.yaml'
    cfg_path = os.path.join(os.path.join(default_params['target_dir'], experiment_name, default_params['network_output_path']), cfg_file_name)
    params = read_config(cfg_path)
    return params


def delete_experiment(experiment_name, global_config_path):
    """Delete Existing Experiment folder
    """
    default_params = read_config(global_config_path)
    cfg_file_name = experiment_name + '_config.yaml'
    cfg_path = os.path.join(os.path.join(default_params['target_dir'], experiment_name, default_params['network_output_path']), cfg_file_name)
    params = read_config(cfg_path)
    shutil.rmtree(os.path.join(params['target_dir'], experiment_name))
