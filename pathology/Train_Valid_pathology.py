"""
Created on March 4, 2022.
Train_Valid_brats.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os.path
import time
import pdb
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import torchmetrics
import torch.nn.functional as F
import syft as sy

from config.serde import read_config, write_config

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15
hook = sy.TorchHook(torch)



class Training:
    def __init__(self, cfg_path, num_epochs=10, resume=False):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        num_epochs: int
            Total number of epochs for training

        resume: bool
            if we are resuming training from a checkpoint
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.num_epochs = num_epochs

        if resume == False:
            self.model_info = self.params['Network']
            self.epoch = 0
            self.best_loss = float('inf')
            self.setup_cuda()
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path']))


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.

        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def time_duration(self, start_time, end_time):
        """calculating the duration of training or one iteration

        Parameters
        ----------
        start_time: float
            starting time of the operation

        end_time: float
            ending time of the operation

        Returns
        -------
        elapsed_hours: int
            total hours part of the elapsed time

        elapsed_mins: int
            total minutes part of the elapsed time

        elapsed_secs: int
            total seconds part of the elapsed time
        """
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = elapsed_time - (elapsed_mins * 60)
            # elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_model(self, model, optimiser, loss_function):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function
        """

        # prints the network's total number of trainable parameters and
        # stores it to the experiment config
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal # of trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        self.model = model.to(self.device)
        # self.model = self.model.half() # float16

        self.loss_function = loss_function()
        self.optimiser = optimiser

        # Saves the model, optimiser,loss function name for writing to config file
        # self.model_info['model'] = model.__name__
        # self.model_info['optimiser'] = optimiser.__name__
        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['num_epochs'] = self.num_epochs
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)


    def load_checkpoint(self, model, optimiser, loss_function):
        """In case of resuming training from a checkpoint,
        loads the weights for all the models, optimizers, and
        loss functions, and device, tensorboard events, number
        of iterations (epochs), and every info from checkpoint.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function
        """
        checkpoint = torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                self.params['checkpoint_name']))
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.loss_function = loss_function()
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.epoch + 1)


    def training_init(self, train_loader, valid_loader=None):
        """Training epoch
        """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1

            train_loss = self.train_epoch(train_loader)
            valid_loss = self.valid_epoch(valid_loader)
            pdb.set_trace()


    def train_epoch(self, train_loader):
        """Training epoch
        """

        self.epoch += 1

        # initializing the loss list
        total_loss = 0

        self.model.train()
        for batchIdx, (data, target) in enumerate(train_loader):

            self.optimiser.zero_grad()
            with torch.set_grad_enabled(True):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                loss = self.loss_function(output, target)
                loss.backward()
                self.optimiser.step()

                total_loss += loss.item()

                trainPrint = True
                if trainPrint and batchIdx % 100 == 0:
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.epoch, batchIdx * len(data), len(train_loader.dataset),
                               100. * batchIdx / len(train_loader), loss.item()))

        return total_loss / len(train_loader)



    def valid_epoch(self, valid_loader):
        """valid epoch
        """
        # initializing the loss list
        total_loss = 0

        self.model.eval()
        for batchIdx, (data, target) in enumerate(valid_loader):

            with torch.no_grad():
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_function(output, target)

                total_loss += loss.item()

                trainPrint = True
                if trainPrint and batchIdx % 100 == 0:
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.epoch, batchIdx * len(data), len(valid_loader.dataset),
                               100. * batchIdx / len(valid_loader), loss.item()))

        return total_loss / len(valid_loader)



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs, total_hours,
                       total_mins, total_secs, train_loss, total_time, total_overhead_time=0, total_datacopy_time=0, valid_loss=None, valid_F1=None, valid_accuracy=None,
                       valid_specifity=None, valid_sensitivity=None, valid_precision=None):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        Parameters
        ----------
        iteration_hours: int
            hours part of the elapsed time of each iteration

        iteration_mins: int
            minutes part of the elapsed time of each iteration

        iteration_secs: int
            seconds part of the elapsed time of each iteration

        total_hours: int
            hours part of the total elapsed time

        total_mins: int
            minutes part of the total elapsed time

        total_secs: int
            seconds part of the total elapsed time

        train_loss: float
            training loss of the model

        valid_acc: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specifity: float
            validation specifity of the model

        valid_loss: float
            validation loss of the model
        """

        # Saves information about training to config file
        self.params['Network']['num_epoch'] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        overhead_hours, overhead_mins, overhead_secs = self.time_duration(0, total_overhead_time)
        noncopy_time = total_time - total_datacopy_time
        netto_time = total_time - total_overhead_time - total_datacopy_time
        noncopy_hours, noncopy_mins, noncopy_secs = self.time_duration(0, noncopy_time)
        netto_hours, netto_mins, netto_secs = self.time_duration(0, netto_time)

        # Saving the model based on the best loss
        if valid_loss:
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'],
                                                                 self.params['network_output_path'], self.params['trained_model_name']))
        else:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'],
                                                                 self.params['network_output_path'], self.params['trained_model_name']))

        # Saving every couple of epochs
        if (self.epoch) % self.params['network_save_freq'] == 0:
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path'],
                       'epoch{}_'.format(self.epoch) + self.params['trained_model_name']))

        # Save a checkpoint every epoch
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'loss_state_dict': self.loss_function.state_dict(), 'num_epochs': self.num_epochs,
                    'model_info': self.model_info, 'best_loss': self.best_loss},
                   os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))

        print('------------------------------------------------------'
              '----------------------------------')
        print(f'epoch: {self.epoch} | '
              f'epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s | '
              f'total time: {total_hours}h {total_mins}m {total_secs:.2f}s | communication overhead time so far: {overhead_hours}h {overhead_mins}m {overhead_secs:.2f}s')
        print(f'\n\tTrain loss: {train_loss:.4f}')

        if valid_loss:
            print(f'\t Val. loss: {valid_loss:.4f} | Average Dice score (whole tumor): {valid_F1.mean().item() * 100:.2f}% | accuracy: {valid_accuracy.mean().item() * 100:.2f}%'
            f' | specifity WT: {valid_specifity.mean().item() * 100:.2f}%'
            f' | recall (sensitivity) WT: {valid_sensitivity.mean().item() * 100:.2f}% | precision WT: {valid_precision.mean().item() * 100:.2f}%\n')

            print('Individual Dice scores:')
            print(f'Dice label 1 (necrotic tumor core): {valid_F1[0].item() * 100:.2f}%')
            print(f'Dice label 2 (peritumoral edematous/invaded tissue): {valid_F1[1].item() * 100:.2f}%\n')
            print(f'Dice label 4, i.e., enhancing tumor (ET): {valid_F1[2].item() * 100:.2f}%')
            print(f'Dice average 1 and 4, i.e., tumor core (TC): {(valid_F1[0].item() + valid_F1[2].item()) / 2 * 100:.2f}%')
            print(f'Dice average all 1, 2, 4, i.e., whole tumor (WT): {valid_F1.mean().item() * 100:.2f}%\n')

            # saving the training and validation stats
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s | ' \
                  f'communication overhead time so far: {overhead_hours}h {overhead_mins}m {overhead_secs:.2f}s\n' \
                  f' | total time - copy time: {noncopy_hours}h {noncopy_mins}m {noncopy_secs:.2f}s' \
                  f' | total time - copy time - overhead time: {netto_hours}h {netto_mins}m {netto_secs:.2f}s' \
                  f'\n\n\tTrain loss: {train_loss:.4f} | ' \
                   f'Val. loss: {valid_loss:.4f} | Average Dice score (whole tumor): {valid_F1.mean().item() * 100:.2f}% | accuracy: {valid_accuracy.mean().item() * 100:.2f}% ' \
                   f' | specifity WT: {valid_specifity.mean().item() * 100:.2f}%' \
                   f' | recall (sensitivity) WT: {valid_sensitivity.mean().item() * 100:.2f}% | precision WT: {valid_precision.mean().item() * 100:.2f}%\n\n' \
                   f'  Dice label 1 (necrotic tumor core): {valid_F1[0].item() * 100:.2f}% | ' \
                   f'Dice label 2 (peritumoral edematous/invaded tissue): {valid_F1[1].item() * 100:.2f}%\n\n' \
                   f'- Dice label 4, i.e., enhancing tumor (ET): {valid_F1[2].item() * 100:.2f}%\n' \
                   f'- Dice average 1 and 4, i.e., tumor core (TC): {(valid_F1[0].item() + valid_F1[2].item())/2 * 100:.2f}%\n' \
                   f'- Dice average all 1, 2, 4, i.e., whole tumor (WT): {valid_F1.mean().item() * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n\n\ttrain loss: {train_loss:.4f}' \
                  f' | communication overhead time so far: {overhead_hours}h {overhead_mins}m {overhead_secs:.2f}s\n' \
                  f' | total time - copy time: {noncopy_hours}h {noncopy_mins}m {noncopy_secs:.2f}s' \
                  f' | total time - copy time - overhead time: {netto_hours}h {netto_mins}m {netto_secs:.2f}s\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def calculate_tb_stats(self, valid_loss=None, valid_F1=None, valid_accuracy=None, valid_specifity=None, valid_sensitivity=None, valid_precision=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        Parameters
        ----------
        valid_acc: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specifity: float
            validation specifity of the model

        valid_loss: float
            validation loss of the model
        """
        if valid_loss is not None:
            self.writer.add_scalar('Valid_loss', valid_loss, self.epoch)
            self.writer.add_scalar('Valid_specifity_WT', valid_specifity.mean().item(), self.epoch)
            self.writer.add_scalar('Valid_Dice_WT', valid_F1.mean().item(), self.epoch)
            self.writer.add_scalar('Valid_Dice_TC', ((valid_F1[0] + valid_F1[2])/ 2).item(), self.epoch)
            self.writer.add_scalar('Valid_Dice_label_4_ET', valid_F1[2].item(), self.epoch)
            self.writer.add_scalar('Valid_Dice_label_1', valid_F1[0].item(), self.epoch)
            self.writer.add_scalar('Valid_Dice_label_2', valid_F1[1].item(), self.epoch)
            self.writer.add_scalar('Valid_precision_WT', valid_precision.mean().item(), self.epoch)
            self.writer.add_scalar('Valid_recall_sensitivity_WT', valid_sensitivity.mean().item(), self.epoch)