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
import copy

from config.serde import read_config, write_config
from data.augmentation_brats import random_augment
from models.EDiceLoss_loss import EDiceLoss
from models.UNet3D import UNet3D

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15
hook = sy.TorchHook(torch)



class Training:
    def __init__(self, cfg_path, num_epochs=10, resume=False, augment=False):
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
        self.augment = augment

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
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_model(self, model, optimiser, loss_function, weight=None):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function

        weight: 1D tensor of float
            class weights
        """

        # prints the network's total number of trainable parameters and
        # stores it to the experiment config
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal # of trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        self.model = model.to(self.device)
        if not weight==None:
            # self.loss_weight = weight.to(self.device)
            # self.loss_function = loss_function(self.loss_weight) # for binary
            # self.loss_function = loss_function(pos_weight=self.loss_weight) # for multi label
            self.loss_function = loss_function()
        else:
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


    def load_checkpoint(self, model, optimiser, loss_function, weight=None):
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
        # self.loss_weight = weight.to(self.device)
        # self.loss_function = loss_function(self.loss_weight)
        self.loss_function = loss_function()
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.epoch + 1)



    def train_epoch(self, train_loader, valid_loader=None):
        """Training epoch
        """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1

            # initializing the loss list
            batch_loss = 0
            start_time = time.time()

            for idx, (image, label) in enumerate(train_loader):
                self.model.train()

                # if we would like to have data augmentation during training
                if self.augment:
                    image, label = random_augment(image, label, self.cfg_path)

                label = label.long()
                image = image.float()
                image = image.to(self.device)
                label = label.to(self.device)

                self.optimiser.zero_grad()

                with torch.set_grad_enabled(True):

                    output = self.model(image)

                    # loss = self.loss_function(output, label[:, 0]) # for cross entropy loss
                    loss = self.loss_function(output, label) # for dice loss

                    loss.backward()
                    self.optimiser.step()

                    batch_loss += loss.item()

            # Prints train loss after number of steps specified.
            train_loss = batch_loss / len(train_loader)
            self.writer.add_scalar('Train_loss', train_loss, self.epoch)

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    valid_loss, valid_F1, valid_accuracy, valid_specifity, valid_sensitivity, valid_precision = self.valid_epoch(valid_loader)
                    end_time = time.time()
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(valid_loss=valid_loss, valid_F1=valid_F1, valid_accuracy=valid_accuracy, valid_specifity=valid_specifity,
                                            valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, valid_loss=valid_loss,
                                        valid_F1=valid_F1, valid_accuracy=valid_accuracy, valid_specifity= valid_specifity,
                                        valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                else:
                    end_time = time.time()
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss)



    def training_setup_federated(self, train_loader, valid_loader=None):
        """
        """
        self.params = read_config(self.cfg_path)

        # create a couple workers
        client1 = sy.VirtualWorker(hook, id="client1")
        client2 = sy.VirtualWorker(hook, id="client2")
        client3 = sy.VirtualWorker(hook, id="client3")
        secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        # client1.clear_objects()
        # client2.clear_objects()
        # client3.clear_objects()
        # secure_worker.clear_objects()

        client1.add_workers([client2, client3, secure_worker])
        client2.add_workers([client1, client3, secure_worker])
        client3.add_workers([client1, client2, secure_worker])
        secure_worker.add_workers([client1, client2, client3])

        # newtrain_loader_client1 = []
        # for batch_idx, (data, target) in enumerate(tqdm(train_loader[0])):
        #     data = data.send(client1)
        #     target = target.send(client1)
        #     newtrain_loader_client1.append((data, target))
        # print('\nclient 1 done!')
        #
        state_dict_list = []
        for name in self.model.state_dict():
            state_dict_list.append(name)

        total_start_time = time.time()
        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1

            start_time = time.time()
            self.model.train()

            client1.clear_objects()
            client2.clear_objects()
            client3.clear_objects()
            secure_worker.clear_objects()

            # model_client1 = copy.deepcopy(self.model).send(client1)
            # model_client2 = copy.deepcopy(self.model).send(client2)
            # model_client3 = copy.deepcopy(self.model).send(client3)

            # print('self model:', self.model.output_block.conv_out.weight.data.sum().item())

            # model_client1 = self.model.copy()
            # model_client2 = self.model.copy()
            # model_client3 = self.model.copy()
            model_client1 = self.model.copy().send(client1)
            model_client2 = self.model.copy().send(client2)
            model_client3 = self.model.copy().send(client3)

            # print('model_client1:', model_client1.output_block.conv_out.weight.data.sum().clone().get().item())

            # print(len(client1._objects))
            # print(len(client2._objects))

            optimizer_client1 = torch.optim.Adam(model_client1.parameters(), lr=float(self.params['Network']['lr']),
                                         weight_decay=float(self.params['Network']['weight_decay']),
                                                 amsgrad=self.params['Network']['amsgrad'])
            optimizer_client2 = torch.optim.Adam(model_client2.parameters(), lr=float(self.params['Network']['lr']),
                                         weight_decay=float(self.params['Network']['weight_decay']),
                                                 amsgrad=self.params['Network']['amsgrad'])
            optimizer_client3 = torch.optim.Adam(model_client3.parameters(), lr=float(self.params['Network']['lr']),
                                         weight_decay=float(self.params['Network']['weight_decay']),
                                                 amsgrad=self.params['Network']['amsgrad'])

            model_client1, loss_client1 = self.train_epoch_federated(train_loader[0], optimizer_client1, model_client1)
            model_client2, loss_client2 = self.train_epoch_federated(train_loader[1], optimizer_client2, model_client2)
            model_client3, loss_client3 = self.train_epoch_federated(train_loader[2], optimizer_client3, model_client3)


            # loss_client1 = 0
            # loss_client2 = 0
            # loss_client3 = 0
            model_client1.move(secure_worker)
            model_client2.move(secure_worker)
            model_client3.move(secure_worker)
            # model_client1 = model_client1.get()
            # pdb.set_trace()
            # model_client1 = model_client1.to(self.device)
            # self.model.load_state_dict(model_client1.state_dict())
            # pdb.set_trace()
            # self.model = model_client1.copy()
            # self.model.load_state_dict(model_client1.state_dict())

            # model = torch.nn.Linear(2,1)
            # model2 = torch.nn.Linear(2,1)
            # # for name, param in model.named_parameters():
            # #     pdb.set_trace()
            # #     asd=234
            # print(model.weight.data)
            # print(model.bias.data)
            # print(model2.weight.data)
            # print(model2.bias.data)
            # model2 = model2.send(client1)
            #
            # with torch.no_grad():
            #     for param, param2 in zip(model.parameters(), model2.parameters()):
            #         param.set_((param2 +1).get())
            #

            # pdb.set_trace()
            temp_dict = {}
            for weightbias in state_dict_list:
                # model_temp.state_dict()[weightbias] = model_client1.state_dict()[weightbias]
                # temp_dict[weightbias] = (model_client1.state_dict()[weightbias] + model_client2.state_dict()[weightbias] + model_client3.state_dict()[weightbias]) / 3
                # temp_dict[weightbias] = model_client1.state_dict()[weightbias].clone().get()
                temp_dict[weightbias] = ((model_client1.state_dict()[weightbias] + model_client2.state_dict()[weightbias] + model_client3.state_dict()[weightbias]) / 3).clone().get()

            self.model.load_state_dict(temp_dict)
            # model_client1.state_dict()['input_block.in_double_conv1.conv.0.bias'] = name
            # with torch.no_grad():
            #     for (name, param), (name1, param1), (name2, param2), (name3, param3) in zip(self.model.named_parameters(), model_client1.named_parameters(), model_client2.named_parameters(), model_client3.named_parameters()):
            #         print(name == name1, name == name2)
            #         pdb.set_trace()
            #
            #         param.set_(((param1 + param2 + param3) / 3).get())

            # print('model_client1 after:', model_client1.output_block.conv_out.weight.data.sum().clone().get().item())

            # with torch.no_grad():
            #     for param, param1, param2, param3 in zip(self.model.parameters(), model_client1.parameters(), model_client2.parameters(), model_client3.parameters()):
            #         param.set_(param1)
                    # param.set_((param1 + param2 + param3) / 3)
                    # param.set_(((param1 + param2 + param3) / 3).get())
                    # param.set_((param1).get())

            # print('self model after:', self.model.output_block.conv_out.weight.data.sum().item())

            # train loss just as an average of client losses
            train_loss = (loss_client1 + loss_client2 + loss_client3) / 3

            # Prints train loss after number of steps specified.
            end_time = time.time()
            iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
            total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

            print('train epoch {} | loss client1: {:.3f} | loss client2: {:.3f} | loss client3: {:.3f}'.
                  format(self.epoch, loss_client1, loss_client2, loss_client3),
                  f'\ntime: {iteration_hours}h {iteration_mins}m {iteration_secs}s',
                  f'| total: {total_hours}h {total_mins}m {total_secs}s\n')
            self.writer.add_scalar('Train_loss_client1', loss_client1, self.epoch)
            self.writer.add_scalar('Train_loss_client2', loss_client2, self.epoch)
            self.writer.add_scalar('Train_loss_client3', loss_client3, self.epoch)

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    valid_loss, valid_F1, valid_accuracy, valid_specifity, valid_sensitivity, valid_precision = self.valid_epoch(valid_loader)
                    end_time = time.time()
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(valid_loss=valid_loss, valid_F1=valid_F1, valid_accuracy=valid_accuracy,
                                            valid_specifity=valid_specifity, valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours, total_mins,
                                        total_secs, train_loss, valid_loss=valid_loss, valid_F1=valid_F1,
                                        valid_accuracy=valid_accuracy, valid_specifity=valid_specifity,
                                        valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                else:
                    end_time = time.time()
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss)



    def train_epoch_federated(self, train_loader, optimizer, model):
        """Training epoch
        """

        batch_loss = 0
        model.train()

        # training epoch of a client
        for idx, (image, label) in enumerate(train_loader):

            # if we would like to have data augmentation during training
            if self.augment:
                image, label = random_augment(image, label, self.cfg_path)

            loc = model.location
            image = image.send(loc)
            label = label.send(loc)

            label = label.long()
            image = image.float()
            image = image.to(self.device)
            label = label.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                output = model(image)
                loss_client = self.loss_function(output, label)  # for dice loss

                loss_client.backward()
                optimizer.step()

                batch_loss += loss_client

        batch_loss = batch_loss.get().data
        avg_loss = batch_loss / len(train_loader)

        return model, avg_loss.item()




    def valid_epoch(self, valid_loader):
        """Validation epoch

        """
        self.model.eval()
        total_loss = 0.0
        total_f1_score = []
        total_accuracy = []
        total_specifity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        for idx, (image, label) in enumerate(valid_loader):
            label = label.long()
            image = image.float()
            image = image.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                # loss = self.loss_function(output, label[:, 0]) # for cross entropy loss
                loss = self.loss_function(output, label)  # for dice loss

                # max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability (multi-class)
                output_sigmoided = F.sigmoid(output.permute(0, 2, 3, 4, 1))
                output_sigmoided = (output_sigmoided > 0.5).float()

            ############ Evaluation metric calculation ########
            total_loss += loss.item()

            # Metrics calculation (macro) over the whole set
            confusioner = torchmetrics.ConfusionMatrix(num_classes=label.shape[1], multilabel=True).to(self.device)
            confusion = confusioner(output_sigmoided.flatten(start_dim=0, end_dim=3), label.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=3))

            F1_disease = []
            accuracy_disease = []
            specifity_disease = []
            sensitivity_disease = []
            precision_disease = []

            for idx, disease in enumerate(confusion):
                TN = disease[0, 0]
                FP = disease[0, 1]
                FN = disease[1, 0]
                TP = disease[1, 1]
                F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
                accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
                specifity_disease.append(TN / (TN + FP + epsilon))
                sensitivity_disease.append(TP / (TP + FN + epsilon))
                precision_disease.append(TP / (TP + FP + epsilon))

            # Macro averaging
            total_f1_score.append(torch.stack(F1_disease))
            total_accuracy.append(torch.stack(accuracy_disease))
            total_specifity_score.append(torch.stack(specifity_disease))
            total_sensitivity_score.append(torch.stack(sensitivity_disease))
            total_precision_score.append(torch.stack(precision_disease))

        average_loss = total_loss / len(valid_loader)
        average_f1_score = torch.stack(total_f1_score).mean(0)
        average_accuracy = torch.stack(total_accuracy).mean(0)
        average_specifity = torch.stack(total_specifity_score).mean(0)
        average_sensitivity = torch.stack(total_sensitivity_score).mean(0)
        average_precision = torch.stack(total_precision_score).mean(0)

        return average_loss, average_f1_score, average_accuracy, average_specifity, average_sensitivity, average_precision



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs, total_hours,
                       total_mins, total_secs, train_loss, valid_loss=None, valid_F1=None, valid_accuracy=None,
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
              f'epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs}s | '
              f'total time: {total_hours}h {total_mins}m {total_secs}s')
        print(f'\n\tTrain loss: {train_loss:.4f}')

        if valid_loss:
            print(f'\t Val. loss: {valid_loss:.4f} | F1 (Dice score): {valid_F1.mean().item() * 100:.2f}% | accuracy: {valid_accuracy.mean().item() * 100:.2f}%'
            f' | specifity: {valid_specifity.mean().item() * 100:.2f}%'
            f' | recall (sensitivity): {valid_sensitivity.mean().item() * 100:.2f}% | precision: {valid_precision.mean().item() * 100:.2f}%\n')

            print('Individual F1 scores:')
            print(f'Class 1: {valid_F1[0].item() * 100:.2f}%')
            print(f'Class 2: {valid_F1[1].item() * 100:.2f}%')
            print(f'Class 3: {valid_F1[2].item() * 100:.2f}%\n')

            # saving the training and validation stats
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain loss: {train_loss:.4f} | ' \
                   f'Val. loss: {valid_loss:.4f} | F1 (Dice score): {valid_F1.mean().item() * 100:.2f}% | accuracy: {valid_accuracy.mean().item() * 100:.2f}% ' \
                   f' | specifity: {valid_specifity.mean().item() * 100:.2f}%' \
                   f' | recall (sensitivity): {valid_sensitivity.mean().item() * 100:.2f}% | precision: {valid_precision.mean().item() * 100:.2f}%\n\n' \
                   f' | F1 class 1: {valid_F1[0].item() * 100:.2f}% | F1 class 2: {valid_F1[1].item() * 100:.2f}% | F1 class 3: {valid_F1[2].item() * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs}s\n\n\ttrain loss: {train_loss:.4f}\n\n'
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
            self.writer.add_scalar('Valid_specifity', valid_specifity.mean().item(), self.epoch)
            self.writer.add_scalar('Valid_F1', valid_F1.mean().item(), self.epoch)
            self.writer.add_scalar('Valid_F1 class 1', valid_F1[0].item(), self.epoch)
            self.writer.add_scalar('Valid_F1 class 2', valid_F1[1].item(), self.epoch)
            self.writer.add_scalar('Valid_F1 class 3', valid_F1[2].item(), self.epoch)
            self.writer.add_scalar('Valid_precision', valid_precision.mean().item(), self.epoch)
            self.writer.add_scalar('Valid_recall_sensitivity', valid_sensitivity.mean().item(), self.epoch)