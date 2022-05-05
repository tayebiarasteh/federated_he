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
        """
        """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1

            start_time = time.time()

            train_loss = self.train_epoch(train_loader)

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    valid_loss, valid_accuracy = self.valid_epoch(valid_loader)
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(train_loss=train_loss, valid_loss=valid_loss, valid_accuracy=valid_accuracy)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time, valid_loss=valid_loss, valid_accuracy=valid_accuracy)
                else:
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(train_loss=train_loss)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time)



    def train_epoch(self, train_loader):
        """Training epoch
        """
        total_loss = 0

        self.model.train()
        for batchIdx, (image, label) in enumerate(train_loader):

            self.optimiser.zero_grad()
            with torch.set_grad_enabled(True):
                image, label = image.to(self.device), label.to(self.device)
                output = self.model(image)

                loss = self.loss_function(output, label)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()

        return total_loss / len(train_loader)



    def training_setup_federated(self, train_loader, valid_loader=None, HE=False, precision_fractional=15):
        """

        Parameters
        ----------
        train_loader
        valid_loader

        HE: bool
            if we want to have homomorphic encryption when aggregating the weights

        precision_fractional: int
            number of decimal points we want to have when encoding decimal to binary for HE
            for lossless encoding: encoded_num > 2 ** 63 (if the original number is long)
        """
        self.params = read_config(self.cfg_path)

        client_list = []

        for idx in range(len(train_loader)):
            # create a couple workers
            client_list.append(sy.VirtualWorker(hook, id="client" + str(idx)))
        secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        if len(train_loader) == 2:
            client_list[0].add_workers([client_list[1], secure_worker])
            client_list[1].add_workers([client_list[0], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1]])

        elif len(train_loader) == 3:
            client_list[0].add_workers([client_list[1], client_list[2], secure_worker])
            client_list[1].add_workers([client_list[0], client_list[2], secure_worker])
            client_list[2].add_workers([client_list[0], client_list[1], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1], client_list[2]])

        elif len(train_loader) == 4:
            client_list[0].add_workers([client_list[1], client_list[2], client_list[3], secure_worker])
            client_list[1].add_workers([client_list[0], client_list[2], client_list[3], secure_worker])
            client_list[2].add_workers([client_list[0], client_list[1], client_list[3], secure_worker])
            client_list[3].add_workers([client_list[0], client_list[1], client_list[2], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1], client_list[2], client_list[3]])

        elif len(train_loader) == 5:
            client_list[0].add_workers([client_list[1], client_list[2], client_list[3], client_list[4], secure_worker])
            client_list[1].add_workers([client_list[0], client_list[2], client_list[3], client_list[4], secure_worker])
            client_list[2].add_workers([client_list[0], client_list[1], client_list[3], client_list[4], secure_worker])
            client_list[3].add_workers([client_list[0], client_list[1], client_list[2], client_list[4], secure_worker])
            client_list[4].add_workers([client_list[0], client_list[1], client_list[2], client_list[3], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1], client_list[2], client_list[3], client_list[4]])

        elif len(train_loader) == 6:
            client_list[0].add_workers([client_list[1], client_list[2], client_list[3], client_list[4], client_list[5], secure_worker])
            client_list[1].add_workers([client_list[0], client_list[2], client_list[3], client_list[4], client_list[5], secure_worker])
            client_list[2].add_workers([client_list[0], client_list[1], client_list[3], client_list[4], client_list[5], secure_worker])
            client_list[3].add_workers([client_list[0], client_list[1], client_list[2], client_list[4], client_list[5], secure_worker])
            client_list[4].add_workers([client_list[0], client_list[1], client_list[2], client_list[3], client_list[5], secure_worker])
            client_list[5].add_workers([client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], client_list[5]])

        self.state_dict_list = []
        for name in self.model.state_dict():
            self.state_dict_list.append(name)

        total_start_time = time.time()
        total_overhead_time = 0
        total_datacopy_time = 0

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1

            start_time = time.time()
            epoch_overhead_time = 0
            epoch_datacopy_time = 0
            self.model.train()

            secure_worker.clear_objects()
            model_client_list = []
            optimizer_client_list = []
            new_model_client_list = []
            loss_client_list = []
            for idx in range(len(train_loader)):
                communication_start_time = time.time()
                client_list[idx].clear_objects()
                model_client_list.append(self.model.copy().send(client_list[idx]))
                optimizer_client_list.append(torch.optim.Adam(model_client_list[idx].parameters(), lr=float(self.params['Network']['lr']),
                                                     weight_decay=float(self.params['Network']['weight_decay']),
                                                     amsgrad=self.params['Network']['amsgrad']))
                total_overhead_time += (time.time() - communication_start_time)
                epoch_overhead_time += (time.time() - communication_start_time)

                new_model_client, loss_client, overhead = self.train_epoch_federated(train_loader[idx], optimizer_client_list[idx], model_client_list[idx])
                total_datacopy_time += overhead
                epoch_datacopy_time += overhead
                new_model_client_list.append(new_model_client)
                loss_client_list.append(loss_client)

            communication_start_time = time.time()
            temp_dict = {}
            if HE:
                for weightbias in self.state_dict_list:
                    temp_one_param_list = []

                    if len(train_loader) == 2:
                        temp_one_param_list.append(new_model_client_list[0].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[1].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], crypto_provider=secure_worker).get())
                        if 'num_batches_tracked' in weightbias:
                            temp_dict[weightbias] = torch.round((temp_one_param_list[0] + temp_one_param_list[1]).get().float_precision() / 2)
                        else:
                            temp_dict[weightbias] = (temp_one_param_list[0] + temp_one_param_list[1]).get().float_precision() / 2

                    elif len(train_loader) == 3:
                        temp_one_param_list.append(new_model_client_list[0].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[1].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[2].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], crypto_provider=secure_worker).get())
                        if 'num_batches_tracked' in weightbias:
                            temp_dict[weightbias] = torch.round((temp_one_param_list[0] + temp_one_param_list[1] +
                                                     temp_one_param_list[2]).get().float_precision() / 3)
                        else:
                            temp_dict[weightbias] = (temp_one_param_list[0] + temp_one_param_list[1] +
                                                     temp_one_param_list[2]).get().float_precision() / 3

                    elif len(train_loader) == 4:
                        temp_one_param_list.append(new_model_client_list[0].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[1].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[2].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[3].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], crypto_provider=secure_worker).get())
                        if 'num_batches_tracked' in weightbias:
                            temp_dict[weightbias] = torch.round((temp_one_param_list[0] + temp_one_param_list[1] +
                                                     temp_one_param_list[2] + temp_one_param_list[3]).get().float_precision() / 4)
                        else:
                            temp_dict[weightbias] = (temp_one_param_list[0] + temp_one_param_list[1] +
                                                     temp_one_param_list[2] + temp_one_param_list[3]).get().float_precision() / 4

                    elif len(train_loader) == 5:
                        temp_one_param_list.append(new_model_client_list[0].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[1].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[2].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[3].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[4].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], crypto_provider=secure_worker).get())
                        if 'num_batches_tracked' in weightbias:
                            temp_dict[weightbias] = torch.round((temp_one_param_list[0] + temp_one_param_list[1] + temp_one_param_list[2] +
                                                     temp_one_param_list[3] + temp_one_param_list[4]).get().float_precision() / 5)
                        else:
                            temp_dict[weightbias] = (temp_one_param_list[0] + temp_one_param_list[1] + temp_one_param_list[2] +
                                                     temp_one_param_list[3] + temp_one_param_list[4]).get().float_precision() / 5

                    elif len(train_loader) == 6:
                        temp_one_param_list.append(new_model_client_list[0].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], client_list[5], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[1].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], client_list[5], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[2].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], client_list[5], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[3].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], client_list[5], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[4].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], client_list[5], crypto_provider=secure_worker).get())
                        temp_one_param_list.append(new_model_client_list[5].state_dict()[weightbias].fix_precision(precision_fractional=precision_fractional).share(
                            client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], client_list[5], crypto_provider=secure_worker).get())
                        if 'num_batches_tracked' in weightbias:
                            temp_dict[weightbias] = torch.round((temp_one_param_list[0] + temp_one_param_list[1] + temp_one_param_list[2] +
                                                     temp_one_param_list[3] + temp_one_param_list[4] + client_list[5]).get().float_precision() / 6)
                        else:
                            temp_dict[weightbias] = (temp_one_param_list[0] + temp_one_param_list[1] + temp_one_param_list[2] +
                                                     temp_one_param_list[3] + temp_one_param_list[4] + client_list[5]).get().float_precision() / 6

            else:
                for idx in range(len(train_loader)):
                    new_model_client_list[idx].move(secure_worker)

                for weightbias in self.state_dict_list:
                    temp_weight_list = []
                    for idx in range(len(train_loader)):
                        temp_weight_list.append(new_model_client_list[idx].state_dict()[weightbias])
                    temp_dict[weightbias] = (sum(temp_weight_list) / len(temp_weight_list)).clone().get()

            self.model.load_state_dict(temp_dict)
            total_overhead_time += (time.time() - communication_start_time)
            epoch_overhead_time += (time.time() - communication_start_time)

            epoch_overhead_hours, epoch_overhead_mins, epoch_overhead_secs = self.time_duration(0, epoch_overhead_time)
            epoch_datacopy_hours, epoch_datacopy_mins, epoch_datacopy_secs = self.time_duration(0, epoch_datacopy_time)
            total_datacopy_hours, total_datacopy_mins, total_datacopy_secs = self.time_duration(0, total_datacopy_time)

            # train loss just as an average of client losses
            train_loss = sum(loss_client_list) / len(loss_client_list)

            # Prints train loss after number of steps specified.
            end_time = time.time()
            iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
            total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

            print('------------------------------------------------------'
                  '----------------------------------')
            print(f'train epoch {self.epoch} | time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s',
                  f'| total: {total_hours}h {total_mins}m {total_secs:.2f}s | epoch communication overhead time: {epoch_overhead_hours}h {epoch_overhead_mins}m {epoch_overhead_secs:.2f}s '
                  f'\nepoch data copying time: {epoch_datacopy_hours}h {epoch_datacopy_mins}m {epoch_datacopy_secs:.2f}s '
                  f'| total data copying time: {total_datacopy_hours}h {total_datacopy_mins}m {total_datacopy_secs:.2f}s\n')

            for idx in range(len(train_loader)):
                print('loss client{}: {:.3f}'.format((idx + 1), loss_client_list[idx]))
                self.writer.add_scalar('Train_loss_client' + str(idx + 1), loss_client_list[idx], self.epoch)

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    valid_loss, valid_accuracy = self.valid_epoch(valid_loader)
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(train_loss=train_loss, valid_loss=valid_loss, valid_accuracy=valid_accuracy)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours, total_mins,
                                        total_secs, train_loss, total_time, total_overhead_time, total_datacopy_time, valid_loss=valid_loss, valid_accuracy=valid_accuracy)
                else:
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
                    self.calculate_tb_stats(train_loss=train_loss)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time, total_overhead_time, total_datacopy_time)



    def train_epoch_federated(self, train_loader, optimizer, model):
        """Training epoch
        """
        batch_loss = 0
        epoch_datacopy = 0

        model.train()
        for batchIdx, (image, label) in enumerate(train_loader):

            communication_start_time = time.time()
            loc = model.location
            image = image.send(loc)
            label = label.send(loc)
            epoch_datacopy += (time.time() - communication_start_time)
            image = image.to(self.device)
            label = label.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                output = model(image)
                loss_client = self.loss_function(output, label)
                loss_client.backward()
                optimizer.step()
                batch_loss += loss_client

        batch_loss = batch_loss.get().data
        avg_loss = batch_loss / len(train_loader)

        return model, avg_loss.item(), epoch_datacopy



    def valid_epoch(self, valid_loader):
        """valid epoch
        """
        total_loss = 0
        total_accuracy = 0

        self.model.eval()
        for batchIdx, (image, label) in enumerate(valid_loader):

            with torch.no_grad():
                image, label = image.to(self.device), label.to(self.device)
                output = self.model(image)
                loss = self.loss_function(output, label)

                total_loss += loss.item()

                output_sigmoided = F.sigmoid(output)
                max_preds = output_sigmoided.argmax(dim=1)

                accuracy_calculator = torchmetrics.Accuracy().to(self.device)
                total_accuracy += accuracy_calculator(max_preds, label).item()

        final_accuracy = total_accuracy / len(valid_loader)

        return total_loss / len(valid_loader), final_accuracy



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs, total_hours,
                       total_mins, total_secs, train_loss, total_time, total_overhead_time=0, total_datacopy_time=0, valid_loss=None, valid_accuracy=None):
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
            print(f'\t Val. loss: {valid_loss:.4f} | accuracy: {valid_accuracy * 100:.2f}%\n')

            # saving the training and validation stats
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s | ' \
                  f'communication overhead time so far: {overhead_hours}h {overhead_mins}m {overhead_secs:.2f}s\n' \
                  f' | total time - copy time: {noncopy_hours}h {noncopy_mins}m {noncopy_secs:.2f}s' \
                  f' | total time - copy time - overhead time: {netto_hours}h {netto_mins}m {netto_secs:.2f}s' \
                  f'\n\n\tTrain loss: {train_loss:.4f} | ' \
                   f'Val. loss: {valid_loss:.4f} | accuracy: {valid_accuracy * 100:.2f}% \n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n\n\ttrain loss: {train_loss:.4f}' \
                  f' | communication overhead time so far: {overhead_hours}h {overhead_mins}m {overhead_secs:.2f}s\n' \
                  f' | total time - copy time: {noncopy_hours}h {noncopy_mins}m {noncopy_secs:.2f}s' \
                  f' | total time - copy time - overhead time: {netto_hours}h {netto_mins}m {netto_secs:.2f}s\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def calculate_tb_stats(self, train_loss=None, valid_loss=None, valid_accuracy=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        """
        self.writer.add_scalar('train_loss', train_loss, self.epoch)
        if valid_loss is not None:
            self.writer.add_scalar('valid_loss', valid_loss, self.epoch)
            self.writer.add_scalar('valid_accuracy', valid_accuracy, self.epoch)
