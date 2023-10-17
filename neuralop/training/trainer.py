import torch
from torch.cuda import amp
from timeit import default_timer
import sys 
import wandb

import neuralop.mpu.comm as comm

from .losses import LpLoss
from .callbacks import PipelineCallback
from .algo import Incremental

class Trainer:
    def __init__(self, *, 
                 model, 
                 n_epochs, 
                 output_field_indices=None, 
                 wandb_log=True, 
                 device=None, 
                 amp_autocast=False, 
                 callbacks = None,
                 log_test_interval=1, 
                 log_output=False, 
                 use_distributed=False, 
                 verbose=True, incremental = False, incremental_loss_gap = False, incremental_resolution = False, 
                 dataset_name = None, save_interval=2, model_save_dir='./checkpoints'):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        output_field_indices : dict | None
            if a model has multiple output fields, this maps to
            the indices of a model's output associated with each field. 
        wandb_log : bool, default is True
        device : torch.device
        amp_autocast : bool, default is False
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is True
        """

        if callbacks:
            assert type(callbacks) == list, "Callbacks must be a list of Callback objects"
            self.callbacks = PipelineCallback(callbacks=callbacks)
            self.override_load_to_device = (self.callbacks.device_load_callback_idx is not None)
            self.overrides_loss = self.callbacks.overrides_loss
        else:
            self.callbacks = []
            self.override_load_to_device = False
            self.overrides_loss = False
        
        if verbose:
            print(f"{self.override_load_to_device=}")
            print(f"{self.overrides_loss=}")

        if self.callbacks:
            self.callbacks.on_init_start(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose, incremental = False, incremental_loss_gap = False, incremental_resolution = False, dataset_name = None, save_interval=2, model_save_dir='./checkpoints')
        
        self.incremental_loss_gap = incremental_loss_gap
        self.incremental_grad = incremental
        self.incremental_resolution = incremental_resolution
        self.incremental = self.incremental_loss_gap or self.incremental_grad
        self.dataset_name = dataset_name
        self.save_interval = save_interval
        self.model_save_dir = model_save_dir
        self.save = False
        
        if self.incremental or self.incremental_resolution:
            self.incremental_scheduler = Incremental(model, incremental = self.incremental_grad, incremental_loss_gap = self.incremental_loss_gap, incremental_resolution = self.incremental_resolution, dataset_name = self.dataset_name)
            self.index = 1
                    
        self.model = model
        self.n_epochs = n_epochs

        if not output_field_indices:
            self.output_field_indices = {'':None}
        else:
            self.output_field_indices = output_field_indices
        self.output_fields = list(self.output_field_indices.keys())

        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.amp_autocast = amp_autocast
        
        if self.callbacks:
            self.callbacks.on_init_end(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose, incremental = False, incremental_loss_gap = False, incremental_resolution = False, dataset_name = None, save_interval=2, model_save_dir='./checkpoints')
        
        
    def train(self, train_loader, test_loaders,
            optimizer, scheduler=[None, None], regularizer=None,
              training_loss=None, eval_losses=None):
        
        """Trains the given model on the given datasets.
        params:
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: function to use 
        """

        if self.callbacks:
            self.callbacks.on_train_start(train_loader=train_loader, test_loaders=test_loaders,
                                    optimizer=optimizer, scheduler=scheduler, 
                                    regularizer=regularizer, training_loss=training_loss, 
                                    eval_losses=eval_losses)

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 
        
        for epoch in range(self.n_epochs):

            if self.callbacks:
                self.callbacks.on_epoch_start(epoch=epoch)

            avg_loss = 0
            avg_lasso_loss = 0
            self.model.train()
            t1 = default_timer()
            train_err = 0.0
            batch_size = 10
            S = 128
            self.index = 1
            for idx, sample in enumerate(train_loader):

                if self.callbacks:
                    if self.dataset_name == 'Burgers' or self.dataset_name == 'Re5000':
                        x, y = sample[0], sample[1]
                    else:
                        x, y = sample['x'], sample['y']
                    if self.dataset_name == 'Re5000':
                        x = x.to(self.device).view(batch_size, 1, S, S)
                        y = y.to(self.device).view(batch_size, 1, S, S)   
                    else:      
                        x = x.to(self.device)
                        y = y.to(self.device)
                
                    #self.index = 1
                    if self.incremental_resolution:
                        x, y, self.index = self.incremental_scheduler.step(epoch = epoch, x = x, y = y)
                    sample[0] = x
                    sample[1] = y  
                    self.callbacks.on_batch_start(idx=idx, sample=sample)

                # Decide what to do about logging later when we decide on batch naming conventions
                '''if epoch == 0 and idx == 0 and self.verbose and is_logger:
                    print(f'Training on raw inputs of size {x.shape=}, {y.shape=}')'''
                """if self.dataset_name == 'Burgers' or self.dataset_name == 'Re5000':
                    x, y = sample[0], sample[1]
                else:
                    x, y = sample['x'], sample['y']
                if self.dataset_name == 'Re5000':
                    x = x.to(self.device).view(batch_size, 1, S, S)
                    y = y.to(self.device).view(batch_size, 1, S, S)
                else:      
                    x = x.to(self.device)
                    y = y.to(self.device)
                self.index = 1
                if self.incremental_resolution:
                    x, y, self.index = self.incremental_scheduler.step(epoch = epoch, x = x, y = y)
                sample[0] = x
                sample[1] = y"""  
                
                y = sample[1]

                # load everything from the batch onto self.device if 
                # no callback overrides default load to device
                
                if self.override_load_to_device:
                    self.callbacks.on_load_to_device(sample=sample)
                else:
                    for idx in range(len(sample)):
                        if hasattr(sample[idx], 'to'):
                            sample[idx] = sample[idx].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out = self.model(**sample)
                else:
                    #self.index = 1
                    out = self.model(sample[0], resolution = int(S // self.index), mode = "train").reshape(batch_size, 1, int(S // self.index), int(S // self.index))

                if self.callbacks:
                    self.callbacks.on_before_loss(out=out)

                loss = 0.

                if self.overrides_loss:
                    if isinstance(out, torch.Tensor):
                        loss += self.callbacks.compute_training_loss(out=out.float(), **sample, amp_autocast=self.amp_autocast)
                    elif isinstance(out, dict):
                        loss += self.callbacks.compute_training_loss(**out, **sample, amp_autocast=self.amp_autocast)
                else:
                    if self.amp_autocast:
                        with amp.autocast(enabled=True):
                            if isinstance(out, torch.Tensor):
                                loss = training_loss(out.float(), **sample)
                            elif isinstance(out, dict):
                                loss += training_loss(**out, **sample)
                    else:
                        if isinstance(out, torch.Tensor):
                            loss = training_loss(out.float(), sample[1])
                        elif isinstance(out, dict):
                            loss += training_loss(**out, **sample)
                
                del out

                if regularizer:
                    loss += regularizer.loss
                
                loss.backward()
                
                optimizer.step()
                train_err += loss.item()
        
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

                if self.callbacks:
                    self.callbacks.on_batch_end()

            if isinstance(scheduler[0], torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler[0].step(train_err)
            else:
                if epoch >= 500:
                    scheduler[1].step()
                else:
                    scheduler[0].step()

            epoch_train_time = default_timer() - t1            

            train_err /= len(train_loader)
            if self.dataset_name == 'Re5000':
                train_err/= 400
            else:
                train_err = train_err
            avg_loss  /= (self.n_epochs*400)
            
            if epoch % self.log_test_interval == 0: 

                if self.callbacks:
                    self.callbacks.on_before_val(epoch=epoch, train_err=train_err, time=epoch_train_time, \
                                           avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss)
                    
                if self.incremental and epoch % self.log_test_interval == 0:
                    print("Model is currently using {} number of modes".format(self.model.convs.incremental_n_modes))
                
                _ = self.evaluate(eval_losses, test_loaders)

                if self.callbacks:
                    self.callbacks.on_val_end()
            
            if self.callbacks:
                self.callbacks.on_epoch_end(epoch=epoch, train_err=train_err, avg_loss=avg_loss)

    def evaluate(self, loss_dict, data_loader,
                 log_prefix=''):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        if self.callbacks:
            self.callbacks.on_val_epoch_start(loss_dict = loss_dict, data_loader=data_loader)

        self.model.eval()

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}
        batch_size = 10
        S = 128
        n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                
                if self.callbacks:
                    x = sample[0]
                    y = sample[1]
                    if self.dataset_name == 'Re5000':
                        x = x.to(self.device).view(batch_size, 1, S, S)
                        y = y.to(self.device).view(batch_size, 1, S, S)
                    else:
                        y = y.to(self.device)
                        x = x.to(self.device)
                
                    sample[0] = x
                    sample[1] = y
                    self.callbacks.on_val_batch_start(idx=idx, sample=sample)
                """x = sample[0]
                y = sample[1]
                if self.dataset_name == 'Re5000':
                    x = x.to(self.device).view(batch_size, 1, S, S)
                    y = y.to(self.device).view(batch_size, 1, S, S)
                else:
                    y = y.to(self.device)
                    x = x.to(self.device)
            
                sample[0] = x
                sample[1] = y"""
                y = sample[1]
                n_samples += y.size(0)

                # load everything from the batch onto self.device if 
                # no callback overrides default load to device
                
                if self.override_load_to_device:
                    self.callbacks.on_load_to_device(sample=sample)
                else:
                    for idx in range(len(sample)):
                        if hasattr(sample[idx], 'to'):
                            sample[idx] = sample[idx].to(self.device)
                #self.index = 1
                out = self.model(sample[0], resolution = int(S // self.index), mode = "train").reshape(batch_size, 1, 128, 128)

                if self.callbacks:
                    self.callbacks.on_before_val_loss(out=out)
                
                for loss_name, loss in loss_dict.items():
                    if self.overrides_loss:
                        if isinstance(out, torch.Tensor):
                            val_loss = self.callbacks.compute_training_loss(out.float(), **sample)
                        elif isinstance(out, dict):
                            val_loss = self.callbacks.compute_training_loss(**out, **sample)
                    else:
                        if isinstance(out, torch.Tensor):
                            val_loss = loss(out, sample[1]).item()
                        elif isinstance(out, dict):
                            val_loss = loss(out, **sample).item()

                    errors[f'{log_prefix}_{loss_name}'] += val_loss

                if self.callbacks:
                    self.callbacks.on_val_batch_end()
        
        del y, out

        for key in errors.keys():
            if self.dataset_name == 'Re5000':
                errors[key] /= (n_samples * 1)
            else:
                errors[key] /= n_samples
        
        if self.callbacks:
            self.callbacks.on_val_epoch_end(errors=errors)

        return errors
