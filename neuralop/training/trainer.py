import torch
from torch.cuda import amp
from timeit import default_timer
import pathlib

from .callbacks import PipelineCallback
import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
import numpy as np
# (c) Meta Platforms, Inc. and affiliates. 
import logging
import socket
from datetime import datetime, timedelta

import torch

from torch.autograd.profiler import record_function

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
class Trainer:
    def __init__(self, *, 
                 model, 
                 n_epochs, 
                 wandb_log=True, 
                 device=None, 
                 amp_autocast=False,
                 data_processor=None,
                 callbacks = None,
                 log_test_interval=1, 
                 log_output=False, 
                 use_distributed=False, 
                 verbose=False, 
                 nstime=False,
                 ns2dtime=False,
                 ):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        amp_autocast : bool, default is False
        data_processor : class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
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
                 verbose=verbose)

        self.model = model
        self.n_epochs = n_epochs

        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.amp_autocast = amp_autocast
        self.data_processor = data_processor
        self.incremental_resolution = False
        self.nstime = nstime
        self.ns2dtime = ns2dtime
        # If the data_processor is an IncrementalDataProcessor, then we need to do curriculum learning - Increase the resolution of the samples incrementally
        if type(self.data_processor).__name__ == "IncrementalDataProcessor":
            self.incremental_resolution = True
        
        if self.callbacks:
            self.callbacks.on_init_end(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose)
        
    def train(self, train_loader, test_loaders,
            optimizer, scheduler, regularizer,
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
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
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

        errors = None

        """
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
        ) as prof:
        """
        for epoch in range(self.n_epochs):
            #prof.step()
            if self.callbacks:
                self.callbacks.on_epoch_start(epoch=epoch)

            avg_loss = 0
            avg_lasso_loss = 0
            self.model.train()
            t1 = default_timer()
            train_err = 0.0

            for idx, sample in enumerate(train_loader):

                if self.callbacks:
                    self.callbacks.on_batch_start(idx=idx, sample=sample)

                if regularizer:
                    regularizer.reset()

                if self.data_processor is not None:
                    if not self.incremental_resolution:
                        sample = self.data_processor.preprocess(sample)
                    else:
                        sample = self.data_processor.preprocess(sample, epoch=epoch, mode = "Train")
                else:
                    # load data to device if no preprocessor exists
                    sample = {k:v.to(self.device) for k,v in sample.items() if torch.is_tensor(v)}
                    
                x = sample['x']
                y = sample['y']
                batch, res = x.shape[0], x.shape[1]
                if self.nstime:
                    x = sample['x']
                    grid = self.get_grid(x.shape, x.device)
                    sample['x'] = torch.cat((x, grid), dim=-1).permute(0, 4, 1, 2, 3)
                
                loss1 = 0.
                #with record_function("## forward ##"):
                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out  = self.model(**sample)
                else:
                    if self.nstime:
                        out  = self.model(**sample).view(batch, res, res, 10)
                    elif self.ns2dtime:
                        for t in range(0, 10):
                            sample['y'] = y[..., t:t+1]
                            if t == 0:
                                xx = sample['x']
                            x = xx
                            sample['x'] = x.permute(0, 3, 1, 2)
                            out = self.model(**sample).permute(0, 2, 3, 1)
                            sample['y'] = sample['y'].view(batch, -1)
                            loss1 += training_loss(out.view(batch, -1), **sample)
                            if t == 0:
                                pred = out
                            else:
                                pred = torch.cat((pred, out), -1)
                            xx = torch.cat((xx[..., 1:], out), dim=-1)
                    else:
                        out = self.model(**sample)

                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)

                if self.callbacks:
                    self.callbacks.on_before_loss(out=out)
                
                loss = 0
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
                            if self.nstime:
                                sample['y'] = sample['y'].view(batch, -1)
                                loss = training_loss(out.view(batch, -1), **sample)
                            elif self.ns2dtime:
                                loss += loss1
                            else:
                                loss = training_loss(out.float(), **sample)
                        elif isinstance(out, dict):
                            loss += training_loss(**out, **sample)
                if regularizer:
                    loss += regularizer.loss
                #with record_function("## backward ##"):  
                loss.backward()
                del out
                #with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_err += loss.item()
        
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

                if self.callbacks:
                    self.callbacks.on_batch_end()

            #with record_function("## scheduler ##"):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1            

            train_err /= len(train_loader)
            avg_loss  /= self.n_epochs
            
            if epoch % self.log_test_interval == 0: 

                if self.callbacks:
                    self.callbacks.on_before_val(epoch=epoch, train_err=train_err, time=epoch_train_time, \
                                        avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss)
                

                for loader_name, loader in test_loaders.items():
                    errors = self.evaluate(eval_losses, loader, log_prefix=loader_name)

                if self.callbacks:
                    self.callbacks.on_val_end()
            
            if self.callbacks:
                self.callbacks.on_epoch_end(epoch=epoch, train_err=train_err, avg_loss=avg_loss)
        #rof.export_memory_timeline(f"trial2.html", device="cuda:0")
        return errors

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
            self.callbacks.on_val_epoch_start(log_prefix=log_prefix, loss_dict = loss_dict, data_loader=data_loader)

        self.model.eval()

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):

                n_samples += sample['y'].size(0)
                if self.callbacks:
                    self.callbacks.on_val_batch_start(idx=idx, sample=sample)

                if self.data_processor is not None:
                    if not self.incremental_resolution:
                        sample = self.data_processor.preprocess(sample)
                    else:
                        sample = self.data_processor.preprocess(sample, mode = "Val")
                else:
                    # load data to device if no preprocessor exists
                    sample = {k:v.to(self.device) for k,v in sample.items() if torch.is_tensor(v)}
                x = sample['x']
                y = sample['y']
                batch, res = x.shape[0], x.shape[1]
                loss1 = 0.
                loss2 = 0.
                if self.nstime:
                    x = sample['x']
                    grid = self.get_grid(x.shape, x.device)
                    x = torch.cat((x, grid), dim=-1)
                    sample['x'] = x.permute(0, 4, 1, 2, 3)
            
                    out = self.model(**sample).reshape(batch, res, res, 10)
                elif self.ns2dtime:
                    for t in range(0, 10):
                        sample['y'] = y[..., t:t+1]
                        if t == 0:
                            xx = sample['x']
                        x = xx
                        sample['x'] = x.permute(0, 3, 1, 2)
                        out = self.model(**sample).permute(0, 2, 3, 1)
                        sample['y'] = sample['y'].view(batch, -1)
                        loss1 += loss_dict["h1"](out.view(batch, -1), **sample)
                        loss2 += loss_dict["l2"](out.view(batch, -1), **sample)
                        if t == 0:
                            pred = out
                        else:
                            pred = torch.cat((pred, out), -1)
                        xx = torch.cat((xx[..., 1:], out), dim=-1)
                else:
                    out = self.model(**sample)

                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)

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
                            if self.nstime:
                                sample['y'] = sample['y'].view(batch, -1)
                                val_loss = loss(out.view(batch, -1), **sample)
                            elif self.ns2dtime:
                                if loss_name == "h1":
                                    val_loss = loss1
                                else:
                                    val_loss = loss2
                            else:
                                val_loss = loss(out, **sample)
                        elif isinstance(out, dict):
                            val_loss = loss(out, **sample)
                        if val_loss.shape == ():
                            val_loss = val_loss.item()

                    errors[f'{log_prefix}_{loss_name}'] += val_loss

                if self.callbacks:
                    self.callbacks.on_val_batch_end()
    
        for key in errors.keys():
            errors[key] /= n_samples
        
        if self.callbacks:
            self.callbacks.on_val_epoch_end(errors=errors, sample=sample, out=out)
        
        del out

        return errors

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)