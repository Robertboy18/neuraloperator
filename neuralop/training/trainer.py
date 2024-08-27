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

import socket
from datetime import datetime
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfileTimeline

categories = {"PARAMETER": 0,
              "OPT": 1,
              "INPUT": 2,
              "TEMP": 3,
              "ACTIVATION": 4,
              "GRADS": 5,
              "AUTOGRAD_DETAIL": 6,
              "None": 7}
GB = 1024**3
device_str = "cuda:0"
rank1 = "0.1"
def trace_handler(prof):
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"trace_burgers_{rank1}"

    # export raw memory timeline
    mem_tl = MemoryProfileTimeline(prof._memory_profile())
    times, sizes = mem_tl._coalesce_timeline(device_str)
    times = np.array(times)
    sizes = np.array(sizes)

    t_min = min(times)
    times -= t_min
    stacked = np.cumsum(sizes, axis=1) / GB
    device = torch.device(device_str)

    msg= f"Memory Timeline for {device_str}\n"
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)
    msg += f"Max CUDA allocated (GB): {max_memory_allocated / GB :.2f}\n"
    msg += f"Max CUDA reserved (GB): {max_memory_reserved / GB :.2f}\n"

    opt_mems = sizes[:,categories["OPT"]]
    max_opt_mem = np.max(opt_mems)
    msg += f"Max optimizer state (GB): {max_opt_mem / GB :.2f}\n"
    
    grad_mems = sizes[:,categories["GRADS"]]
    max_grad_mem = np.max(grad_mems)
    msg += f"Max gradient memory (GB): {max_grad_mem / GB :.2f}\n"

    param_mems = sizes[:, categories["PARAMETER"]]
    max_param_mem = np.max(param_mems)
    msg += f"Max parameter memory (GB): {max_param_mem / GB :.2f}\n"


    with open(f"./mem_stats_{file_prefix}.txt", "w") as f:
        f.write(msg)
    f.close()

    # Construct the memory timeline file.
    #fname = f"./snapshots/{file_prefix}.raw.json.gz"
    #prof.export_memory_timeline(fname, device=device_str)
    
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
                 rank="Baseline"
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
        self.burgers = True
        self.rank = rank
        rank1 = rank
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
              training_loss=None, eval_losses=None, use_fft=False):
        
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
        self.fc = use_fft   
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
                    x = sample[0]
                    y = sample[1]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    #print(x.shape, y.shape)
                    sample = {'x': x, 'y': y}
                    sample = {k:v.to(self.device) for k,v in sample.items() if torch.is_tensor(v)}
                    
                x = sample['x']
                y = sample['y']
                #print(x.shape, y.shape, x.dtype, y.dtype)
                batch, res = x.shape[0], x.shape[1]
                
                loss1 = 0.
                #with record_function("## forward ##"):
                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out  = self.model(**sample)
                else:
                    """
                    for t in range(0, 3):
                        sample['y'] = y[..., t:t+1]
                        if t == 0:
                            xx = sample['x']
                        x = xx
                        sample['x'] = x
                        out = self.model(**sample)
                        #print("Output", out.shape)
                        sample['y'] = sample['y'].view(batch, -1)
                        loss1 += training_loss(out.view(batch, -1), **sample)
                        if t == 0:
                            pred = out
                        else:
                            pred = torch.cat((pred, out), -1)
                        xx = torch.cat((xx[..., 1:], out), dim=-1)
                        """
                    out = self.model(sample['x'])

                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)

                if self.callbacks:
                    self.callbacks.on_before_loss(out=out)
                #print(out.shape, sample['y'].shape)
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
                                if self.burgers:
                                    #print(out.shape, y.shape)
                                    if out.dtype == torch.complex64:
                                        if self.fc == True:
                                            out_fft = torch.fft.fft(out, axis=2)
                                            y_fft = torch.fft.fft(y, axis=2)
                                            loss = training_loss(torch.view_as_real(out_fft), torch.view_as_real(y_fft))
                                        else:
                                            #print(out.shape, sample['y'].shape)
                                            out = out.squeeze()
                                            out = torch.view_as_real(out)
                                            out = out.permute(0, 2, 1)
                                            y = sample['y'].squeeze()
                                            y = torch.view_as_real(y)
                                            y = y.permute(0, 2, 1)
                                            #print(out.shape, y.shape)
                                            loss = training_loss(out, y) 
                                    else:
                                        loss = training_loss(out, y)
                                    #training_loss(out.float().squeeze(), **sample)
                                else:
                                    loss = training_loss(out.float(), **sample)
                        elif isinstance(out, dict):
                            loss += training_loss(**out, **sample)
                if regularizer:
                    loss += regularizer.loss
                #with record_function("## backward ##"):  
                #print(loss)
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
                
                
                if self.burgers:
                    errors = self.evaluate(eval_losses, test_loaders, log_prefix='test')
                else:
                    for loader_name, loader in test_loaders.items():
                        errors = self.evaluate(eval_losses, loader, log_prefix=loader_name)

                if self.callbacks:
                    self.callbacks.on_val_end()
            
            if self.callbacks:
                self.callbacks.on_epoch_end(epoch=epoch, train_err=train_err, avg_loss=avg_loss)
        #prof.export_memory_timeline(f"burgers_rank_{self.rank}.html", device="cuda:0")
            if epoch % 20 == 0:
                torch.save(self.model.state_dict(), f'/raid/robert/em/model_weights/model_{epoch}.pth')
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
                
                if self.callbacks:
                    self.callbacks.on_val_batch_start(idx=idx, sample=sample)

                if self.data_processor is not None:
                    if not self.incremental_resolution:
                        sample = self.data_processor.preprocess(sample)
                    else:
                        sample = self.data_processor.preprocess(sample, mode = "Val")
                else:
                    # load data to device if no preprocessor exists
                    x = sample[0]
                    y = sample[1]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    sample = {'x': x, 'y': y}
                    sample = {k:v.to(self.device) for k,v in sample.items() if torch.is_tensor(v)}


                n_samples += sample['y'].size(0)

                x = sample['x']
                y = sample['y']
                batch, res = x.shape[0], x.shape[1]
                loss1 = 0.
                loss2 = 0.

                out = self.model(sample['x'])

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
                                #print(out.dtype, y.dtype)
                                if self.burgers:
                                    if out.dtype == torch.complex64:
                                        out = out.squeeze()
                                        out = torch.view_as_real(out)
                                        out = out.permute(0, 2, 1)
                                        y = sample['y'].squeeze()
                                        y = torch.view_as_real(y)
                                        y = y.permute(0, 2, 1)
                                        val_loss = loss(out, y) 
                                else:
                                    val_loss = loss(out, **sample)
                        elif isinstance(out, dict):
                            val_loss = loss(out, **sample)
                        #val_loss = val_loss.item()
                    #print(val_loss)
                    errors[f'{log_prefix}_{loss_name}'] += val_loss.item()

                if self.callbacks:
                    self.callbacks.on_val_batch_end()
    
        for key in errors.keys():
            errors[key] /= n_samples
        
        if self.callbacks:
            self.callbacks.on_val_epoch_end(errors=errors, sample=sample, out=out)
        
        del out

        return errors