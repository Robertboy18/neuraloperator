import torch
import wandb
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop import get_model
from neuralop import Trainer
from neuralop.training import setup
from neuralop.datasets.navier_stokes import load_navier_stokes_hdf5
from neuralop.utils import get_wandb_api_key, count_params
from neuralop import LpLoss, H1Loss
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP


# Read the configuration
config_name = 'default'
pipe = ConfigPipeline([YamlConfig('./incremental.yaml', config_name='default', config_folder='../config'),
                       ArgparseConfig(infer_types=True, config_name=None, config_file=None),
                       YamlConfig(config_folder='../config')
                      ])
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

#Set-up distributed communication, if using
device, is_logger = setup(config)

#Set up WandB logging
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = '_'.join(
            f'{var}' for var in [config_name, config.fno.n_modes, config.fno.hidden_channels, config.fno.projection_channels, config.fno.incremental_n_modes])
    wandb.init(config=config, name=wandb_name,
               project=config.wandb.project, entity=config.wandb.entity)
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

#Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()
    
# Main
ntrain = 900
ntest = 100

modes = 20
width = 128

in_dim = 1
out_dim = 1

batch_size = 50
epochs = 50
learning_rate = 0.0005
scheduler_step = 10
scheduler_gamma = 0.5

loss_k = 0 # H0 Sobolev loss = L2 loss
loss_group = True

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

sub = 1 # spatial subsample
S = 64

T_in = 100 # skip first 100 seconds of each trajectory to let trajectory reach attractor
T = 400 # seconds to extract from each trajectory in data
T_out = T_in + T
step = 1 # Seconds to learn solution operator

data = np.load('/home/robert/data/2D_NS_Re5000.npy?download=1')
data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]

rate = 160000
train_a = data[:ntrain,T_in-1:T_out-1].reshape(rate, S, S)
train_u = data[:ntrain,T_in:T_out].reshape(rate, S, S)

test_a = data[-ntest:,T_in-1:T_out-1].reshape(rate, S, S)
test_u = data[-ntest:,T_in:T_out].reshape(rate, S, S)

assert (S == train_u.shape[2])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loaders = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

model = get_model(config)
model = model.to(device)

#Use distributed data parallel 
if config.distributed.use_distributed:
    model = DDP(model,
                device_ids=[device.index],
                output_device=device.index,
                static_graph=True)

#Log parameter count
if is_logger:
    n_params = count_params(model)

    if config.verbose:
        print(f'\nn_params: {n_params}')
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {'n_params': n_params}
        if config.n_params_baseline is not None:
            to_log['n_params_baseline'] = config.n_params_baseline,
            to_log['compression_ratio'] = config.n_params_baseline/n_params,
            to_log['space_savings'] = 1 - (n_params/config.n_params_baseline)
        wandb.log(to_log)
        wandb.watch(model)

#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config.opt.learning_rate, 
                                weight_decay=config.opt.weight_decay)

if config.opt.scheduler == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.opt.gamma, patience=config.opt.scheduler_patience, mode='min')
elif config.opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.opt.scheduler_T_max)
elif config.opt.scheduler == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config.opt.step_size,
                                                gamma=config.opt.gamma)
elif config.opt.scheduler == 'CyclicLR':
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,step_size_down=config.opt.step_size_down,base_lr=config.opt.base_lr,max_lr=config.opt.max_lr,step_size_up=config.opt.step_size_up,mode=config.opt.mode,last_epoch=-1,cycle_momentum=False)
else:
    raise ValueError(f'Got {config.opt.scheduler=}')

output_encoder = None
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == 'l2':
    train_loss = l2loss
elif config.opt.training_loss == 'h1':
    train_loss = h1loss
else:
    raise ValueError(f'Got training_loss={config.opt.training_loss} but expected one of ["l2", "h1"]')
eval_losses={'h1': h1loss, 'l2': l2loss}

if config.verbose:
    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    print(f'\n### Beginning Training...\n')
    sys.stdout.flush()

trainer = Trainer(model, n_epochs=config.opt.n_epochs,
                  device=device,
                  mg_patching_levels=config.patching.levels,
                  mg_patching_padding=config.patching.padding,
                  mg_patching_stitching=config.patching.stitching,
                  wandb_log=config.wandb.log,
                  log_test_interval=config.wandb.log_test_interval,
                  log_output=config.wandb.log_output,
                  use_distributed=config.distributed.use_distributed,
                  verbose=config.verbose, incremental = config.incremental.incremental_grad.use, 
                  incremental_loss_gap=config.incremental.incremental_loss_gap.use, 
                  incremental_resolution=config.incremental.incremental_resolution.use, dataset_name="Burgers", save_interval=config.checkpoint.interval, model_save_dir=config.checkpoint.directory + config.checkpoint.name)

if config.checkpoint.save:
    # load model from dict
    model_load_epoch = config.checkpoint.last_epoch
    trainer.load_model_checkpoint(model_load_epoch, model, optimizer, config.checkpoint.sub_list_index)
    msg = f'[{model_load_epoch}]'

trainer.train(train_loader, test_loaders, output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

if config.wandb.log and is_logger:
    wandb.finish()
