import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets.navier_stokes import load_ns_high
from neuralop.datasets.data_transforms import MGPatchingDataProcessor
from neuralop.training import setup, BasicLoggerCallback
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.datasets.data_transforms import MGPatchingDataProcessor
from neuralop.training import setup, BasicLoggerCallback
from neuralop.models import FNO
from neuralop.training.callbacks import IncrementalCallback
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.datasets import data_transforms
import os

os.environ["WANDB_DIR"] = os.path.abspath("/pscratch/sd/r/rgeorge/robert_wandb/")

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./default_config.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
wandb_init_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.fno2d.n_layers,
                config.fno2d.n_modes_width,
                config.fno2d.n_modes_height,
                config.fno2d.hidden_channels,
                config.fno2d.factorization,
                config.fno2d.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger
torch.manual_seed(config.seed)

# Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

data_path = "/pscratch/sd/r/rgeorge/data/vorticty3.pt"
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_ns_high(data_path, ntrain=8, ntest=2, subsampling_rate=1, batch_size=50, T = 5000, in_dim = 1, out_dim = 1, ntimeindex = 1, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(in_normalizer=data_processor.in_normalizer,
                                             out_normalizer=data_processor.out_normalizer,
                                             positional_encoding=data_processor.positional_encoding,
                                             padding_fraction=config.patching.padding,
                                             stitching=config.patching.stitching,
                                             levels=config.patching.levels)

if data_processor is not None:
    data_processor = data_processor.to(device)
#model = get_model(config)
#model = model.to(device)
modes = config.mode
s1 = tuple([modes, modes])
if config.incremental.incremental_loss_gap or config.incremental.incremental_grad:
    s = (2,2)
else:
    s = s1
    
model = FNO(
    max_n_modes=s1,
    n_modes=s,
    hidden_channels=128,
    in_channels=1,
    out_channels=1,
).to(device)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.incremental.incremental_loss_gap or config.incremental.incremental_grad:
    callbacks = [
    IncrementalCallback(
        incremental_loss_gap=config.incremental.incremental_loss_gap,
        incremental_grad=config.incremental.incremental_grad,
        incremental_grad_eps=config.incremental.grad_eps,
        incremental_loss_eps = config.incremental.loss_eps,
        incremental_buffer=5,
        incremental_max_iter=config.incremental.max_iter,
        incremental_grad_max_iter=config.incremental.grad_max), BasicLoggerCallback(wandb_init_args)]
else:
    callbacks = [BasicLoggerCallback(wandb_init_args)]
    
data_transform = None
if config.incremental.incremental_res:
    data_transform = data_transforms.IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        positional_encoding=None,
        device=device,
        dataset_sublist=config.incremental.sub_list1,
        dataset_resolution=256,
        dataset_indices=[2,3],
        epoch_gap=config.incremental.epoch_gap,
        verbose=True,
    ).to(device)
    

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()
s
trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_transform,
    device=device,
    amp_autocast=config.opt.amp_autocast,
    callbacks=callbacks,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log = config.wandb.log
)

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log)
        wandb.watch(model)


trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)


if config.wandb.log and is_logger:
    wandb.finish()
