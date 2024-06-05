import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets import load_darcy_flow_small, load_darcy_pt
from neuralop.datasets.data_transforms import MGPatchingDataProcessor
from neuralop.training import setup
from neuralop.training.callbacks import BasicLoggerCallback
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.models import FNO
from neuralop.training.callbacks import IncrementalCallback
from neuralop.datasets import data_transforms
from neuralop.training import AdamW


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./darcy_config.yaml", config_name="default", config_folder="../config"
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
wandb_args = None
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
                config.fno2d.hidden_channels,
                config.fno2d.n_modes_width,
                config.fno2d.n_modes_height,
                config.fno2d.factorization,
                config.fno2d.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_args =  dict(
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

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

TRAIN_PATH = '/raid/robert/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = '/raid/robert/piececonst_r421_N1024_smooth1.mat'

train_loader, test_loaders, data_processor = load_darcy_pt(TRAIN_PATH, 800, [200]
                                                ,train_resolution=421, test_resolutions=[421], batch_size=32, test_batch_sizes=[32], positional_encoding=True, encode_input=False, encode_output=True, encoding='channel-wise', channel_dim=1)

# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(in_normalizer=data_processor.in_normalizer,
                                             out_normalizer=data_processor.out_normalizer,
                                             positional_encoding=data_processor.positional_encoding,
                                             padding_fraction=config.patching.padding,
                                             stitching=config.patching.stitching,
                                             levels=config.patching.levels)
data_transform = data_processor.to(device)

if config.incremental.incremental_loss_gap or config.incremental.incremental_grad:
    s = (2,2)
else:
    s = (90,90)
    
model = FNO(
    max_n_modes=(90, 90),
    n_modes=s,
    hidden_channels=128,
    in_channels=3,
    out_channels=1,
)

#get_model(config)
model = model.to(device)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
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
        incremental_grad_max_iter=config.incremental.grad_max), BasicLoggerCallback(wandb_args)]
else:
    callbacks = [BasicLoggerCallback(wandb_args)]

if config.incremental.incremental_res:
    data_transform = data_transforms.IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        positional_encoding=None,
        device=device,
        dataset_sublist=[10,9,8,7,6,5,4,3,2,1],
        dataset_resolution=421,
        dataset_indices=[2, 3],
        epoch_gap=config.incremental.epoch_gap,
        verbose=True,
    ).to(device)

if config.galore:
    galore_params = []
    galore_params.extend(list(model.fno_blocks.convs.parameters()))
    print(galore_params[0].shape, galore_params[1].shape, galore_params[2].shape, galore_params[3].shape)
    galore_params.pop(0)
    id_galore_params = [id(p) for p in galore_params]
    # make parameters without "rank" to another group
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    # then call galore_adamw
    param_groups = [{'params': regular_params}, 
                    {'params': galore_params, 'type': "cp", 'rank': config.rank , 'update_proj_gap': 50, 'scale': config.scale, 'proj_type': "std", 'dim': 5}]
    param_groups1 = [{'type': "cp", 'rank': config.rank , 'update_proj_gap': 50, 'scale': config.scale, 'proj_type': "std", 'dim': 5}]
    optimizer = AdamW(param_groups, lr=5e-3)
else:
    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )
    param_groups1 = [{'rank': 'baseline'}]

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

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    data_processor=data_transform,
    amp_autocast=config.opt.amp_autocast,
    wandb_log=config.wandb.log,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
    callbacks=callbacks)

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
        wandb.log(param_groups1[0])
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
