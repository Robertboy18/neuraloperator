import sys
import torch
import wandb
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from neuralop import H1Loss, LpLoss, BurgersEqnLoss, ICLoss, WeightedSumLoss, Trainer, get_model
from neuralop.datasets import load_burgers_mat
from neuralop.datasets.data_transforms import MGPatchingDataProcessor
from neuralop.training import setup, BasicLoggerCallback
from neuralop.models import FNO
from neuralop.training.callbacks import IncrementalCallback
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.training import AdamW


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./burgers_config.yaml", config_name="default", config_folder="../config"
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
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        if config.galore:
            wandb_name = "burgers_galore-cp"
        else:
            wandb_name = "burgers_baseline_nogalore"
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

else: 
    wandb_init_args = None
# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

# Load the Burgers dataset
data_path = "/raid/robert/burgers_data_R10.mat"
train_loader, test_loaders = load_burgers_mat(data_path, 800, 200)
output_encoder = None
#model = get_model(config)
#model = model.to(device)

if config.incremental.incremental_loss_gap or config.incremental.incremental_grad:
    s = (2,)
else:
    s = (90,)
    
model = FNO(
    max_n_modes=(90,),
    n_modes=s,
    hidden_channels=128,
    in_channels=2,
    out_channels=1,
).to(device)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

if config.galore:
    print("Using Galore")
    galore_params = []
    galore_params.extend(list(model.fno_blocks.convs.parameters()))
    print(galore_params[0].shape, galore_params[1].shape, galore_params[2].shape, galore_params[3].shape)
    galore_params.pop(0)
    id_galore_params = [id(p) for p in galore_params]
    # make parameters without "rank" to another group
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    # then call galore_adamw
    param_groups = [{'params': regular_params}, 
                    {'params': galore_params, 'type': config.type, 'rank': config.rank , 'update_proj_gap': config.proj_gap, 'scale': config.scale, 'proj_type': "std", 'dim': 5}]
    param_groups1 = [{'type': config.type, 'rank': config.rank , 'update_proj_gap': config.proj_gap, 'scale': config.scale, 'proj_type': "std", 'dim': 5}]
    optimizer = AdamW(param_groups, lr=5e-3)
else:
    print("Not using Galore")
    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )
    param_groups1 = [{'rank': 'baseline'}]

if config.incremental.incremental_loss_gap or config.incremental.incremental_grad:
    callbacks = [
    IncrementalCallback(
        incremental_loss_gap=config.incremental.incremental_loss_gap,
        incremental_grad=config.incremental.incremental_grad,
        incremental_grad_eps=config.incremental.grad_eps,
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
        dataset_sublist=[4, 2, 1],
        dataset_resolution=421,
        dataset_indices=[2, 3],
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
ic_loss = ICLoss()
equation_loss = BurgersEqnLoss(method=config.opt.get('pino_method', None), 
                               visc=0.01, loss=F.mse_loss)

training_loss = config.opt.training_loss
if not isinstance(training_loss, (tuple, list)):
    training_loss = [training_loss]

losses = []
weights = []
for loss in training_loss:
    # Append loss
    if loss == 'l2':
        losses.append(l2loss)
    elif loss == 'h1':
        losses.append(h1loss)
    elif loss == 'equation':
        losses.append(equation_loss)
    elif loss == 'ic':
        losses.append(ic_loss)
    else:
        raise ValueError(f'Training_loss={loss} is not supported.')

    # Append loss weight
    if "loss_weights" in config.opt:
        weights.append(config.opt.loss_weights.get(loss, 1.))
    else:
        weights.append(1.)

train_loss = WeightedSumLoss(losses=losses, weights=weights)
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

# only perform MG patching if config patching levels > 0

callbacks = [
    BasicLoggerCallback(wandb_init_args)
]

tr = False
if tr:
    data_transform = MGPatchingDataProcessor(model=model,
                                           levels=config.patching.levels,
                                           padding_fraction=config.patching.padding,
                                           stitching=config.patching.stitching,
                                           device=device,
                                           in_normalizer=output_encoder,
                                           out_normalizer=output_encoder)

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
    callbacks=callbacks,
    rank = str(config.rank))

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
