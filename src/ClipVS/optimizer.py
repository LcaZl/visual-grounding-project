import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR
from torch.optim import Optimizer
from torch.nn import Module

def get_optimizer(config: dict, model: Module) -> tuple[Optimizer, ReduceLROnPlateau]:
    """
    Create an AdamW optimizer and a PolynomialLR scheduler for training a model.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing learning rates for different components of the model:
        - 'nn_opt_learning_rate': Learning rate for the optimizer.
    model : Module
        ClipVS model.
        
    Returns
    -------
    tuple[Optimizer, ReduceLROnPlateau]
        - optimizer : Optimizer
            AdamW optimizer for the model parameters.
        - scheduler : PolynomialLR
            Scheduler to adjust the learning rate.
    """

    trainable_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("clip_model.")
    ]
    
    optimizer = torch.optim.AdamW(
        params=trainable_params,
        lr=config["nn_opt_learning_rate"],
        weight_decay=config["nn_wd"],
    )
        
    scheduler = PolynomialLR(
        optimizer,
        total_iters=config["nn_epochs"],
        power = config["nn_scheduler_power"]
    )

    return optimizer, scheduler
