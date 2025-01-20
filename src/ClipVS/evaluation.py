import torch
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_

from src.utils.os_interaction import save_performance_to_yaml, save_to_experiment_folder
from src.ClipVS.batch_preparation import collate_fn
from src.ClipVS.loss import custom_loss
from src.ClipVS.optimizer import get_optimizer
from src.ClipVS.postprocessing import postprocessing

def experiment(
    net: torch.nn.Module,
    dataset: dict[str, object],
    config: dict[str, object],
    note: str = ""
) -> None:
    """
    Run a complete training, validation and testing pipeline for the given model.

    Parameters
    ----------
    net : torch.nn.Module
        ClipVS model to train and evaluate.
    dataset : dict
        Dictionary containing 'train', 'val' and 'test' datasets.
    config : dict
        Configuration dictionary containing hyperparameters, paths and other settings.
    note : str, optional
        A note to append to the experiment folder name for identification, by default "".

    Returns
    -------
        Results are saved to the experiment folder specified in 'config'.
    """

    def get_next_experiment_id(basepath: str) -> int:
        """
        Get the next available experiment id based on the existing folders in the basepath.

        Parameters
        ----------
        basepath : str
            Path to the folder containing experiment subdirectories.

        Returns
        -------
        int
            The next experiment ID.
        """
        ids = [int(f.split("_")[1]) for f in os.listdir(basepath) if f.startswith("experiment_")]
        return max(ids, default=-1) + 1

    # Setup experiment folder
    basepath = f"{config['output_to']}/ClipVS_experiments/"
    new_id = get_next_experiment_id(basepath)
    new_experiment_path = os.path.join(basepath, f"experiment_{new_id}_{note}")
    os.makedirs(new_experiment_path, exist_ok=True)
    print(f"Created new experiment folder: {new_experiment_path}")
    config["exp_folder_path"] = new_experiment_path
    
    # create data loaders
    batch_size = config["nn_batch_size"]
    device = config["device"]
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, collate_fn= lambda batch : collate_fn(batch, device), shuffle=False)
    val_loader = DataLoader(dataset['val'], batch_size=batch_size, collate_fn= lambda batch : collate_fn(batch, device), shuffle=False)
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, collate_fn= lambda batch : collate_fn(batch, device), shuffle=False)

    optimizer, scheduler = get_optimizer(config, net)
    loss_function = custom_loss
    epochs = config["nn_epochs"]

    # Initialize lists to store losses
    all_train_losses = []
    all_val_losses = []
    all_train_losses_components = []
    all_val_losses_components = []
    
    for e in range(epochs):

        print(f"\nEpoch {e + 1}/{epochs}")
        print(f"Current learning rates:\n{scheduler.get_last_lr()}\n")
        
        # Training and validation steps
        train_losses, train_losses_components = training_step(net, train_loader, optimizer, loss_function, config)
        val_losses, val_losses_components = test_step(net, val_loader, loss_function, config, mode='Validation')

        # Store metrics
        all_train_losses_components.append(train_losses_components)
        all_val_losses_components.append(val_losses_components)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        report = epoch_report(e, epochs, config, mode='train',
                        train_losses=train_losses, train_losses_components=train_losses_components, val_losses=val_losses, val_losses_components=val_losses_components)
        save_to_experiment_folder(config, report, f"train_epoch_{e+1}.png")
        plt.close()
        
        # Update learning rate scheduler with validation loss
        if config["nn_use_scheduler"]:
            scheduler.step()
    
    # Testing step after training
    test_losses, test_losses_components = test_step(net, test_loader, loss_function, config, mode='Testing')
    report = epoch_report(e, epochs, config, mode='test',
                test_losses=test_losses, test_losses_components=test_losses_components)
    save_to_experiment_folder(config, report, f"test.png")
    plt.close()

    # Store test results
    config["train_losses"] = all_train_losses
    config["val_losses"] = all_val_losses
    config["test_losses"] = test_losses
    config["test_losses_components"] = test_losses_components
    config["val_losses_components"] = all_val_losses_components
    config["train_losses_components"] = all_train_losses_components

    save_performance_to_yaml(config)
    
    # Save model weights if required
    if config["nn_store_model_weights"]:
        model_path = os.path.join(config["exp_folder_path"], "model_weights.pth")
        torch.save(net.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")
        
def test_step(
    net: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_function: callable,
    config: dict,
    mode: str = 'Validation'
) -> tuple[list[float], list[dict]]:
    """
    Perform a validation or testing step for ClipVS.

    Parameters
    ----------
    net : torch.nn.Module
        ClipVS model to evaluate.
    data_loader : torch.utils.data.DataLoader
        DataLoader providing the validation or test data.
    loss_function : callable
        Function to compute the loss and its components.
    config : dict
        Configuration dictionary with additional settings.
    mode : str, optional
        Describes the mode ('Validation' or 'Testing'), by default 'Validation'.

    Returns
    -------
    tuple[list[float], list[dict]]
        - batch_losses: List of losses for each batch.
        - batch_losses_components: List of dictionaries containing each component of the loss for each batch.
    """
    
    batch_losses = []
    batch_losses_components = []
    
    net.eval()
    with tqdm(total=len(data_loader), desc=f"{mode}", unit="batch") as pbar:  # Progress bar for validation or testing
        with torch.no_grad():
            for i, batch in enumerate(data_loader):

                output = net(batch)
                postprocessing(batch, output, config)

                loss, components = loss_function(batch, output, config)

                batch_losses.append(loss.item())
                batch_losses_components.append({key : comp for key, comp in components.items()})
                
                #debug_plotting(batch, output, "")
                #plot_prediction(batch["imgs_raw"][-1], output["pred_bboxes"][-1].detach().cpu().numpy(), batch["gt_bboxes"][-1].cpu().numpy(), output["attention_masks_resized"][-1].detach().cpu().numpy().transpose(1,2,0), output["iou_accuracies"][-1].item())

                average_dict = {k: round(sum(d[k] for d in batch_losses_components) / len(batch_losses_components),4) for k in batch_losses_components[0]}
                pbar.set_postfix({"Avg. Loss": f"{np.mean(batch_losses):.4f}", "Loss Components & IoU":average_dict })
                pbar.update()
    
    return batch_losses, batch_losses_components

def training_step(
    net: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: callable,
    config: dict,
) -> tuple[list[float], list[dict]]:
    """
    Perform a single training step for ClipVS.

    Parameters
    ----------
    net : torch.nn.Module
        ClipVS model to train.
    data_loader : torch.utils.data.DataLoader
        DataLoader providing the training data.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating model parameters.
    loss_function : callable
        Function to compute the loss and its components.
    config : dict
        Configuration dictionary with additional settings.

    Returns
    -------
    tuple[list[float], list[dict]]
        - batch_losses: List of losses for each batch.
        - batch_losses_components: List of dictionaries containing each component of the loss for each batch.
    """  
    batch_losses = []
    batch_losses_components = []
    net.train()

    with tqdm(total=len(data_loader), desc="Training", unit="batch") as pbar:  # Progress bar for training
        for i, batch in enumerate(data_loader):


            output = net(batch)
            postprocessing(batch, output, config)

            loss, components = loss_function(batch, output, config)
            loss.backward()
            
            if config["nn_grad_clipping"]:
                clip_grad_norm_(net.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(loss.item())
            batch_losses_components.append({key : comp for key, comp in components.items()})

            average_dict = {k: round(sum(d[k] for d in batch_losses_components) / len(batch_losses_components),4) for k in batch_losses_components[0]}
            
            pbar.set_postfix({"Avg. Loss": f"{np.mean(batch_losses):.4f}", "Loss Components & IoU":average_dict })
            pbar.update()
                    
    return batch_losses, batch_losses_components


def epoch_report(
    epoch: int,
    epochs: int,
    config: dict,
    mode: str = 'train',
    train_losses: list[float] = None,
    train_losses_components: list[dict[str, float]] = None,
    val_losses: list[float] = None,
    val_losses_components: list[dict[str, float]] = None,
    test_losses: list[float] = None,
    test_losses_components: list[dict[str, float]] = None
) -> plt.Figure:
    """
    Generate a report for the training, validation, or testing phase of the model.
    
    Parameters
    ----------
    epoch : int
        Current epoch number.
    epochs : int
        Total number of epochs.
    config : dict
        Configuration dictionary for additional settings or metadata.
    mode : str, optional
        The phase for which the report is generated ('train' or 'test'), by default 'train'.
    train_losses : list[float], optional
        List of training losses for each batch, required if mode is 'train'.
    train_losses_components : list[dict], optional
        List of dictionaries containing detailed training loss components per batch.
    val_losses : list[float], optional
        List of validation losses for each batch, required if mode is 'train'.
    val_losses_components : list[dict], optional
        List of dictionaries containing detailed validation loss components per batch.
    test_losses : list[float], optional
        List of test losses for each batch, required if mode is 'test'.
    test_losses_components : list[dict], optional
        List of dictionaries containing detailed test loss components per batch.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the report with a table and plot of losses.
    """

    def create_table(ax, losses_components: list[dict[str, float]], losses: list[float]):
        """
        Helper function to create a table of batch losses and components.
        """
       
        ax.axis('off')
        ax.axis('tight')

        component_keys = list(losses_components[0].keys())
        header = ["Batch", "Loss"] + component_keys
        table_data = []

        # Populate the table with batch losses and components
        for i, (losses_dict, loss) in enumerate(zip(losses_components, losses)):
            row = [i + 1, round(loss, 4)] + [round(losses_dict[key], 4) for key in component_keys]
            table_data.append(row)

        # Calculate mean loss and components
        mean_loss = np.mean(losses)
        mean_components = {key: np.mean([losses_dict[key] for losses_dict in losses_components]) for key in component_keys}
        
        # Add the mean row (for the Loss and other components) at the end
        mean_row = ["Mean", round(mean_loss, 4)] + [round(mean_components[key], 4) for key in component_keys]
        table_data.append(mean_row)

        # Create the table for loss and loss components
        table = ax.table(cellText=table_data, colLabels=header, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(header))))
        table.scale(1.2, 1.2)

        # Make the header and mean row bold
        mean_row_idx = len(table_data)  # The index of the mean row is the last row in the table
        for col_idx in range(len(header)):
            table[0, col_idx].set_text_props(fontweight='bold')  # Header
            table[mean_row_idx, col_idx].set_text_props(fontweight='bold')  # Mean row


    
    def plot_losses(ax, losses: list[float], label: str, color: str, linestyle: str, marker: str):
        """
        Helper function to plot losses.
        """
        x_values = list(range(1, len(losses) + 1))
        ax.plot(x_values, losses, label=label, color=color, linestyle=linestyle, 
                marker=marker, markersize=6, linewidth=2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 20), gridspec_kw={'width_ratios': [1, 2]})

    if mode == 'train':
        
        # Training mode: table and plots for training and validation losses
        create_table(ax1, train_losses_components, train_losses)

        # Plot training and validation losses
        plot_losses(ax2, train_losses, "Training losses", '#1f77b4', '-', 'o')
        plot_losses(ax2, val_losses, "Validation losses", '#ff7f0e', '--', 's')

        # Set the title with average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        ax2.set_title(f"Epoch {epoch + 1}/{epochs} - Avg. Train Loss: {avg_train_loss:.4f}, "
                      f"Avg. Val Loss: {avg_val_loss:.4f}", fontsize=14, fontweight='bold', color='#333')

    elif mode == 'test':
        
        # Test mode: table and plot for test losses
        create_table(ax1, test_losses_components, test_losses)

        # Plot test losses
        plot_losses(ax2, test_losses, "Test losses", '#ff7f0e', '-', 'o')

        # Set the title with average test loss
        avg_test_loss = np.mean(test_losses)
        ax2.set_title(f"Test Losses - Mean: {avg_test_loss:.4f}", fontsize=14, fontweight='bold', color='#333')

    ax2.legend(loc='best', fontsize=11, frameon=True, shadow=True, borderpad=1)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("Batch", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Loss", fontsize=12, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    plt.tight_layout()
    
    return fig