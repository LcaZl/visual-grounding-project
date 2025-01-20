import torch
import cv2
import numpy as np
from clip import clip
from torch.utils.data import DataLoader
import pandas as pd
import itertools
import torchvision

from src.ClipVS.model import ClipVS
from src.baseline.evaluation import collate_fn_bs, evaluate
from src.baseline.model import baselineCLIP
from src.ClipVS.batch_preparation import collate_fn
from src.utils.os_interaction import save_performance_to_csv
from src.ClipVS.evaluation import training_step, test_step
from src.ClipVS.optimizer import get_optimizer
from src.ClipVS.loss import custom_loss

def grid_search(config: dict, dataset: dict, csv_path: str, model: str) -> None:
    """
    Perform a grid search over specified hyperparameters for a given model
    and dataset configuration. Results are logged to a CSV file.

    This function supports grid search for both ClipVS and Baseline models.
    Hyperparameters to be grid-searched are specified in the configuration
    file with a 'gs_' prefix. For example:

        - Base parameter: 'nn_batch_size: 32'
        - Grid-search parameter: 'gs_nn_batch_size: [16, 32, 64]'

    Parameters with the 'gs_' prefix are extracted and all combinations of 
    these values are tested.

    For Baseline, the prefixes used are 'gs_bs_' for model parameters and 'gs_dt_' for dataset parameters.
    For ClipVS the prefixes used are 'gs_nn_' for model parameters and always 'gs_dt_' for dataset parameters.
    ClipVS model is evaluated on the test set, and this performance is stored.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and hyperparameter settings,
        including the grid search parameters (prefixed with 'gs_').
    dataset : dict
        Dataset to be used.
    csv_path : str
        Path to the CSV file for saving results.
    model : str, optional
        Model name to evaluate, either 'ClipVS' or 'Baseline'.

    """
    clip_model, clip_image_preprocessing = clip.load('RN50', device=config['device'])
    clip_tokenize = clip.tokenize

    # Prepare data loaders based on the model type
    if model == 'Baseline':
        prefix_to_exlude = "nn_"
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True).eval().to(config["device"])
        frcnn_model = torchvision.models.detection.retinanet_resnet50_fpn(weights='DEFAULT').eval().to(config["device"])
        ssd_model = torchvision.models.detection.ssd300_vgg16(weights='DEFAULT').eval().to(config["device"])
    else:
        prefix_to_exlude = "bs_"

    # Load existing results if available
    try:
        df_existing_results = pd.read_csv(csv_path)
    except FileNotFoundError:
        df_existing_results = pd.DataFrame()  # Create empty DataFrame if no results exist

    # Determine prefixes for hyperparameters based on model type
    prefixes = ['gs_dt_', 'gs_bs_'] if model == 'Baseline' else ['gs_nn_', 'gs_dt_']

    # Extract grid search parameters from config
    fgs_params = {key: config[key] for key in config if any(key.startswith(prefix) for prefix in prefixes)}

    # Generate all parameter combinations
    all_combinations = list(itertools.product(*fgs_params.values()))

    print(f'\nStarting Grid Search over {len(all_combinations)} combinations for {model}\n')

    # Track existing combinations to avoid redundant evaluations
    existing_combinations = {}
    if len(df_existing_results) > 0:
        existing_combinations = {
            row['comb_id']: row['avg_iou_accuracy'] for _, row in df_existing_results.iterrows()
        }

    # Iterate through parameter combinations
    for i, comb in enumerate(all_combinations):
        comb_id = str('_'.join(map(str, comb))).lower()
        
        if comb_id in existing_combinations:
            print(f'Skipping combination {i+1}/{len(all_combinations)} - {comb_id} (Already exists) (IoU Accuracy: {existing_combinations[comb_id]})')
            continue

        print(f'\nCombination {i+1}/{len(all_combinations)} - {comb} - {comb_id}')

        # Update config with current combination
        current_config = config.copy()
        current_config['comb_id'] = comb_id
        
        for j, key in enumerate(fgs_params):
            stripped_key = key.replace('gs_', '')  # Remove 'gs_' prefix for target keys
            current_config[stripped_key] = comb[j]

        # Update the kind of samples requested
        for split, dt in dataset.items():
            dt.apply_templates = current_config["dt_apply_template"]
            dt.enh_sents = current_config["dt_extra_similar_sents"]
            dt.reload_similarity_matrix(current_config)

        # Evaluate the model
        
        if model == 'ClipVS':
            
            # For ClipVS, use the batch size specified in the config
        
            train_loader = DataLoader(
                dataset['train'],
                batch_size=config['nn_batch_size'],
                collate_fn=lambda batch: collate_fn(batch, config['device']), shuffle=False
            )
            test_loader = DataLoader(
                dataset['test'],
                batch_size=config['nn_batch_size'],
                collate_fn=lambda batch: collate_fn(batch, config['device']), shuffle=False
            )
            
            # Initialize ClipVS model and perform training and evaluation
            model_instance = ClipVS(current_config).to(current_config['device'])
            
            if config["nn_reload_weights_from"] is not None:
                
                weights_path = config["nn_reload_weights_from"]
                
                try:
                    
                    # Load the weights
                    state_dict = torch.load(weights_path, map_location=current_config['device'])
                    
                    # Load the weights into the model
                    model_instance.load_state_dict(state_dict)
                    
                    print(f"Successfully loaded weights from {weights_path}")
                except FileNotFoundError:
                    print(f"File not found: {weights_path}")
                except RuntimeError as e:
                    print(f"Error loading weights: {e}")
                    
            optimizer, scheduler = get_optimizer(current_config, model_instance)

            train_losses = []
            for e in range(current_config["nn_epochs"]):
                print(f"\nEpoch {e + 1}/{current_config['nn_epochs']}")
                print(f"Current learning rates:\n{scheduler.get_last_lr()}\n")
        
                losses, _ = training_step(model_instance, train_loader, optimizer, custom_loss, current_config)
                train_losses.append(np.mean(losses).item())
                
                if current_config["nn_use_scheduler"]:
                    scheduler.step()

            test_losses, test_losses_components = test_step(model_instance, test_loader, custom_loss, current_config, mode='Testing')

            avg_iou_accuracy = np.mean([el['IoU Acc.'] for el in test_losses_components])
            current_config['test_losses'] = test_losses
            current_config['train_losses'] = train_losses
            
        else:
            
            # For Baseline, batch size is always 1
            
            config['nn_batch_size'] = 1
            test_loader = DataLoader(
                dataset['test'],
                batch_size=1,
                collate_fn=lambda batch: collate_fn_bs(batch), shuffle=False
            )
        
            # Initialize and evaluate Baseline model
            model_instance = baselineCLIP(current_config, clip_model, clip_image_preprocessing, clip_tokenize, 
                                          yolo_model=yolo_model, 
                                          frcnn_model=frcnn_model, 
                                          ssd_model=ssd_model, 
                                          verbose=False)
            
            avg_iou_accuracy = evaluate(model_instance, test_loader)
            
        del model_instance, test_loader
            
        # Save the results
        current_config['avg_iou_accuracy'] = round(float(avg_iou_accuracy), 4)
        save_performance_to_csv(current_config, csv_path, exclude_prefix=prefix_to_exlude)

        print(f'-> Avg. Accuracy: {current_config["avg_iou_accuracy"]}')