import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from clip import clip
from src.utils.os_interaction import load_experiment_config
from src.dataset.dataset import refcocog
from src.debug.debug_dataset import dataset_details
from src.ClipVS.model import ClipVS
from src.ClipVS.evaluation import experiment
from src.utils.logger import setup_logger

def main(config_path):
        
    setup_logger(log_note="tr")
    
    config_path = "configuration.yaml"
    config = load_experiment_config(config_path)
    
    verbose = config["verbose"]
    device = config["device"]
    
    CLIP_MODEL, CLIP_IMAGE_PREPROCESS = clip.load("RN50", device=config["device"])
    CLIP_TOKENIZE = clip.tokenize
    
    dataset = { split : refcocog(config, split, CLIP_TOKENIZE, CLIP_IMAGE_PREPROCESS, CLIP_MODEL) for split in config["dt_splits"] }

    if verbose:
        dataset_details(dataset['test'])

    # Initialize the model
    net = ClipVS(config).to(device)

    # Check if a path to pre-trained weights is provided
    if config["nn_reload_weights_from"] is not None:
        weights_path = config["nn_reload_weights_from"]
        try:
            # Load the weights
            state_dict = torch.load(weights_path, map_location=device)
            
            # Load the weights into the model
            net.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {weights_path}")
        except FileNotFoundError:
            print(f"File not found: {weights_path}")
        except RuntimeError as e:
            print(f"Error loading weights: {e}")

    
    experiment(net, dataset, config, CLIP_MODEL, note = 'maintr')

    
    print("Program complete.")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="scripts/configuration.yaml", help="Path to the config file.")
    args = parser.parse_args()
    
    main(args.config)
