import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from clip import clip
from src.utils.os_interaction import load_experiment_config
from src.utils.grid_search import grid_search
from src.dataset.dataset import refcocog
from src.debug.debug_dataset import dataset_details
from src.utils.logger import setup_logger

def main(config_path):
    
    setup_logger(log_note="gs")
    
    config_path = "configuration.yaml"
    config = load_experiment_config(config_path)
    
    verbose = config["verbose"]
    
    CLIP_MODEL, CLIP_IMAGE_PREPROCESS = clip.load("RN50", device=config["device"])
    CLIP_TOKENIZE = clip.tokenize
    
    dataset = { split : refcocog(config, split, CLIP_TOKENIZE, CLIP_IMAGE_PREPROCESS, CLIP_MODEL) for split in config["dt_splits"] }

    if verbose:
        dataset_details(dataset['test'])
    
    grid_search(config, dataset,
                    csv_path = f"{config['output_to']}/clipvs_grid_search_L18_sl{config['dt_samples_limit']}.csv", model = "ClipVS")
        
    print("Program complete.")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="scripts/configuration.yaml", help="Path to the config file.")
    args = parser.parse_args()
    
    main(args.config)


