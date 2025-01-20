import yaml
import os
import csv
import matplotlib

from src.debug.debug_clipvs import print_map

def load_experiment_config(config_path: str) -> dict:
    """
    Load experiment configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file.

    Returns
    -------
    dict
        Loaded configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if config["verbose"]:
        print_map(config, "Experiment configuration:")
    return config

def save_to_experiment_folder(config: dict, image: matplotlib.figure.Figure, name: str) -> None:
    """
    Save an image to the specified experiment folder.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing 'exp_folder_path'.
    image : matplotlib.figure.Figure
        The image/figure to save.
    name : str
        Name of the saved file.
    """
    
    output_folder = config['exp_folder_path']
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, name)
    image.savefig(output_path)
    
def save_performance_to_csv(config : dict, file_path : str, exclude_prefix : str = None) -> None:
    """
    Save performance metrics to a CSV file, appending new rows and filtering keys.

    Parameters
    ----------
    config : dict
        Configuration and performance data to save.
    file_path : str
        Path to the CSV file.
    exclude_prefix : str, optional
        Prefix to exclude from the saved keys, by default None.
        Used to avoid saving baseline settings while saving results for ClipVS, for example.
    """
    max_id = -1

    # Check if the CSV file already exists
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            if 'id' in reader.fieldnames:
                # Extract the current max id from existing rows
                for row in reader:
                    max_id = max(max_id, int(row['id']))
            else:
                raise ValueError("Existing file does not have an 'id' column")
    
    # Increment the id for the new row
    exp_id = max_id + 1
    config["exp_id"] = exp_id
    
    # Filter out keys that start with "gs_" or the dynamically specified prefix
    filtered_config = {
        key: value for key, value in config.items()
        if not key.startswith("gs_") and (exclude_prefix is None or not key.startswith(exclude_prefix))
    }
    
    # Convert config dictionary keys to strings
    fieldnames = ['id'] + list(map(str, filtered_config.keys()))
    file_exists = os.path.exists(file_path)
    
    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Create a row with filtered config data
        row = {key: filtered_config.get(key, None) for key in fieldnames if key != 'id'}
        row['id'] = config["exp_id"]
        
        # Write the row to the CSV file
        writer.writerow(row)
    
def save_performance_to_yaml(config: dict) -> None:
    """
    Save the entire configuration to a YAML file in the experiment folder.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing 'exp_folder_path'.
    """
    yaml_file_path = os.path.join(config["exp_folder_path"], 'configuration.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file)
    print(f"Configuration saved to: {yaml_file_path}")