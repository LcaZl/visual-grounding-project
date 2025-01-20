import torch
import cv2
import numpy as np

from src.utils.os_interaction import load_experiment_config, save_to_experiment_folder
from src.ClipVS.model import ClipVS
from torch.utils.data import DataLoader
from src.ClipVS.batch_preparation import collate_fn
from src.ClipVS.postprocessing import postprocessing
from src.ClipVS.loss import custom_loss
from src.ClipVS.evaluation import epoch_report
from src.debug.debug_clipvs import debug_batch_info, debug_tensors, debug_plotting

def create_vocab(elements_list: list[object]) -> tuple[dict[object, int], dict[int, object]]:
    """
    Creates a vocabulary mapping for a given list of elements, assigning to each 
    unique element an unique integer id. Also generates a reverse mapping from 
    ids back to elements.

    Parameters
    ----------
    elements_list : list
        A list of elements to map.

    Returns
    -------
    tuple
        vocab : dict
            Maps elements to integer IDs.
        reverse : dict
            Maps integer IDs back to elements.
    """
    vocab = {} 
    for el in elements_list:
        if el not in vocab:
            vocab[el] = len(vocab) 
            
    reverse = {id: el for el, id in vocab.items()}
    return vocab, reverse


def get_clip_mean_embeddings(sents_tokenized: torch.Tensor, clip_model) -> torch.Tensor:
    """
    Computes the mean normalized text embeddings of a list of tokenized 
    sentences using CLIP model.

    Parameters
    ----------
    sents_tokenized : torch.Tensor
        A tensor of tokenized sentences of shape [batch_size, 77].
    clip_model : CLIP model used to encode the text.

    Returns
    -------
    torch.Tensor
        A normalized tensor of shape [1, 1024] representing the mean embedding of all sentences.
    """
    
    with torch.no_grad():
        texts_z = clip_model.encode_text(sents_tokenized).float()
    texts_z /= texts_z.norm(dim=-1, keepdim=True)

    # Compute the mean embedding
    texts_z_mean = texts_z.mean(dim=0)

    # Renormalize the mean embedding
    texts_z_mean /= texts_z_mean.norm(dim=-1, keepdim=True)

    return texts_z_mean

def get_clip_embeddings(sents_tokenized: torch.Tensor, clip_model) -> torch.Tensor:
    """
    Compute the CLIP embeddings for a batch of tokenized sentences.
    
    Parameters
    ----------
    sents_tokenized : torch.Tensor
        A tensor of tokenized sentences of shape [batch_size, 77].
    clip_model : Any
        The CLIP model used to encode the text.

    Returns
    -------
    torch.Tensor
        Normalized tensor of shape [batch_size, 1024] of the sentences.
    """
    
    with torch.no_grad():
        texts_z = clip_model.encode_text(sents_tokenized).float()
    
    texts_z /= texts_z.norm(dim=-1, keepdim=True)  # Normalize the embeddings

    return texts_z

def polygons_to_binary_mask(polygons: list[list[float]], image_size: tuple[int, int]) -> np.ndarray:
    """
    Converts a list of polygons into a binary mask of a specified size.

    Parameters
    ----------
    polygons : list[list[float]]
        List of polygons, each defined as a flat list of [x, y] coordinates.
    image_size : tuple[int, int]
        The dimensions (width, height) of the output binary mask.

    Returns
    -------
    np.ndarray
        A binary mask with the polygon regions filled with ones.
    """
    binary_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    for polygon in polygons:
        
        # Reshape flat list of [x, y] into a 2D array of points
        polygon_points = np.array(polygon).reshape((-1, 2))
        cv2.fillPoly(binary_mask, [polygon_points.astype(np.int32)], 1)

    return binary_mask

def apply_templates(sents: list[str], templates: list[str]) -> list[str]:
    """
    Applies a set of string templates to a list of sentences.

    Parameters
    ----------
    sents : list[str]
        A list of input sentences.
    templates : list[str]
        A list of templates with placeholders (e.g., "This is a {}").

    Returns
    -------
    list[str]
        A list of formatted sentences, with each sentence applied to all templates.
    """

    return [t.format(s) for s in sents for t in templates]

def test_forward_pass(config_path: str, dataset: dict) -> None:
    """
    Perform a forward pass on a single batch of the provided dataset using the specified model configuration and
    debug the output by displaying batch information, tensor statistics and loss components.
    Used to test ClipVS during devolopment.
    
    Parameters
    ----------
    config_path : str
        The path to the configuration file.
    dataset : dict
        The dataset containing the test data.
    """
    
    # Load the experiment configuration and set up the device
    config = load_experiment_config(config_path)
    device = config["device"]
    config["nn_batch_size"] = 2 # If changed, will be displayed a different sample

    # Initialize the model
    net = ClipVS(config).to(device)

    # Load pre-trained weights if specified in the config
    if config["nn_reload_weights_from"] is not None:
        weights_path = config["nn_reload_weights_from"]
        state_dict = torch.load(weights_path, map_location=device)
        net.load_state_dict(state_dict)

    # Set up the data loader for the test set
    data_loader = DataLoader(dataset['test'], batch_size=config["nn_batch_size"], collate_fn=lambda b: collate_fn(b, device))

    # Initialize lists to store batch losses and components
    batch_losses = []
    batch_losses_components = []
    config["verbose"] = True

    # Iterate over the test data
    for i, batch in enumerate(data_loader):
        
        # Forward pass
        output = net(batch)
        postprocessing(batch, output, config)
        loss, components = custom_loss(batch, output, config)

        # Debugging and displaying output
        debug_batch_info(batch)
        debug_tensors(output, title="Debug forward output")
        print(f"Loss components: {components}")
        debug_plotting(batch, output, title=f"Batch last el. plots - Current sample IoU accuracy: {round(output['iou_accuracies'][-1].item(), 4)}")

        # Store loss and components
        batch_losses.append(loss.item())
        batch_losses_components.append(components)

        # Exit after the first batch for testing purposes
        break

    # Output inspection and saving report
    #print(f"Final epoch mechanisms test.")
    #report = epoch_report(1, 1, config, mode='test',
     #                     test_losses=batch_losses, test_losses_components=batch_losses_components)

    # Save the results to the experiment folder
    #config["exp_folder_path"] = f"{config['output_to']}/ClipVS_experiments/experiment_4_maintr"
    #save_to_experiment_folder(config, report, f"test_forward_pass.png")
