import torch
import torch.nn.functional as F
from src.debug.debug_clipvs import debug_tensor_transformation

def convert_bboxes_to_resolutions(bboxes: torch.Tensor, resolutions: list, device: torch.device) -> torch.Tensor:
    """
    Convert bounding boxes from normalized format [xmin, ymin, xmax, ymax] to actual image dimensions.
    
    Parameters
    ----------
    bboxes : torch.Tensor
        Tensor of bounding boxes in normalized format [xmin, ymin, xmax, ymax].
    resolutions : list
        List of tuples with image resolutions [(width, height)] for each image in the batch.
    device : torch.device
        The device (CPU or GPU) on which the tensor should reside.

    Returns
    -------
    torch.Tensor
        Tensor of bounding boxes converted to the format [xmin, ymin, width, height] in image dimensions.
    """
    bboxes = bboxes.to(device)
    converted_bboxes = []

    for i, (bbox, (scale_w, scale_h)) in enumerate(zip(bboxes, resolutions)):

        # Convert normalized [xmin, ymin, xmax, ymax] to image dimensions
        xmin = bbox[0] * scale_w
        ymin = bbox[1] * scale_h
        width = bbox[2] * scale_w
        height = bbox[3] * scale_h

        # Stack the converted bounding box tensor in the format [xmin, ymin, width, height]
        converted_bbox = torch.stack([xmin, ymin, width, height])
        converted_bboxes.append(converted_bbox)

    return torch.stack(converted_bboxes, dim=0).to(device)


def process_map(tensor: torch.Tensor, imgs_dims: list, percentile: float = 50) -> list:
    """
    Process a map by reducing channels, rescaling to original image size and generating segmentation maps based on a percentile threshold.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor representing maps with shape [batch_size, channels, height, width].
    imgs_dims : list
        List of image dimensions [(width, height)] for each image in the batch.
    percentile : float, optional
        Percentile threshold for generating segmentation maps

    Returns
    -------
    list
        List of rescaled tensors, each with shape matching the original image sizes.
    """
    
    def rescale_to_original_size(tensor: torch.Tensor, original_sizes: list) -> list:
        """
        Rescale a tensor to match the original image size using bilinear interpolation.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape [batch_size, height, width].
        original_sizes : list
            List of original image sizes [(width, height)] for each image in the batch.

        Returns
        -------
        list
            List of resized tensors matching the original image sizes.
        """
        resized_maps = []
        for i in range(tensor.size(0)):
            original_size = original_sizes[i]  # Get (width, height)
            resized_map = F.interpolate(tensor[i:i+1].unsqueeze(0), size=(original_size[1], original_size[0]), mode='bilinear')
            
            # Normalize values to range [0, 1]
            resized_map_min = resized_map.min()
            resized_map_max = resized_map.max()
            if resized_map_max > resized_map_min:
                resized_map = (resized_map - resized_map_min) / (resized_map_max - resized_map_min)
            else:
                resized_map = resized_map.clamp(0, 1)  # If min == max, clamp the values directly
            
            resized_maps.append(resized_map.squeeze(0))  # Remove unnecessary dimensions
        
        return resized_maps

    # Reduce channels
    reduced_tensors = tensor.mean(dim=1)

    # Rescale the normalized map to the original image size
    resized_tensors = rescale_to_original_size(reduced_tensors, imgs_dims)
        
    return resized_tensors

def apply_percentile_threshold(attention_maps: list, percentile: float) -> tuple:
    """
    Apply a percentile-based threshold to create binary foreground and background masks for each map.
    
    Parameters
    ----------
    attention_maps : list
        List of attention maps, where each map is a tensor of shape [height, width].
    percentile : float
        Percentile threshold for creating the segmentation map.

    Returns
    -------
    tuple
        Tuple containing two lists of tensors: foreground masks and background masks.
    """
    foreground_masks = []
    background_masks = []

    for attention_map in attention_maps:
        
        # Compute the percentile threshold for the current attention map
        threshold = torch.quantile(attention_map.view(-1), percentile / 100.0)
        
        foreground_mask = (attention_map > threshold).float()
        background_mask = (attention_map <= threshold).float()
        
        foreground_masks.append(foreground_mask)
        background_masks.append(background_mask)

    return foreground_masks, background_masks
    
def postprocessing(batch: dict, output: dict, config: dict) -> None:
    """
    Perform postprocessing on the output of ClipVS, including resizing feature maps, adjusting bounding boxes format
    and generating segmentation masks for foreground and background. This prepare the output to be presented.
    
    Parameters
    ----------
    batch : dict
        The batch containing ground truth and image dimensions.
    output : dict
        The model's output containing various maps and bounding boxes.
    config : dict
        Configuration dictionary containing settings like percentile for segmentation.

    """    
    output["fpn_imgs_l1_resized"] = process_map(output["fpn_imgs_l1"], batch['imgs_dims'], percentile=config["nn_fg_bg_percentile"])
    output["fpn_imgs_l2_resized"] = process_map(output["fpn_imgs_l2"], batch['imgs_dims'], percentile=config["nn_fg_bg_percentile"])
    output["fpn_imgs_l3_resized"] = process_map(output["fpn_imgs_l3"], batch['imgs_dims'], percentile=config["nn_fg_bg_percentile"])
    
    output["fpn_imgs_l1_raw_resized"] = process_map(output["fpn_imgs_l1_raw"], batch['imgs_dims'], percentile=config["nn_fg_bg_percentile"])
    output["fpn_imgs_l2_raw_resized"] = process_map(output["fpn_imgs_l2_raw"], batch['imgs_dims'], percentile=config["nn_fg_bg_percentile"])
    output["fpn_imgs_l3_raw_resized"] = process_map(output["fpn_imgs_l3_raw"], batch['imgs_dims'], percentile=config["nn_fg_bg_percentile"])
    
    output["fused_features_resized"] = process_map(output["fused_features"], batch['imgs_dims'], percentile=config["nn_fg_bg_percentile"])
    output["orig_bboxes"] = output["pred_bboxes"].clone()
    output["pred_bboxes"] = convert_bboxes_to_resolutions(output["pred_bboxes"], batch["imgs_dims"], config["device"])
    if config["verbose"]:
        debug_tensor_transformation(output["orig_bboxes"], output["pred_bboxes"], batch["imgs_dims"])
