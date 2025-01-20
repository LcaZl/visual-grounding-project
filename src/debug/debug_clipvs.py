from src.debug.debug_general import *

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.patches as patches

def debug_batch_info(batch: dict) -> None:
    """
    Debug and display information about a batch, including sizes, shapes and types of various batch elements.

    Parameters
    ----------
    batch : dict
        The batch containing tensors, images, bounding boxes, segmentations and sentences.

    """
    
    lim = 2
    
    print("=" * 50)
    print(f"Batch debug info:")

    batch_size = len(batch['ids'])
    print(f"Number of samples in the batch: {batch_size}")
    
    print("\n- IDs:")
    print(f"Type: {type(batch['ids'])}, Length: {len(batch['ids'])}")
    
    print("\n- Ground Truth Bounding Boxes:")
    print(f"Type: {type(batch['gt_bboxes'])}, Length: {len(batch['gt_bboxes'])}")
    for i, bbox in enumerate(batch['gt_bboxes'][:lim]):
        print(f"  Sample {i}: {bbox.shape if hasattr(bbox, 'shape') else len(bbox)}")

    print("\n- Image Dimensions (Width, Height):")
    print(f"Type: {type(batch['imgs_dims'])}, Length: {len(batch['imgs_dims'])}")
    for i, dims in enumerate(batch['imgs_dims'][:lim]):
        print(f"  Sample {i}: {dims}")

    print("\n- Raw Images:")
    print(f"Type: {type(batch['imgs_raw'])}, Length: {len(batch['imgs_raw'])}")
    for i, img in enumerate(batch['imgs_raw'][:lim]):
        print(f"  Sample {i}: {img.size if hasattr(img, 'size') else img.shape}")

    print("\n- Transformed Images:")
    print(f"Type: {type(batch['imgs_transformed'])}, Shape: {batch['imgs_transformed'].shape}")

    print("\n- Ground Truth Segmentations:")
    print(f"Type: {type(batch['gt_segmentations'])}, Length: {len(batch['gt_segmentations'])}")
    print(f"Segmentation unique lens: {np.unique([len(sm) for sm in batch['gt_segmentations']])}")
    for i, seg in enumerate(batch['gt_segmentations'][:lim]):
        print(f"  Sample {i}: {len(seg)} points")

    print("\n- Ground Truth Segmentation Masks:")
    print(f"Type: {type(batch['gt_segmentation_masks'])}, Length: {len(batch['gt_segmentation_masks'])}")
    for i, mask in enumerate(batch['gt_segmentation_masks'][:lim]):
        print(f"  Sample {i}: {mask.shape}")

    print("\n- Maximum Number of Sentences per Sample:")
    print(f"Max Sentences: {batch['max_texts']}")

    print("\n- Raw Sentences:")
    print(f"Type: {type(batch['texts_raw'])}, Length: {len(batch['texts_raw'])}")
    for i, sents in enumerate(batch['texts_raw'][:lim]):
        print(f"  Sample {i}: {len(sents)} sentences")

    print("\n- Transformed and Padded Sentences:")
    print(f"Type: {type(batch['texts_transformed_padded'])}, Shape: {batch['texts_transformed_padded'].shape}")

    print("\n- Sentence Padding Masks:")
    print(f"Type: {type(batch['texts_padding_masks'])}, Shape: {batch['texts_padding_masks'].shape}")

    print("End batch debug info")
    print("=" * 50)

def debug_tensors(output: dict, title: str = "Debugging") -> None:
    """
    Display debugging information about tensors and lists of tensors in the output dictionary.

    Parameters
    ----------
    output : dict
        A dictionary containing tensors and lists of tensors.
    title : str, optional
        Title for the debug output.

    """
    
    def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
        """ Print tensor statistics (shape, dtype, device, min/max/mean/std) """
        print(f"\n{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Type: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  Gradient: {tensor.grad_fn}")
        print(f"  Min: {tensor.float().min().item():.6f}, Max: {tensor.float().max().item():.6f}")
        print(f"  Mean: {tensor.float().mean().item():.6f}, Std: {tensor.float().std().item():.6f}")

    def print_list_stats(name: str, tensor_list: list) -> None:
        """ Print statistics for a list of tensors """
        list_len = len(tensor_list)
        print(f"\n{name}:")
        print(f"  List length: {list_len}")
        
        if list_len > 0 and isinstance(tensor_list[0], torch.Tensor):
            first_tensor = tensor_list[0]
            print(f"  First element shape: {first_tensor.shape}")
            print(f"  First element Dtype: {first_tensor.dtype}")
            print(f"  First element Grad_fn: {first_tensor.grad_fn}")
            print(f"  First element Min: {first_tensor.min().item():.6f}, Max: {first_tensor.max().item():.6f}")
            print(f"  First element Mean: {first_tensor.mean().item():.6f}, Std: {first_tensor.std().item():.6f}")
        else:
            print(f"  First element is not a tensor or list is empty.")

    print("=" * 50)
    print(title)

    # Loop through the output dictionary
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print_tensor_stats(key, value)
        elif isinstance(value, list):
            print_list_stats(key, value)
        else:
            print(f"\n{key}: Not a tensor or list of tensors, cannot print stats.")
            
    print("End of debug output.")
    print("=" * 50)

def debug_tensor_transformation(original_tensor: torch.Tensor, transformed_tensor: torch.Tensor, dims: list, num_samples: int = 5) -> None:
    """
    Debug the transformation of a tensor by comparing the original and transformed tensors.
    Used to compare bounding boxes coordinate before and after conversions.

    Parameters
    ----------
    original_tensor : torch.Tensor
        The original tensor before transformation.
    transformed_tensor : torch.Tensor
        The transformed tensor.
    dims : list
        The target dimensions after transformation.
    num_samples : int, optional
        The number of samples to display.

    """
    
    assert original_tensor.shape == transformed_tensor.shape, "Original and transformed tensors must have the same shape."
    
    print("=" * 50)
    print("Debugging tensor transformation:")
    print(f"Original Tensor Shape: {original_tensor.shape}")
    print(f"Transformed Tensor Shape: {transformed_tensor.shape}")
    
    num_samples = min(num_samples, original_tensor.shape[0])
    
    # Print the original and transformed values side by side
    for i in range(num_samples):
        print(f"Sample {i+1}: {original_tensor[i]} -> {transformed_tensor[i]} (target res.: {dims[i]})")

    print("End debugging output")
    print("=" * 50)

    
def debug_metrics(tensor: torch.Tensor, metric_name: str = "Metric", num_samples: int = 5) -> None:
    """
    Debug the metric values by displaying basic statistics and the first few samples. Used to display loss components.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The tensor containing the metric values.
    metric_name : str, optional
        The name of the metric.
    num_samples : int, optional
        The number of samples to display.
        
    """
    
    tensor = tensor.cpu().detach().numpy()
    num_samples = min(num_samples, tensor.shape[0])
    
    print("=" * 50)
    print(f"Metric: {metric_name}")
    print(f"Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")
    print(f"Values: {' '.join([f'{tensor[i]:.6f}' for i in range(num_samples)])}")
    print("=" * 50)


def debug_plotting(batch: dict, output: dict, title: str = "", id: int = -1) -> None:
    """
    Visualize various elements from the batch and model output, including images, bounding boxes, segmentations and feature maps inside a single organzied plot.

    Parameters
    ----------
    batch : dict
        The batch containing images, bounding boxes and ground truth segmentations.
    output : dict
        The model's output containing predicted bounding boxes, feature maps and attention masks.
    title : str, optional
        Title for the plot.
    id : int, optional
        Index of the sample to visualize.
        
    """
    
    img_raw = np.asarray(batch["imgs_raw"][id])
    img_transformed = np.clip(batch["imgs_transformed"][id].detach().cpu().numpy().transpose(1,2,0), 0, 1)
    gt_bbox = batch["gt_bboxes"][id].cpu().numpy()

    pred_bbox = output["pred_bboxes"][id].detach().cpu().numpy()
    gt_segmentation_mask = batch["gt_segmentation_masks"][id].detach().cpu().numpy()

    #imgs_emb = output["imgs_emb_resized"][id].detach().cpu().numpy().transpose(1,2,0)
    imgs_l1 = output["fpn_imgs_l1_resized"][id].detach().cpu().numpy().transpose(1,2,0)
    imgs_l2 = output["fpn_imgs_l2_resized"][id].detach().cpu().numpy().transpose(1,2,0)
    imgs_l3 = output["fpn_imgs_l3_resized"][id].detach().cpu().numpy().transpose(1,2,0)
    imgs_l1_raw = output["fpn_imgs_l1_raw_resized"][id].detach().cpu().numpy().transpose(1,2,0)
    imgs_l2_raw = output["fpn_imgs_l2_raw_resized"][id].detach().cpu().numpy().transpose(1,2,0)
    imgs_l3_raw = output["fpn_imgs_l3_raw_resized"][id].detach().cpu().numpy().transpose(1,2,0)

    fused_features = output["fused_features_resized"][id].detach().cpu().numpy().transpose(1,2,0)
    attn = output["attention_masks_resized"][id].detach().cpu().numpy().transpose(1,2,0)
    background = output["background_masks"][id].detach().cpu().numpy().transpose(1,2,0)
    foreground = output["foreground_masks"][id].detach().cpu().numpy().transpose(1,2,0)

    fig, ax = plt.subplots(3, 5, figsize=(30, 15))

    # Original image
    ax[0, 0].imshow(img_raw)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    
    # Transformed image
    ax[0, 1].imshow(img_transformed)
    ax[0, 1].set_title("Transformed image")
    ax[0, 1].axis("off")
    
    # Ground truth segmentation mask
    ax[0, 2].imshow(gt_segmentation_mask)  # Attention map overlay
    ax[0, 2].set_title("Gt Segmentation Mask")
    ax[0, 2].axis("off")
    
    # Ground truth segmentation mask overlay
    ax[0, 3].imshow(img_raw)
    gt_bbox_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], linewidth=3, edgecolor='r', facecolor='none')
    ax[0, 3].add_patch(gt_bbox_rect)

    for polygon_coords in batch['gt_segmentations'][id]:
        # Reshape and create a Polygon object for each segmentation
        gt_segmentation_polygon = Polygon(np.array(polygon_coords).reshape(-1, 2), closed=True, 
                                        fill=True, edgecolor='r', facecolor='r', alpha=0.3)
        # Add the polygon to the plot
        ax[0, 3].add_patch(gt_segmentation_polygon)
    ax[0, 3].set_title("Gt bbox and segm.")
    ax[0, 3].axis("off")
    
    # Prediction
    ax[0, 4].imshow(img_raw)
    gt_bbox_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], linewidth=3, edgecolor='b', facecolor='none')
    ax[0, 4].add_patch(gt_bbox_rect)
    pred_bbox_rect = patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2] , pred_bbox[3], linewidth=3, edgecolor='r', facecolor='none')

    ax[0, 4].add_patch(pred_bbox_rect)
    ax[0, 4].set_title("Predicted bbox in Red")
    ax[0, 4].axis("off")
    
    ax[1, 0].imshow(imgs_l1_raw, cmap="jet")
    ax[1, 0].set_title("FPN Raw L1")
    ax[1, 0].axis("off")
    
    ax[1, 1].imshow(imgs_l1,  cmap='jet')
    ax[1, 1].set_title("FPN L1")
    ax[1, 1].axis("off")
    
    ax[1, 2].imshow(imgs_l2_raw, cmap='jet')
    ax[1, 2].set_title("FPN Raw L2")
    ax[1, 2].axis("off")
    
    ax[1, 3].imshow(imgs_l2, cmap='jet')
    ax[1, 3].set_title("FPN L2")
    ax[1, 3].axis("off")

    ax[1, 4].imshow(imgs_l3_raw, cmap='jet')
    ax[1, 4].set_title("FPN Raw L3")
    ax[1, 4].axis("off")
    
    ax[2, 0].imshow(imgs_l3, cmap='jet')
    ax[2, 0].set_title("FPN L3")
    ax[2, 0].axis("off")

    ax[2, 1].imshow(fused_features, cmap='jet')
    ax[2, 1].set_title("Fused features")
    ax[2, 1].axis("off")
    
    ax[2, 2].imshow(attn, cmap='jet')
    ax[2, 2].set_title("Attention Mask")
    ax[2, 2].axis("off")

    ax[2, 3].imshow(foreground, cmap='jet')
    ax[2, 3].set_title("Foreground")
    ax[2, 3].axis("off")
    
    ax[2, 4].imshow(background, cmap='jet')
    ax[2, 4].set_title("Background")
    ax[2, 4].axis("off")
    
    #plot_prediction(img_raw, pred_bbox, gt_bbox, output["attention_masks_resized"][id].detach().cpu().numpy().transpose(1,2,0), output["iou_accuracies"][id].item())
    
    plt.suptitle(title)
    plt.show()     

    
def plot_prediction(raw_image, pred_bbox, gt_bbox, attn, accuracy):
    """
    Display an image with predicted and ground truth bounding boxes.

    Parameters
    ----------
    raw_image : np.ndarray or PIL.Image
        The raw image to display.
    pred_bbox : list or np.ndarray
        Predicted bounding box in [x, y, width, height] format.
    gt_bbox : list or np.ndarray
        Ground truth bounding box in [x, y, width, height] format.
    accuracy : float
        IoU accuracy of the predicted bounding box.

    """

    fig, ax = plt.subplots(1,2, figsize=(10, 7))
    ax[0].set_title(f"Model Prediction - Accuracy: {round(accuracy * 100, 2)}%", fontsize=16)
    ax[0].imshow(raw_image)
    
    # Add predicted bounding box in red
    pred_rect = patches.Rectangle(
        (pred_bbox[0], pred_bbox[1]),
        pred_bbox[2],
        pred_bbox[3],
        linewidth=3,
        edgecolor='r',
        facecolor='none',
        label="Predicted BBox"
    )
    
    # Add ground truth bounding box in blue
    gt_rect = patches.Rectangle(
        (gt_bbox[0], gt_bbox[1]),
        gt_bbox[2],
        gt_bbox[3],
        linewidth=3,
        edgecolor='b',
        facecolor='none',
        label="Ground Truth BBox"
    )
    ax[0].add_patch(pred_rect)
    ax[0].add_patch(gt_rect)
    ax[0].legend(loc="upper right", fontsize=12)
    ax[0].axis('off')
    
    ax[1].imshow(attn, cmap='jet')
    ax[1].set_title("Attention Mask")
    ax[1].axis("off")
    
    plt.show()

