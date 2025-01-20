import torch
import torch.nn as nn

from src.debug.debug_clipvs import debug_metrics

def iou_accuracy(predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
    """
    
    Compute Intersection over Union (IoU) accuracy for batches of bounding boxes.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Tensor of shape (batch_size, 4) representing predicted bounding boxes in the format (x_min, y_min, x_max, y_max).
    ground_truths : torch.Tensor
        Tensor of shape (batch_size, 4) representing ground truth bounding boxes in the format (x, y, w, h).
    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size) containing IoU values for each pair of bounding boxes.
    
    Notes
    -------
    
    IoU = (Area of Intersection) / (Area of Union)
    
    Where:
    - Intersection: Overlapping area between predicted and ground truth bounding boxes.
    - Union: Total area covered by both bounding boxes.
    
    IoU measures the overlap between predicted and ground truth bounding boxes, with values ranging from 0 to 1.
    Higher IoU indicates better localization accuracy.
    """
    
    # Extract predictions coordinates
    x_min_pred = predictions[:, 0]
    y_min_pred = predictions[:, 1]
    x_max_pred = x_min_pred + predictions[:, 2]
    y_max_pred = y_min_pred + predictions[:, 3]
    
    # Extract ground truth coordinates and convert (x, y, w, h) to (x_min, y_min, x_max, y_max)
    x_min_gt = ground_truths[:, 0]
    y_min_gt = ground_truths[:, 1]
    x_max_gt = x_min_gt + ground_truths[:, 2]
    y_max_gt = y_min_gt + ground_truths[:, 3]
    
    # Compute intersection coordinates
    inter_x_min = torch.max(x_min_pred, x_min_gt)
    inter_y_min = torch.max(y_min_pred, y_min_gt)
    inter_x_max = torch.min(x_max_pred, x_max_gt)
    inter_y_max = torch.min(y_max_pred, y_max_gt)
    
    # Compute intersection area
    inter_width = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_height = torch.clamp(inter_y_max - inter_y_min, min=0)
    intersection = inter_width * inter_height
    
    # Compute areas of prediction and ground truth boxes
    area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    area_gt = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)
    
    # Compute union area
    union = area_pred + area_gt - intersection

    # Compute IoU
    iou = intersection / union
    return iou


def giou_loss(pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Generalized IoU (GIoU) loss for each pair of predicted and ground truth bounding boxes.
    
    Parameters
    ----------
    pred_bboxes : torch.Tensor
        Tensor of shape (batch_size, 4) representing predicted bounding boxes in the format (x_min, y_min, x_max, y_max).
    gt_bboxes : torch.Tensor
        Tensor of shape (batch_size, 4) representing ground truth bounding boxes in the format (x, y, w, h).
    
    Returns
    -------
    torch.Tensor
        Tensor of shape [batch_size] containing GIoU loss values.
    
    Notes
    -------
    GIoU = IoU - ((Area of Enclosing Box - Area of Union) / Area of Enclosing Box)
    GIoU Loss = 1 - GIoU
    
    - GIoU extends IoU by accounting for the smallest enclosing box that contains both predicted and ground truth boxes.
    """

    # Compute the coordinates of the intersection rectangle
    x1_inter = torch.max(pred_bboxes[:, 0], gt_bboxes[:, 0])
    y1_inter = torch.max(pred_bboxes[:, 1], gt_bboxes[:, 1])
    x2_inter = torch.min(pred_bboxes[:, 0] + pred_bboxes[:, 2], gt_bboxes[:, 0] + gt_bboxes[:, 2])
    y2_inter = torch.min(pred_bboxes[:, 1] + pred_bboxes[:, 3], gt_bboxes[:, 1] + gt_bboxes[:, 3])

    # Compute the area of the intersection rectangle
    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)
    inter_area = inter_width * inter_height

    # Compute the area of both the predicted and ground truth bounding boxes
    pred_area = pred_bboxes[:, 2] * pred_bboxes[:, 3]
    gt_area = gt_bboxes[:, 2] * gt_bboxes[:, 3]

    # Compute the union area
    union_area = pred_area + gt_area - inter_area

    # IoU
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    # Compute the coordinates of the smallest enclosing box
    x1_enclosing = torch.min(pred_bboxes[:, 0], gt_bboxes[:, 0])
    y1_enclosing = torch.min(pred_bboxes[:, 1], gt_bboxes[:, 1])
    x2_enclosing = torch.max(pred_bboxes[:, 0] + pred_bboxes[:, 2], gt_bboxes[:, 0] + gt_bboxes[:, 2])
    y2_enclosing = torch.max(pred_bboxes[:, 1] + pred_bboxes[:, 3], gt_bboxes[:, 1] + gt_bboxes[:, 3])

    # Compute the area of the enclosing box
    enclosing_width = torch.clamp(x2_enclosing - x1_enclosing, min=0)
    enclosing_height = torch.clamp(y2_enclosing - y1_enclosing, min=0)
    enclosing_area = enclosing_width * enclosing_height

    # GIoU
    giou = iou - (enclosing_area - union_area) / torch.clamp(enclosing_area, min=1e-6)

    # GIoU Loss
    giou_loss = 1 - giou

    return giou_loss

def get_bce_loss(attention_maps_resized: list[torch.Tensor], segmentation_masks: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute Binary Cross-Entropy (BCE) loss for attention and segmentation masks.

    Parameters
    ----------
    attention_maps_resized : list[torch.Tensor]
        List of resized attention maps, each of shape [1, width, height].
    segmentation_masks : list[torch.Tensor]
        List of ground truth segmentation masks, each of shape [1, width, height].

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the mean BCE loss across the batch.
    """
    bce_loss = nn.BCELoss()
    losses = []

    for attention_map, segmentation_mask in zip(attention_maps_resized, segmentation_masks):
        losses.append(bce_loss(attention_map.squeeze(), segmentation_mask.float()))

    return torch.stack(losses)

def smooth_l1_loss(predictions: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Compute the Smooth L1 loss between the predicted and ground truth bounding boxes.
    
    Parameters
    ----------
    predictions : torch.Tensor
        The predicted bounding boxes in [x_min, y_min, x_max, y_max] format.
    ground_truth : torch.Tensor
        The ground truth bounding boxes in [x_min, y_min, width, height] format.
    
    Returns
    -------
    torch.Tensor
        The computed Smooth L1 loss for each batch.
    """
    
    # Convert ground truth from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
    gt_x_min = ground_truth[:, 0]
    gt_y_min = ground_truth[:, 1]
    gt_width = ground_truth[:, 2]
    gt_height = ground_truth[:, 3]
    gt_x_max = gt_x_min + gt_width
    gt_y_max = gt_y_min + gt_height
    
    # Create ground truth in [x_min, y_min, x_max, y_max] format
    ground_truth = torch.stack([gt_x_min, gt_y_min, gt_x_max, gt_y_max], dim=1)
    
    # Initialize the SmoothL1Loss function
    loss_fn = nn.SmoothL1Loss(reduction='none')  # 'none' to get the loss for each element
    
    # Compute the Smooth L1 loss
    loss = loss_fn(predictions, ground_truth)
    
    # Sum the losses across the 4 bounding box coordinates (x_min, y_min, x_max, y_max)
    batch_loss = loss.sum(dim=1)
    
    return batch_loss

def custom_loss(
    batch: dict[str, object],
    output: dict[str, object],
    config: dict[str, object]
) -> tuple[torch.Tensor, dict[str, float], float]:
    """
    Compute the custom loss function for a model, integrating multiple loss components based on the configuration.

    Parameters
    ----------
    batch : dict
        Dictionary containing the batch data. Used keys are:
        - 'gt_bboxes': Ground truth bounding boxes (Tensor of shape [batch_size, 4]).
        - 'gt_segmentation_masks': Ground truth segmentation masks.
        - 'texts_padding_masks': Padding masks for text embeddings.
    output : dict
        Dictionary containing the model's outputs. Used keys are:
        - 'pred_bboxes': Predicted bounding boxes.
        - 'attention_masks_resized': Resized attention maps.
        - 'texts_embeddings': Text embeddings generated by the model.
        - 'imgs_embeddings': Image embeddings generated by the model.
    config : dict
        Configuration dictionary containing some settings. Used are:
        - "device": Device (CPU or GPU).
        - "nn_l1_lambda": Weight for L1 loss.
        - "nn_giou_lambda": Weight for GIoU loss.
        - "nn_bce_lambda": Weight for BCE loss.
        - "verbose": If True, debug metrics are printed.

    Returns
    -------
    torch.Tensor
        Total loss value.
    dict
        Dictionary containing individual loss components.
        
    """    
    
    output["giou_losses"] = giou_loss(output["pred_bboxes"], batch['gt_bboxes'])
    output["l1_losses"] = smooth_l1_loss(output["pred_bboxes"], batch['gt_bboxes'])
    output["iou_accuracies"] = iou_accuracy(output["pred_bboxes"], batch['gt_bboxes'])
    output["bce_losses"] = get_bce_loss(output["attention_masks_resized"], batch["gt_segmentation_masks"])
    
    loss = (
        (output["giou_losses"].mean() * config["nn_giou_lambda"]) +
        (output["l1_losses"].mean() * config["nn_l1_lambda"]) +
        (output["bce_losses"].mean() * config["nn_bce_lambda"]) 
        
    )
    
    components = {
        "GIoU": round((output["giou_losses"].mean() * config["nn_giou_lambda"]).item(), 2),
        "L1": round((output["l1_losses"].mean() * config["nn_l1_lambda"]).item(), 2),
        "BCE": round((output["bce_losses"].mean() * config["nn_bce_lambda"]).item(), 2),
        "IoU Acc.": round(output["iou_accuracies"].mean().item(), 4)
    }
       

    if config["verbose"]:

        debug_metrics(output["l1_losses"], "L1 Losses")
        debug_metrics(output["giou_losses"], "GIoU Losses")
        debug_metrics(output["bce_losses"], "BCE Losses")
        debug_metrics(output["iou_accuracies"], "IoU Accuracies")

        
    return loss, components