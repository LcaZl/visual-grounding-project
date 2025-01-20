from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from src.baseline.model import baselineCLIP

def iou_accuracy(prediction: tuple[float, float, float, float], ground_truth: tuple[float, float, float, float]) -> float:    
    """
    Compute the Intersection over Union (IoU) accuracy between a predicted bounding box and the ground truth.

    Parameters
    ----------
    prediction : tuple[float, float, float, float]
        Predicted bounding box in the format (x_min, y_min, x_max, y_max).
    ground_truth : tuple[float, float, float, float]
        Ground truth bounding box in the format (x, y, width, height).

    Returns
    -------
    float
        IoU accuracy.
    """

    x_min, y_min, x_max, y_max = prediction
    x, y, w, h = ground_truth

    def computeIntersectionArea(fx1, fy1, fx2, fy2, sx1, sy1, sx2, sy2):
        """
        Compute the intersection area of two rectangles given their corner coordinates.

        Parameters
        ----------
        fx1, fy1, fx2, fy2 : float
            Coordinates of the first rectangle (top-left and bottom-right corners).
        sx1, sy1, sx2, sy2 : float
            Coordinates of the second rectangle (top-left and bottom-right corners).

        Returns
        -------
        float
            Area of the intersection. If no intersection exists, returns 0.
        """
        dx = min(fx2, sx2) - max(fx1, sx1)
        dy = min(fy2, sy2) - max(fy1, sy1)
        if dx >= 0 and dy >= 0:
            return dx * dy
        return 0

    intersection = computeIntersectionArea(x_min, y_min, x_max, y_max, x, y, x + w, y + h)

    # Compute areas of the prediction and ground truth rectangles
    area1 = (x_max - x_min) * (y_max - y_min)
    area2 = w * h

    # IoU accuracy
    return intersection / (area1 + area2 - intersection)

def evaluate(model: baselineCLIP, loader: DataLoader) -> float:
    """
    Evaluate the baseline model's performance on a dataset.

    Parameters
    ----------
    model : BaselineModel
        The baseline model.
    loader : DataLoader
        Dataset to use for evaluation.

    Returns
    -------
    float
        Mean IoU accuracy over the given dataset
    """
    accuracies = []  # List to store IoU accuracies for all samples
    total_samples = len(loader) # Total number of samples in the dataset

    with tqdm(total=total_samples, desc="Baseline Evaluation", unit="batch") as pbar:
        for batch in loader:
            for sample in batch:


                output = model.forward(sample)

                # If no bounding box is predicted, set accuracy to 0
                if output["pred_bbox"] is None:
                    acc = 0
                else:
                    acc = iou_accuracy(output["pred_bbox"], sample['gt_bbox'])

                accuracies.append(acc)

                pbar.set_postfix({"IoU Accuracy": f"{np.mean(accuracies):.4f}"})
                pbar.update()

    return np.mean(accuracies)


def collate_fn_bs(batch: list) -> list:
    """
    Collate function for the baseline model.
    
    Parameters
    ----------
    batch : list
        List of samples in the batch.

    Returns
    -------
    list
        Flattened list of samples.
    """
    return [el for el in batch]
