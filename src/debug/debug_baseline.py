import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from src.debug.debug_clipvs import print_dataframe

def display_images_with_bboxes(raw_image: np.ndarray, pred_bbox: list, gt_bbox: list, accuracy: float) -> None:
  """
  Display two images: one with the predicted bounding box and the other with the ground truth bounding box.

  Parameters
  ----------
  raw_image : np.ndarray
      The raw image to display.
  pred_bbox : list or np.ndarray
      Predicted bounding box in [x, y, width, height] format.
  gt_bbox : list or np.ndarray
      Ground truth bounding box in [x, y, width, height] format.
  accuracy : float
      IoU accuracy of the predicted bounding box.

  """
  
  plt.figure(figsize=(22, 7))
  plt.suptitle(f"Model prediction - Accuracy {round(accuracy * 100, 2)}%", fontsize=16)

  # First image with predicted bounding box
  width = pred_bbox[2] - pred_bbox[0]
  height = pred_bbox[3] - pred_bbox[1]
  ax1 = plt.subplot(1, 2, 1)
  ax1.imshow(raw_image)
  rect1 = patches.Rectangle((pred_bbox[0], pred_bbox[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
  ax1.add_patch(rect1)
  plt.axis('off')
  plt.title('Predicted Bounding Box')

  # Second image with ground truth bounding box
  ax2 = plt.subplot(1, 2, 2)
  ax2.imshow(raw_image)
  rect2 = patches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], linewidth=2, edgecolor='r', facecolor='none')
  ax2.add_patch(rect2)
  plt.axis('off')
  plt.title('Ground truth Bounding Box')

  plt.show()

def visualise_probabilities(images: list, sents: list[str], texts_p: torch.Tensor, k: int = 5) -> None:
  """
  Visualize the top-k sentence probabilities for each image, displaying the images alongside the 
  corresponding sentences and their predicted probabilities.

  Parameters
  ----------
  images : list
      A list of images to display.
  sents : list of str
      A list of sentences corresponding to the images.
  texts_p : torch.Tensor
      A tensor containing the probabilities for each sentence.
  k : int, optional
      The number of top predictions to display for each image (default is 5).

  """
  
  # Ensure k does not exceed the number of sentences
  if k > len(sents):
      k = len(sents)
  
  # Generate sentence ID mapping
  sents_with_ids = {f"ID-{i}": s for i, s in enumerate(sents)}
  df = pd.DataFrame(list(sents_with_ids.items()), columns=['ID', 'Sentence'])

  # Prepare the probabilities tensor
  texts_p = texts_p.transpose(0, 1)
  topk_p, topk_labels = texts_p.cpu().topk(k, dim=1)
  
  num_images = len(images)
  plt.figure(figsize=(20, 4 * ((num_images + 1) // 2)))
  
  for i, image in enumerate(images):
      
      # Adjust subplot indices for a tighter layout
      ax1 = plt.subplot((num_images + 1) // 2, 4, 4 * (i // 2) + (i % 2) * 2 + 1)
      ax1.imshow(image)
      ax1.axis("off")
      
      ax2 = plt.subplot((num_images + 1) // 2, 4, 4 * (i // 2) + (i % 2) * 2 + 2)
      y = np.arange(k)
      probabilities = topk_p[i].numpy()
      sentence_indices = topk_labels[i].numpy()
      sentence_ids = [f"ID-{index}" for index in sentence_indices]  # Use sentence IDs
      ax2.barh(y, probabilities, align='center')
      ax2.set_yticks(y)
      ax2.set_yticklabels(sentence_ids, fontsize=10)
      ax2.invert_yaxis()
      ax2.set_xlabel("Probability")
      ax2.grid(True)
  
  # Adjust spacing between subplots for better visibility
  plt.subplots_adjust(wspace=0.3, hspace=0.3)
  plt.title('Probabilities Image/Sentences')

  print_dataframe(df, title='Sentences ID Visualized in Charts.\n')
  plt.show()


def visualise_similarity(similarity: torch.Tensor, images_fp: list[str], texts: list[str]) -> None:
  """
  Visualize the cosine similarity scores between text and image features, alongside the corresponding images.

  Parameters
  ----------
  similarity : torch.Tensor
      A tensor containing the cosine similarity scores between text and image features.
  images_fp : list of str
      A list of file paths to the images to be displayed.
  texts : list of str
      A list of text sentences corresponding to the images.

  """
  similarity = similarity.cpu().numpy()
  count = len(texts)

  sents_with_ids = {f"ID-{i}": s for i, s in enumerate(texts)}
  plt.figure(figsize=(18, 12))

  # Show similarity scores
  plt.imshow(similarity, vmin=0.1, vmax=0.3)

  plt.yticks(range(count), sents_with_ids.keys(), fontsize=18)
  plt.xticks()

  # Visualise each image
  for i, image_fp in enumerate(images_fp):
    image = image_fp.convert("RGB")
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

  # Print the scores
  for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
      plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

  # Update spines
  for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side]

  # Change plot limits
  plt.xlim([-0.5, len(images_fp) - 0.5])
  plt.ylim([count + 0.5, -2])

  plt.title("Cosine similarity between text and image features", size=20)
  plt.show()
  
def plot_bbox_prediction(bbox: list, sample: dict, title: str = "Image with Bounding Boxes and Labels") -> None:
    """
    Visualize the predicted bounding box on an image along with the ground truth label.

    Parameters
    ----------
    bbox : list
        The predicted bounding box in [x, y, width, height] format.
    sample : dict
        A dictionary containing the image and ground truth label.
    title : str, optional
        The title of the plot (default is "Image with Bounding Boxes and Labels").

    """
    
    # Calculate width and height of the bounding box
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    fig, ax = plt.subplots(figsize=(22, 7))
    ax.imshow(sample['img_raw'])
    rect = patches.Rectangle((bbox[0], bbox[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Annotate with the ground truth label
    ax.text(bbox[0], bbox[1] - 10, sample['gt_category'], color='red', fontsize=12)
    ax.axis('off')
    plt.title(title)
    plt.show()

    
def debug_baseline_output(config: dict, output: dict, sample: dict) -> None:
    """
    Debug the baseline output by displaying sentences, ground truth and predicted bounding boxes,
    as well as visualizing similarities and probabilities.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data settings.
    output : dict
        Output dictionary containing predicted bounding boxes, similarities, and probabilities.
    sample : dict
        Sample dictionary containing the raw image and sentence data.
        
    """
    
    print('Sentences:')
    for sent in list(sample['sents_df']['sent']):
        print(f' - {sent}')
    
    print(f'Ground truth box :', sample['gt_bbox'])
    print(f'Predicted box    :', output["pred_bbox"])

    # Display output dataframe
    print_dataframe(output["ext_output_df"], f"Produced output dataframe with {config['bs_supp_model']}")

    # If the model is YOLO, show its specific output
    if config["bs_supp_model"] == "YOLO":
        print(f"YOLO output:\n")
        output["ext_output_df"].show()

    # Display images with bounding boxes
    display_images_with_bboxes(sample['img_raw'], output["pred_bbox"], sample['gt_bbox'], output["iou_accuracy"])

    # Visualize similarity and probabilities based on sentence vector type
    if config["bs_sents_vector_type"] in ["Mean"]:
        visualise_similarity(output["similarity"], output["cropped_images"], [config["bs_sents_vector_type"]])
        visualise_probabilities(output["cropped_images"], [config["bs_sents_vector_type"]], output["text_prob"], k=5)
    else:
        visualise_similarity(output["similarity"], output["cropped_images"], sample['sents_raw'])
        visualise_probabilities(output["cropped_images"], list(sample['sents_raw']), output["text_prob"], k=5)
