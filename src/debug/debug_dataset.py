import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle

from src.debug.debug_general import print_dataframe, print_list, print_dict, plot_matrix
from src.dataset.dataset import refcocog

def plot_segmentation_and_bbox(image: np.ndarray, polygon_coords: list, bbox_coords: list) -> None:
    """
    Visualize the segmentation polygon and bounding box on the image.

    Parameters
    ----------
    image : np.ndarray
        The image on which to visualize the segmentation and bounding box.
    polygon_coords : list
        A list of coordinates representing the polygon for the segmentation.
    bbox_coords : list
        A list of coordinates representing the bounding box in [x_min, y_min, x_max, y_max] format.

    """
    
    # Convert image to NumPy if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]
    
    # Reshape polygon coordinates into (x, y) pairs
    polygon_points = np.array(polygon_coords).reshape(-1, 2)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    
    # Create and plot the polygon
    polygon = Polygon(polygon_points, closed=True, fill=True, edgecolor='r', facecolor='r', alpha=0.3)
    ax.add_patch(polygon)
    
    # Plot the bounding box
    x_min, y_min, x_max, y_max = bbox_coords
    bbox = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='b', facecolor='none', linewidth=2)
    ax.add_patch(bbox)
    
    ax.set_title("Ground Truth Segmentation and Bounding Box")
    ax.axis("off")
    plt.show()

def sample_info(dataset: refcocog, sample_id: int) -> None:
    """
    Display detailed information about a sample from the dataset, including raw image, bounding boxes, 
    segmentation and encoded image.

    Parameters
    ----------
    dataset : refcocog
        refococog dataset from which the sample is extracted.
    sample_id : int
        The index of the sample in the dataset.

    """
    
    print(f"Sample {sample_id} information:")
    sample = dataset.__getitem__(sample_id)
    
    # Extract relevant information
    raw_image = np.asarray(sample['img_raw'])
    bbox = sample['gt_bbox']
    category = sample['gt_category']
    clip_image = sample['img_transformed'].permute(1, 2, 0).cpu()  
    segmentation_mask = sample['gt_segmentation_mask']
    
    # Print dataframes and raw sentences
    print_dataframe(sample['sample_df'], title=f"Sample dataframe (sample_df)")
    print_dataframe(sample['sents_df'], title="Sample sentences dataframe (sents_df)")
    if dataset.enh_sents: 
        print_dataframe(sample['sents_extra_df'], title=f"Top-{dataset.sm.k} most similar sentences dataframe (sents_extra_df)")
    
    print_list(sample["sents_raw"], title=f"Sample raw sentences:\n" +
          f"(Template: {'Enabled' if dataset.templates is not None else 'Disabled'})")
    
    # Print additional sample details
    print(f"\nSample from __get_item__ keys: {sample.keys()}")
    print(f"CLIP encoded image shape: {sample['img_transformed'].shape}")
    print(f"YOLO encoded image shape: {sample['img_raw'].size}")
    print_dict(sample, title="Sample dict from __getitem__", avoid_keys=["sample_df", "sents_df","sents_extra_df"])
    
    # Plot the images and bounding boxes
    plt.figure(figsize=(22, 10))
    plt.suptitle('Sample Info', fontsize=16, y=1.05)
    
    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(raw_image)
    plt.axis('off')
    plt.title('Original Image')

    # Image with Bounding Boxes
    ax = plt.subplot(2, 3, 2)
    ax.imshow(raw_image)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(bbox[0], bbox[1], category, color='red', fontsize=12)
    plt.axis('off')
    plt.title('Image with Bounding Boxes and Labels')

    # CLIP Encoded Image
    plt.subplot(2, 3, 3)
    plt.imshow(np.clip(clip_image, 0, 1))
    plt.axis('off')
    plt.title('Image Example Encoded with CLIP')

    # Ground Truth Segmentation with Bounding Box
    ax = plt.subplot(2, 3, 4)
    ax.imshow(raw_image)
    polygon = Polygon(np.array(sample['gt_segmentation']).reshape(-1, 2), closed=True, fill=True, edgecolor='r', facecolor='r', alpha=0.3)
    ax.add_patch(polygon)
    
    bbox_rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='b', facecolor='none', linewidth=2)
    ax.add_patch(bbox_rect)
    
    ax.set_title("Ground Truth Segmentation and Bounding Box")
    ax.axis("off")

    # Ground Truth Segmentation Mask
    plt.subplot(2, 3, 5)
    plt.imshow(segmentation_mask, cmap='gray')
    plt.axis('off')
    plt.title('Ground Truth Segmentation Mask')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def dataset_details(dt: refcocog) -> None:
    """
    Display detailed information about a dataset, including metadata, instances, categories, images, 
    annotations, sentences and samples. Also visualizes sentence similarity if similarity matrix is available.

    Parameters
    ----------
    dt : refcocog
        The refcocog dataset.

    """
    
    print(f'\n{dt.split.upper()} DATASET {dt.dataset_name} ({dt.splitBy}) INFO:')
    print_dict(dt.info)
    
    print(f' - Dataset licenses:')
    for el in dt.licenses:
        print(f"{el['id']} - {el['name']} -> {el['url']}")

    # Print dataset instances
    print(f'\nInstances:')
    print(f' - keys: {dt.instances.keys()}')
    print(f' - Categories    #: {len(dt.categories)}')
    print(f" - Images        #: {len(dt.images)}")
    print(f" - Annotations   #: {len(dt.annotations)}")
    print(f" - Sentences: {len(dt.sentences)}")
    print(f" - Sent/Image: {len(dt.samples.groupby('image_id'))}")
    print(f" - Raw refs.     #: {len(dt.refs)}")
    print(f" - Final samples #: {len(dt.samples)}\n")
    print_dataframe(pd.DataFrame(dt.samples.groupby('split').size(), columns=['#']))
    print(f'Images transformation:\n', dt.images_preprocess)

    # Display data
    print(f"\nLoaded data:\n")
    print_dataframe(dt.categories, limit=2, title=f'Categories:\n')
    print_dataframe(dt.sentences, limit=5, title=f'Sentences:\n', sort_by='sent')
    print_dataframe(dt.images, limit=2, title=f'Images:\n')
    print_dataframe(dt.annotations, limit=2, title=f'Annotations:\n')
    print_dataframe(dt.refs, limit=2, title=f'References', sort_by='ann_id')
    
    # Display sample data
    cols = ['split', 'ann_id', 'sent_ids', 'image_id', 'category_id', 'name', 'supercategory', 'file_name', 'area', 'bbox', 'width', 'height']
    print_dataframe(dt.samples[cols], title='Samples')

    # If enhanced sentences are available, display related sentences
    if dt.enh_sents:
        print(f'Example of Top-{dt.sm.k} most similar sentences')
        for main_id, related_ids in dt.sm.sent2topk.items():
            print(f"Matrix sentence id: {main_id} -> {dt.sm.reverse_vocab[main_id]} (original id)")
            print(f"Related sentences ids: {related_ids} -> {[dt.sm.reverse_vocab[id] for id in related_ids]} (original ids)")
            main_sent = dt.sentences.loc[dt.sm.reverse_vocab[main_id]]
            related_sents = dt.sentences.loc[[dt.sm.reverse_vocab[id] for id in related_ids]]
            print(f'Main sentence:', main_sent['raw'])
            print_dataframe(related_sents, title=f"Related/Top-{dt.sm.k} most similar sentences")
            break

        # Visualize similarity matrix for sentence embeddings
        plot_matrix(dt.sm.sim_matrix, title='Similarity matrix between sentences embeddings (Limited to 20 sentences)', xlabel='Sent', ylabel='Sent', max_x=20, max_y=20)