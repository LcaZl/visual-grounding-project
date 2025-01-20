
import torch

def collate_fn(batch: list[dict], device: torch.device) -> dict:
    """
    Collate function to prepare a batch of data for ClipVS.

    Parameters
    ----------
    batch : List[Dict]
        list of samples, where each sample is a dictionary as returned from refcocog dataset class.
    device : torch.device
        The device to which tensors are moved.

    Returns
    -------
    Dict
        A dictionary containing the batched data with the following keys:
        - "ids": List of sample IDs.
        - "gt_categories": List of ground truth category labels.
        - "gt_area": List of ground truth object areas.
        - "gt_bboxes": torch.Tensor of shape [batch_size, 4], containing ground truth bounding boxes.
        - "imgs_dims": List of tuples [(width, height)] for image dimensions.
        - "imgs_raw": List of raw PIL images.
        - "imgs_transformed": torch.Tensor of shape [batch_size, C, H, W], containing transformed images.
        - "gt_segmentations": List of ground truth segmentations.
        - "gt_segmentation_masks": List of tensors, each containing a binary segmentation mask.
        - "max_texts": int, the maximum number of sentences in the batch.
        - "texts_raw": List of raw sentence lists for each sample.
        - "texts_transformed_padded": torch.Tensor of shape [batch_size, max_num_sents, 77], padded tokenized sentences.
        - "texts_padding_masks": torch.Tensor of shape [batch_size, max_num_sents], indicating valid sentences.
    """
    
    ids = [sample['id'] for sample in batch]
    imgs_dims = [sample['img_dim'] for sample in batch]
    gt_bboxes = torch.tensor([sample['gt_bbox'] for sample in batch]).to(device)
    gt_categories = [sample['gt_category'] for sample in batch]
    gt_area = [sample['gt_area'] for sample in batch]
    imgs_raw = [sample['img_raw'] for sample in batch]
    sents_raw = [sample['sents_raw'] for sample in batch]

    imgs_transfomed = torch.stack([sample['img_transformed'] for sample in batch]).to(device)
    gt_segmentations = [sample['gt_segmentation'] for sample in batch]
    gt_segmentation_masks = [torch.tensor(sample['gt_segmentation_mask']).to(device) for sample in batch]
    
    max_num_sents = max(len(sample['sents_raw']) for sample in batch)

    # Pad the sentences and create a mask
    sents_padding_masks = []
    sents_transformed_padded = []
    for sample in batch:
        
        # Tokenized sentences tensor [num_sents, 77]
        sentences = sample['sents_tokenized']  # Shape: [num_sents, 77]
        num_sents = sentences.shape[0]

        # Pad sentences to the max number of sentences in the batch
        # Create padding tensor with shape [padding_needed, 77]
        padding_needed = max_num_sents - num_sents
        padding = torch.zeros(padding_needed, sentences.shape[1], dtype=sentences.dtype).to(device)  # Shape: [padding_needed, 77]
        
        # Concatenate the original sentences with the padding
        padded_sents = torch.cat([sentences, padding], dim=0)  # Shape: [max_num_sents, 77]

        # Create a mask: 1 for valid sentences, 0 for padded ones
        mask = torch.cat([torch.ones(num_sents), torch.zeros(padding_needed)], dim=0).to(device)  # Shape: [max_num_sents]

        # Append the padded sentences and mask to the respective lists
        sents_transformed_padded.append(padded_sents)
        sents_padding_masks.append(mask)

    # Stack padded sentences and masks
    sents_transformed_padded = torch.stack(sents_transformed_padded)  # Shape: [batch_size, max_num_sents, 77]
    sents_padding_masks = torch.stack(sents_padding_masks)  # Shape: [batch_size, max_num_sents]

    batch = {
        "ids" : ids,
        "gt_categories" : gt_categories,
        "gt_area" : gt_area,
        "gt_bboxes" : gt_bboxes,
        "imgs_dims" : imgs_dims, # (width, height)
        "imgs_raw" : imgs_raw,
        "imgs_transformed" : imgs_transfomed,
        "gt_segmentations" : gt_segmentations,
        "gt_segmentation_masks" : gt_segmentation_masks,
        "max_texts" : max_num_sents,
        "texts_raw" : sents_raw,
        "texts_transformed_padded" : sents_transformed_padded,
        "texts_padding_masks" : sents_padding_masks
    }
    
    return batch