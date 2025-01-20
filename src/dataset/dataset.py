import pickle
import json
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.dataset.similarity_matrix import SimilarityMatrix
from src.utils.utils import (
    get_clip_embeddings, 
    get_clip_mean_embeddings, 
    apply_templates, 
    polygons_to_binary_mask
)

class refcocog(Dataset):
    """
    Dataset class for loading and processing the RefCOCOg dataset (split by UMD).
    """
    
    def __init__(self, conf: dict, split : str, sents_preprocess : callable, images_preprocess : callable, clip_model : object) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        conf : dict
            Configuration dictionary containing dataset paths, preprocessing methods and other settings.
        split : str
            Dataset split to load ('train', 'val', 'test').
        sents_preprocess : callable
            Function for preprocessing sentences (Clip tokenization).
        images_preprocess : callable
            Function for preprocessing images (Clip image preprocessing).
        clip_model : Any
            CLIP model instance for generating sentence embeddings.
        """
        
        # Initialize dataset properties
        self.dataset_name = 'refcocog'
        self.splitBy = "umd" # Google or UMD
        self.split = split
        self.image_dir = os.path.join(conf["dataset_path"], 'images')
        self.sents_preprocess = sents_preprocess
        self.images_preprocess = images_preprocess # Images transformation
        self.sents_vector_type = conf["dt_sents_vector_type"]
        self.templates = conf["dt_templates"]
        self.apply_templates = conf["dt_apply_template"]
        self.sample_limit = conf["dt_samples_limit"]
        self.enh_sents = conf["dt_extra_similar_sents"]
        self.device = conf["device"]
        self.clip_model = clip_model
        
        print("=" * 50)
        print(f'Loading dataset {self.dataset_name} for {split}...')

        # Load references & instances
        ref_file_path = os.path.join(conf["dataset_path"], f'annotations/refs({self.splitBy}).p')
        with open(ref_file_path, 'rb') as file:
            refs = pickle.load(file)
            
        instances_file_path = os.path.join(conf["dataset_path"], 'annotations/instances.json')
        with open(instances_file_path, 'r') as file:
            self.instances = json.load(file)
        
        self.info = self.instances['info']
        self.licenses = self.instances['licenses']
    
        # Create DataFrames for easy merging and manipulation
        self.refs = pd.DataFrame(refs)
        self.annotations = pd.DataFrame(self.instances['annotations'])
        self.images = pd.DataFrame(self.instances['images'])
        self.categories = pd.DataFrame(self.instances['categories'])

        # Construct dataset samples by merging relevant dataframes
        self.samples = self.refs[self.refs['split'] == self.split] \
        .merge(self.annotations[['id','area','image_id','bbox','category_id','segmentation']], how='left', left_on='ann_id', right_on='id').drop(columns=['id','sentences']) \
        .merge(self.categories, how='left', left_on='category_id_x', right_on='id').drop(columns='id') \
        .merge(self.images, how='left', left_on='image_id_x', right_on='id').drop(columns=['id'])
        
        # Perform sanity checks on merged data
        self.sanity_check() 
        
        # Apply sample limit if specified
        # Validation dataset is never limited
        if (self.sample_limit is not None and self.split == 'val' and not conf["dt_full_val_set"]) or ((self.sample_limit is not None and self.split != 'test') or \
        (self.sample_limit is not None and self.split == 'test' and not conf["dt_full_test_set"])):
            
            limit_count = max(int(len(self.samples) * (self.sample_limit / 100)), 1)
            print(f"Loading Refcocog ({self.split}) using {limit_count}/{len(self.samples)} samples.")
            self.samples = self.samples[:min(limit_count, len(self.samples))]
            
        else:
            print(f"Full loading of Refcocog {self.split} dataset.")      
                   
        # Extract sentences data
        sample_ids = set([id for sent_ids in self.samples['sent_ids'] for id in sent_ids ])
        self.sentences = pd.DataFrame([el for row in self.refs.itertuples() for el in row.sentences if el['sent_id'] in sample_ids]).set_index('sent_id')

        # Initialize similarity matrix if specified
        self.sm = None
        if self.enh_sents:
            self.sm = SimilarityMatrix(conf, self.sentences, self.split, self.clip_model)
            
        print(f'{self.dataset_name} Dataset loaded with {len(self.samples)} samples!')
        print("=" * 50)

    def reload_similarity_matrix(self, conf : dict) -> None:
        """
        Reload the similarity matrix and related parameter.
        Used within grid search when at a certain step a different number of similar sentences is required.
        The matrix will not be recomputed, it is reloaded with the requested different number of top-k precomputed elements.
        
        Parameters
        ----------
        conf : dict
            Updated configuration dictionary.
        """
        self.enh_sents = conf["dt_extra_similar_sents"]
        if self.enh_sents:
            self.sm = SimilarityMatrix(conf, self.sentences, self.split, self.clip_model)
        else:
            self.sm = None
            
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            Dictionary containing the following components:
            - id : int
                Unique identifier of the sample.
            - gt_bbox : list[float]
                Ground truth bounding box for the object in the image, represented as [x_min, y_min, width, height].
            - gt_segmentation : list
                Ground truth segmentation polygon coordinates for the object.
            - gt_segmentation_mask : np.ndarray
                Binary mask of the ground truth segmentation, with the object area filled with 1s.
            - gt_area : float
                Area of the object as specified in the annotations.
            - img_dim : tuple[int, int]
                Dimensions of the original image as (width, height).
            - img_raw : PIL.Image.Image
                The original unprocessed image.
            - img_transformed : torch.Tensor
                Clip preprocessed image.
            - sents_raw : list[str]
                List of all sentences describing the object, including any added templates or extra sentences from similarity matrix.
            - sents_tokenized : torch.Tensor
                Tokenized sentences.
            - sents_embeddings : torch.Tensor
                Embeddings of the sentences generated by the CLIP model. Used for debug.
            - sents_extra : list[str] | None
                Additional sentences retrieved using the similarity matrix, if enabled. Used for debug.
            - gt_category : str
                Name of the object category. Not used.
            - sample_df : pandas.Series
                Row from the dataset DataFrame corresponding to this sample. Used for debug.
            - sents_df : pandas.DataFrame
                DataFrame containing all sentences associated with this sample. Used for debug.
            - sents_extra_df : pandas.DataFrame | None
                DataFrame of additional sentences retrieved from the similarity matrix, if enabled. Used for debug.
        """
        if self.enh_sents and self.sm is None:
            raise ValueError("Requested enhanced sentences but similarity matrix is not available.")

        # Retrieve the sample information from the DataFrame
        sample = self.samples.loc[idx]
        img_dim = (sample['width'], sample['height'])  # Original image dimensions
        sent_ids = set(sample['sent_ids'])  # Sentence IDs associated with the sample

        # Add extra sentences based on the similarity matrix
        if self.enh_sents:
            sents_extra_topk_ids = set()
            for p_id in list(sent_ids):
                sents_extra_topk_ids.update(self.sm.get_top_k_similar(p_id))  # Get top-k similar sentences
            
            # Filter and retrieve the extra sentences
            sents_extra_df = self.sentences.loc[[el for el in sents_extra_topk_ids if el not in sent_ids]]
            
            if self.apply_templates:
                sents_extra = apply_templates(sents_extra_df['sent'].values, self.templates)  # Apply templates to extra sentences
            else:
                sents_extra = sents_extra_df['sent'].values

            sent_ids.update(sents_extra_topk_ids)  # Combine extra sentences with original ones

        # Retrieve all sentences associated with this sample
        sents_df = self.sentences.loc[list(sent_ids)]
        if self.apply_templates:
            sents = apply_templates(sents_df['sent'].values, self.templates)  # Apply templates if enabled
        else:
            sents = sents_df['sent'].values

        # Load annotations and image data
        bbox = sample['bbox']  # Ground truth bounding box
        area = sample['area']  # Object area
        segmentation = sample['segmentation']  # Ground truth segmentation
        segmentation_mask = polygons_to_binary_mask(segmentation, img_dim)  # Convert segmentation to binary mask
        image_raw = Image.open(os.path.join(self.image_dir, sample['file_name']))  # Load raw image

        # Preprocess sentences and image
        sents_tokenized = self.sents_preprocess(sents).to(self.device)  # Tokenize sentences with Clip
        image_transformed = self.images_preprocess(image_raw)  # Apply Clip image transformations

        # Generate sentence embeddings
        if self.sents_vector_type == "Mean":
            sents_embeddings = get_clip_mean_embeddings(sents_tokenized, self.clip_model)  # Compute mean embedding
        elif self.sents_vector_type == "NoMean":
            sents_embeddings = get_clip_embeddings(sents_tokenized, self.clip_model)  # Compute individual embeddings

        # Construct the output dictionary
        sample = {
            'id': sample['ref_id'],  # Reference ID
            'gt_bbox': bbox,  # Ground truth bounding box
            'gt_segmentation': segmentation,  # Ground truth segmentation
            'gt_segmentation_mask': segmentation_mask,  # Binary mask of segmentation
            'gt_area': area,  # Ground truth area
            'img_dim': img_dim,  # Image dimensions
            'img_raw': image_raw,  # Original image
            'img_transformed': image_transformed,  # Preprocessed image
            'sents_raw': sents,  # Raw sentences
            'sents_tokenized': sents_tokenized,  # Tokenized sentences
            'sents_embeddings': sents_embeddings,  # Sentence embeddings
            'gt_category': sample['name'],  # Ground truth category name
            'sample_df': sample,  # DataFrame row for the sample
            'sents_df': sents_df,  # DataFrame of associated sentences
            'sents_extra': sents_extra if self.enh_sents else None,  # Additional sentences (if enabled)
            'sents_extra_df': sents_extra_df if self.enh_sents else None  # DataFrame of additional sentences (if enabled)
        }

        return sample


    def sanity_check(self) -> None:
        """
        Perform sanity checks on the dataset to ensure integrity of merged fields.
        """
        print(f'Sanity check ...')

        # Check and unify image_id fields
        assert (self.samples['image_id_x'] == self.samples['image_id_y']).all(), "image_id_x and image_id_y don't match for some rows."
        self.samples['image_id'] = self.samples['image_id_x']  # Retain image_id_x and rename
        self.samples.drop(['image_id_x', 'image_id_y'], axis=1, inplace=True)

        # Check and unify category_id fields
        assert (self.samples['category_id_x'] == self.samples['category_id_y']).all(), "category_id_x and category_id_y don't match for some rows."
        self.samples['category_id'] = self.samples['category_id_x']
        self.samples.drop(['category_id_x', 'category_id_y'], axis=1, inplace=True)

        # Ensure ann_id matches file_name_x and clean redundant columns
        assert self.samples.apply(lambda row: str(row['ann_id']) in row['file_name_x'], axis=1).all(), "ann_id does not match file_name_x for some rows."
        self.samples.drop(['file_name_x'], axis=1, inplace=True)
        self.samples.rename(columns={'file_name_y': 'file_name'}, inplace=True)