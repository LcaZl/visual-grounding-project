import torch
import pandas as pd
import torchvision
import PIL

# https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair dryer', 'toothbrush'
]

class baselineCLIP(torch.nn.Module):
    """
    A baseline model that integrates CLIP with external object detection
    models for visual grounding task.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing settings.
    clip_model : Any
        Pretrained CLIP model instance for generating text and image embeddings.
    image_preprocess : callable
        Clip image preprocessing function.
    sents_tokenizer : callable
        Clip tokenizer for sentences.
    yolo_model : object
        YOLO model.
    frcnn_model : object
        Faster R-CNN model.
    ssd_model : object
        Single Shot Detector model.
    verbose : bool, optional
        Whether to print additional information during operations, by default False.
    """
    
    def __init__(self, conf: dict, clip_model: object, image_preprocess: callable, sents_tokenizer: callable, 
                 yolo_model : object, frcnn_model : object, ssd_model : object, verbose: bool = False) -> None:        
        super().__init__()
        
        # CLIP-related components
        self.clip_model = clip_model
        self.image_preprocessing = image_preprocess
        self.sent_preprocessing = sents_tokenizer
        self.sents_vector_type = conf["bs_sents_vector_type"]
        self.verbose = verbose
        self.device = conf["device"]
        
        # External object detection model selection
        self.model = conf["bs_supp_model"]
        
        if self.model == "Yolo":
            self.ext_model = yolo_model
            self.find_objects = self.use_yolo
            
        elif self.model == "FRCNN":
            self.ext_model = frcnn_model
            self.find_objects = self.use_frcnn

        elif self.model == "SSD":
            self.ext_model = ssd_model
            self.find_objects = self.use_ssd
            
        else:
            raise ValueError(f"Specified model ({conf['nn_supp_model']}) not supported")

        # Sentence encoding method selection
        if self.sents_vector_type == "Mean":
            self.encode_sents = self.encode_sentences_mean
        elif self.sents_vector_type == "NoMean":
            self.encode_sents = self.encode_sentences
        else:
            raise ValueError(f"Unsupported sentence vector type: {self.sents_vector_type}")
                    
    def extract_bbox_images(self, image: PIL.Image.Image, model_output: pd.DataFrame) -> list[PIL.Image.Image]:
        """
        Extract cropped images based on bounding box coordinates from object detection models output.

        Parameters
        ----------
        image : PIL.Image.Image
            The original image from which to crop bounding boxes.
        yolo_output : pd.DataFrame
            DataFrame containing bounding box coordinates with columns ['xmin', 'ymin', 'xmax', 'ymax'].

        Returns
        -------
        list[PIL.Image.Image]
            List of cropped images corresponding to the bounding boxes.
        """
        cropped_images = []
        for _, row in model_output.iterrows():
            x_min, y_min, x_max, y_max = row[['xmin', 'ymin', 'xmax', 'ymax']]
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_images.append(cropped_image)
        return cropped_images
      
    def encode_sentences_mean(self, sents: torch.Tensor) -> torch.Tensor:
        """
        Encode a list of sentences into a single mean embedding using the CLIP model.

        Parameters
        ----------
        sents : torch.Tensor
            Tensor of tokenized sentences with shape [num_sents, 77].

        Returns
        -------
        torch.Tensor
            A normalized tensor representing the mean embedding of all input sentences.
        """
        
        with torch.no_grad():
            texts_z = self.clip_model.encode_text(sents).float()
        texts_z /= texts_z.norm(dim=-1, keepdim=True)

        # Compute the mean embedding
        texts_z_mean = texts_z.mean(dim=0)

        # Renormalize the mean embedding
        texts_z_mean /= texts_z_mean.norm(dim=-1, keepdim=True)

        return texts_z_mean.unsqueeze(dim=0)

    def encode_sentences(self, sents: list[str]) -> torch.Tensor:
        """
        Encode a list of sentences into individual embeddings using the CLIP model.

        Parameters
        ----------
        sents : list[str]
            List of sentences to encode.

        Returns
        -------
        torch.Tensor
            A tensor containing normalized embeddings for each input sentence.
        """
        
        with torch.no_grad():
            texts_z = self.clip_model.encode_text(sents).float()
            
        texts_z /= texts_z.norm(dim=-1, keepdim=True)
        
        return texts_z
  
    def encode_images(self, cropped_images: list[PIL.Image.Image]) -> torch.Tensor:
        """
        Encode a list of cropped images into embeddings using the CLIP model.

        Parameters
        ----------
        cropped_images : list[PIL.Image]
            List of cropped images to encode.

        Returns
        -------
        torch.Tensor
            A tensor containing embeddings for each input image.
        """
        images = torch.stack([self.image_preprocessing(image) for image in cropped_images]).to(self.device)
        with torch.no_grad():
            images_z = self.clip_model.encode_image(images).float()
        return images_z
  
    def use_yolo(self, image: PIL.Image.Image, conf_threshold: float = 0.5) -> tuple:
        """
        Perform object detection using YOLO and filter results by confidence threshold.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image for object detection.
        conf_threshold : float, optional
            Confidence threshold for filtering detected objects, by default 0.5.

        Returns
        -------
        tuple
            - output : torch.Tensor
                Raw YOLO model output.
            - output_df : pd.DataFrame
                Filtered DataFrame of detected bounding boxes with columns:
                ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'].
        """
    
        output = self.ext_model(image)
        output_df = output.pandas().xyxy[0]
        output_df = output_df[output_df['confidence'] >= conf_threshold].reset_index(drop=True)

        return output, output_df
        
    def use_frcnn(self, image: PIL.Image.Image, conf_threshold: float = 0.5) -> tuple:
        """
        Perform object detection using Faster R-CNN and filter results by confidence threshold.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image for object detection.
        conf_threshold : float, optional
            Confidence threshold for filtering detected objects, by default 0.5.

        Returns
        -------
        tuple
            - output : dict
                Raw Faster R-CNN model output.
            - output_df : pd.DataFrame
                Filtered DataFrame of detected bounding boxes with columns:
                ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'].
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        image_tensor = transform(image).to(self.device)
        with torch.no_grad():
            output = self.ext_model([image_tensor])[0]
            
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        output_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        output_df['confidence'] = scores
        output_df['class'] = labels
        output_df['name'] = output_df['class'].apply(lambda x: COCO_INSTANCE_CATEGORY_NAMES[x] if x < len(COCO_INSTANCE_CATEGORY_NAMES) else "NotAvailable")
        output_df = output_df[output_df['confidence'] >= conf_threshold].reset_index(drop=True)
        
        return output, output_df
    
    def use_ssd(self, image: PIL.Image.Image, conf_threshold: float = 0.5) -> tuple:
        """
        Perform object detection using SSD and filter results by confidence threshold.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image for object detection.
        conf_threshold : float, optional
            Confidence threshold for filtering detected objects, by default 0.5.

        Returns
        -------
        tuple
            - output : dict
                Raw SSD model output.
            - output_df : pd.DataFrame
                Filtered DataFrame of detected bounding boxes with columns:
                ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'].
        """
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        image_tensor = transform(image).to(self.device)
        with torch.no_grad():
            output = self.ext_model([image_tensor])[0]
            
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        output_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        output_df['confidence'] = scores
        output_df['class'] = labels
        output_df['name'] = output_df['class'].apply(lambda x: COCO_INSTANCE_CATEGORY_NAMES[x] if x < len(COCO_INSTANCE_CATEGORY_NAMES) else "NotAvailable")
        output_df = output_df[output_df['confidence'] >= conf_threshold].reset_index(drop=True)
        
        return output, output_df

        
    def forward(self, sample: dict) -> dict:
        """
        Perform a forward pass of the model on a given sample to predict bounding boxes and similarities.

        Parameters
        ----------
        sample : dict
            A sample containing the following keys:
            - 'img_raw': PIL.Image.Image
                The raw image.
            - 'sents_tokenized': list[str]
                List of prepared sentences describing the image.

        Returns
        -------
        dict
            Dictionary containing:
            - 'pred_bbox': list[float] | None
                Predicted bounding box [xmin, ymin, xmax, ymax], or None if no bounding box is selected.
            - 'similarity': torch.Tensor | None
                Matrix of similarities between text and cropped images.
            - 'cropped_images': list[PIL.Image.Image]
                List of cropped images based on detected bounding boxes.
            - 'text_prob': torch.Tensor | None
                Softmax-normalized probabilities for each sentence.
            - 'ext_output_df': pd.DataFrame
                DataFrame of bounding box predictions from the external model.
        """
        pred_bbox = None
        similarity = None
        cropped_images = []
        texts_p = None
              
        # Retrieve bounding boxes
        _, ext_output_df = self.find_objects(sample['img_raw'])
        
        if len(ext_output_df) > 0:
            
            # Extract an image for each bounding box
            cropped_images = self.extract_bbox_images(sample['img_raw'], ext_output_df)        
            images_z = self.encode_images(cropped_images).to(self.device)
            texts_z = self.encode_sents(sample['sents_tokenized'])
            
            # CLIP one-shot cosine similarity matrix
            similarity = (texts_z @ images_z.T)

            texts_p = (100 * similarity).softmax(dim=-1)
            mean_probs = texts_p.mean(dim=0)
            sel_crop = mean_probs.argmax().item()
            pred_bbox = list(ext_output_df.loc[sel_crop, ['xmin','ymin','xmax','ymax']].values)
        
            pred_bbox = [float(coord) for coord in pred_bbox]
        
        return {
            "pred_bbox" : pred_bbox,
            "similarity" : similarity,
            "cropped_images" :  cropped_images,
            "text_prob" : texts_p,
            "ext_output_df" : ext_output_df
        }