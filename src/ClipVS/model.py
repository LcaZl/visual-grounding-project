import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip

from src.ClipVS.weight_initialization import initialize_weights
from src.ClipVS.postprocessing import process_map, apply_percentile_threshold
from src.debug.debug_general import print_map

class FiLMBlock(nn.Module):
    """
    A Feature-wise Linear Modulation (FiLM) block that applies feature-wise 
    affine transformations to input features using modulation parameters.

    This block implements the FiLM equation:
        FiLM(F | gamma, beta) = gamma * F + beta
    where gamma and beta are modulation parameters that are broadcasted across 
    the spatial dimensions of the feature map.

    Parameters (Init and Forward)
    ----------
    x : torch.Tensor
        Input feature tensor of shape [B, C, H, W], where B is the batch size, 
        C is the number of channels and H and W are the spatial dimensions.
    gamma : torch.Tensor
        Scaling tensor of shape [B, C] broadcast to [B, C, 1, 1].
    beta : torch.Tensor
        Shifting tensor of shape [B, C] broadcast to [B, C, 1, 1].

    Returns
    -------
    torch.Tensor
        Modulated feature tensor of the same shape as the input.
    """
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x : torch.Tensor, gamma : torch.Tensor, beta : torch.Tensor) -> torch.Tensor:

        # Reshape gamma and beta to match the spatial dimensions of x
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        # Apply the FiLM transformation
        return gamma * x + beta
        
class ResBlock(nn.Module):
    """
    A residual block with FiLM-based feature modulation.

    This block integrates FiLM block into a residual structure to adaptively condition convolutional feature maps 
    based on textual embeddings.

    Parameters (Init and Forward)
    -----------------------------
    in_channels : int
        Number of input channels for the first convolutional layer.
    out_channels : int
        Number of output channels for the convolutions and FiLMBlock.
    x : torch.Tensor
        Input feature tensor of shape [B, in_channels, H, W].
    gamma : torch.Tensor
        Scaling tensor of shape [B, out_channels].
    beta : torch.Tensor
        Shifting tensor of shape [B, out_channels].

    Returns
    -------
    torch.Tensor
        Output tensor of shape [B, C_out, H, W], after residual addition.
    """
    def __init__(self, in_channels : int, out_channels : int):
        super(ResBlock, self).__init__()
        
        # 1x1 convolution to project input to out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 3x3 convolution for spatial features
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # FiLM modulation
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)
          
    def forward(self, x : torch.Tensor, gamma : torch.Tensor, beta : torch.Tensor) -> torch.Tensor:
        
        # Project input features
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x  # Save for residual addition

        # Apply spatial convolution and normalization
        x = self.conv2(x)

        # Apply FiLM modulation
        x = self.film(x, gamma, beta)
        x = self.relu2(x)

        # Add residual connection
        return x + identity

class MultiFiLM(nn.Module):
    """
    A multi-layer FiLM based feature modulation module for conditioning image features with textual embeddings.

    This module combines multiple FiLM-modulated residual blocks, where each  
    block applies feature-wise transformations using text-conditioned gamma and beta parameters.

    Parameters (Init and Forward)
    -----------------------------
    feature_dim : int
        Number of channels in the input features and output features.
    n_res_blocks : int
        Number of residual blocks to stack in the module.
    text_dim : int
        Dimensionality of the input text embeddings.
    features : torch.Tensor
        Input feature tensor of shape [B, feature_dim, H, W].
    text_embeddings : torch.Tensor
        Input text embeddings of shape [B, text_dim].

    Returns
    -------
    torch.Tensor
        Modulated feature tensor of shape [B, C, H, W].
    """
    def __init__(self, feature_dim : int, n_res_blocks : int, text_dim : int):
        super(MultiFiLM, self).__init__()

        self.film_generator = nn.Linear(text_dim, 2 * n_res_blocks * feature_dim)
        self.res_blocks = nn.ModuleList([ResBlock(feature_dim, feature_dim) for _ in range(n_res_blocks)])
        self.sigmoid = nn.Sigmoid()
        self.n_res_blocks = n_res_blocks
        self.feature_dim = feature_dim
        
    def forward(self, features : torch.Tensor, text_embeddings : torch.Tensor) -> torch.Tensor:
        
        # Generate gamma and beta from text embeddings
        film_vector = F.relu(self.film_generator(text_embeddings)).view(
            features.size(0), self.n_res_blocks, 2, self.feature_dim
        )

        gamma, beta = film_vector[:, :, 0, :], film_vector[:, :, 1, :]
            
        # Apply FiLM modulation in each ResBlock
        for i, res_block in enumerate(self.res_blocks):
            features = res_block(features, gamma[:, i, :], beta[:, i, :])
                    
        return self.sigmoid(features)

class BoundingBoxHead(nn.Module):
    
    """
    Bounding Box Head that combines attention masks and text embeddings to predict bounding boxes.

    This module integrates attention mask features and text embeddings through
    fully connected layers to predict the bounding box coordinates in [xmin, ymin, xmax, ymax] format.

    Parameters (Init and Forward)
    -----------------------------
    mask_dim : int
        Number of channels in the attention mask features (input size for mask-related processing).
    text_dim : int
        Dimension of the text embeddings (input size for text-related processing).
    hidden_dim : int, optional
        Hidden layer size for the multi-layer perceptron (MLP). Default is 512.
    dropout_prob : float, optional
        Dropout probability to be applied after each layer in the MLP. Default is 0.2.
    attention_masks : torch.Tensor
        Attention masks of shape [batch_size, mask_dim, H, W].
    text_embeddings : torch.Tensor
        Text embeddings of shape [batch_size, text_dim].

    Returns
    -------
    torch.Tensor
        Predicted bounding boxes of shape [batch_size, 4], where 4 represents the coordinates [xmin, ymin, xmax, ymax].
    """
    
    def __init__(self, mask_dim: int, text_dim: int, hidden_dim: int = 512, dropout_prob: float = 0.2):

        super(BoundingBoxHead, self).__init__()
        
        # Layers to process attention masks
        self.mask_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool attention masks to 1x1
        self.mask_proj = nn.Linear(mask_dim, hidden_dim)  # Project pooled masks
        
        # Layers to process text embeddings
        self.text_proj = nn.Linear(text_dim, hidden_dim)  # Project text embeddings
        
        # Final layers to predict bounding boxes
        self.bbox_mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),  # Dropout after combining features
            nn.Linear(hidden_dim * 2, hidden_dim),  # Combine text and mask features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),  # Dropout in hidden layers
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),  # Predict 4 bounding box coordinates
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, attention_masks: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        
        # Process attention masks
        pooled_masks = self.mask_pool(attention_masks).squeeze(-1).squeeze(-1)  # Shape: [batch_size, mask_dim]
        mask_features = self.mask_proj(pooled_masks)  # Shape: [batch_size, hidden_dim]

        # Process text embeddings
        text_features = self.text_proj(text_embeddings)  # Shape: [batch_size, hidden_dim]

        # Combine mask and text features
        combined_features = torch.cat([mask_features, text_features], dim=1)  # Shape: [batch_size, hidden_dim * 2]

        # Predict bounding boxes
        bboxes = self.bbox_mlp(combined_features)  # Shape: [batch_size, 4]
        return bboxes

class WeightedFeatureFusion(nn.Module):
    """
    Weighted Feature Fusion module for combining image features from different layers.

    This module integrates features extracted from multiple layers by applying 
    learnable weights to each and summing the weighted features.

    Parameters (Init and Forward)
    -----------------------------
    num_features : int
        The number of feature to be fused (number of layers or sources of features).
    feature_dim : int
        The channel dimension of each feature map.
    features : list[torch.Tensor]
        A list of feature to be fused. Each feature has shape [batch_size, feature_dim, H, W].

    Returns
    -------
    torch.Tensor
        The fused feature of shape [batch_size, feature_dim, H, W].
    """
    
    def __init__(self, num_features: int, feature_dim: int):
        super(WeightedFeatureFusion, self).__init__()
        
        # Learnable weights for each feature map
        self.weights = nn.Parameter(torch.ones(num_features))  # Initialized to 1 for equal weighting
        
        # 1x1 convolution to project fused features back to the original feature dimension
        self.projection = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        
        # Stack feature maps along a new dimension
        stacked_features = torch.stack(features, dim=1)  # Shape: [batch_size, num_features, feature_dim, H, W]
        
        # Apply learnable weights
        weighted_features = stacked_features * self.weights.view(1, -1, 1, 1, 1)  # Broadcasting weights
        
        # Sum the weighted features along the new dimension
        fused_features = weighted_features.sum(dim=1)  # Shape: [batch_size, feature_dim, H, W]
        
        # Apply the projection
        return self.projection(fused_features)


class ClipVS(torch.nn.Module):
    """
    Main model combining CLIP, FPN layers, FiLM-based fusion and a bounding box head to process multi-modal data.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the ClipVS model.

        Parameters
        ----------
        config : dict
            Configuration dictionary with the following keys:
            - "device": (str) Device to run the model on (e.g., "cuda" or "cpu").
            - "nn_fg_bg_percentile": (float) Percentile threshold for foreground-background masks.
            - "nn_film_layers": (int) Number of FiLM layers to use.
            - "verbose": (bool) Whether to enable verbose output.
            
        """        
        super().__init__()

        # Device and configuration settings        
        self.device = config["device"]
        self.bg_threshold = config["nn_fg_bg_percentile"]
        self.film_num_layers = config["nn_film_layers"]
        self.verbose = config["verbose"]

        # Load CLIP backbone
        
        self.clip_model, self.image_preprocess = clip.load("RN50", device=config["device"])
        self.clip_model = self.clip_model.float()
        self.sent_preprocessing = clip.tokenize

        # FPN Layers for multi-scale feature extraction
        
        self.fpn_l1 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(56, 56))
        )

        self.fpn_l2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(56, 56))
        )

        self.fpn_l3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(56, 56))
        )
                
        # Layer to combine fpns
        
        self.fusion = WeightedFeatureFusion(num_features=3, feature_dim=1024)
        
        # FiLM Layer for visual-textual feature fusion
        
        self.film_layer = MultiFiLM(feature_dim=1024, n_res_blocks=self.film_num_layers, text_dim=1024)

        # Head for bounding box prediction
        
        self.bbox_head = BoundingBoxHead(mask_dim=1024, 
                                         text_dim = 1024, 
                                         hidden_dim=1024,
                                         dropout_prob=config["nn_dropout_prob"])
               
        for name, param in self.clip_model.named_parameters():
                param.requires_grad = False  # Freeze all layers

        # Weights initialization

        self.fpn_l1.apply(initialize_weights)
        self.fpn_l2.apply(initialize_weights)
        self.fpn_l3.apply(initialize_weights)
        self.fusion.apply(initialize_weights)
        self.film_layer.apply(initialize_weights)
        self.bbox_head.apply(initialize_weights)
            
    def encode_sentences_mean(self, tokenized_sents : torch.Tensor, sent_mask : torch.Tensor) ->  torch.Tensor:
        """
        Encode tokenized sentences and compute their mean embedding.

        Parameters
        ----------
        tokenized_sents : Tensor
            Tensor of shape [batch_size, max_num_sents, seq_len] containing tokenized sentences.
        sent_mask : Tensor
            Mask of shape [batch_size, max_num_sents] with 1 for valid tokens and 0 for padding.

        Returns
        -------
        Tensor
            Mean embedding of shape [batch_size, embed_dim] for valid tokens in each batch element.
        """
        texts_z_views = []
        
        # Iterate over each image's tokenized sentences and mask
        for tokenized_sents_per_image, mask in zip(tokenized_sents, sent_mask):
            
            with torch.no_grad():
                texts_z = self.clip_model.encode_text(tokenized_sents_per_image).float()  # Shape: [num_sents, 1024]
            texts_z /= texts_z.norm(dim=-1, keepdim=True)  # Normalize embeddings

            # Mask out padded sentences
            valid_embeddings = texts_z * mask.unsqueeze(-1)  # Shape: [num_sents, 1024]

            # Compute the mean of valid embeddings, ignoring padding
            mean_text_embedding = valid_embeddings.sum(dim=0) / mask.sum()  # Shape: [1024]

            texts_z_views.append(mean_text_embedding)
        
        # Stack mean embeddings for the batch
        texts_z = torch.stack(texts_z_views)  # Shape: [batch_size, 1024]

        # Renormalize the final embeddings
        texts_z /= texts_z.norm(dim=-1, keepdim=True)

        return texts_z

    def encode_sentences(self, tokenized_sents: torch.Tensor, sent_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode tokenized sentences, retaining embeddings for all tokens.

        Parameters
        ----------
        tokenized_sents : Tensor
            Tensor of shape [batch_size, max_num_sents, seq_len] containing tokenized sentences.
        sent_mask : Tensor
            Mask of shape [batch_size, max_num_sents] with 1 for valid tokens and 0 for padding.

        Returns
        -------
        Tensor
            Encoded sentences of shape [batch_size, max_num_sents, embed_dim].
        """
        texts_z_views = []

        # Iterate over each image's tokenized sentences and mask
        for tokenized_sents_per_image, mask in zip(tokenized_sents, sent_mask):
            
            with torch.no_grad():
                texts_z = self.clip_model.encode_text(tokenized_sents_per_image).float()  # Shape: [num_sents, 1024]
            texts_z /= texts_z.norm(dim=-1, keepdim=True)  # Normalize embeddings

            # Mask out padded sentences
            sentence_embeddings = texts_z * mask.unsqueeze(-1)  # Shape: [num_sents, 1024]

            texts_z_views.append(sentence_embeddings)

        # Stack sentence embeddings for the batch
        texts_z = torch.stack(texts_z_views) # Shape: [batch_size, max_num_sents, 1024]

        return texts_z

    def encode_images_std(self, images: list) -> torch.Tensor:
        """
        Encode and normalize a batch of images using the CLIP visual encoder.

        Parameters
        ----------
        images : list
            Batch of preprocessed images.

        Returns
        -------
        Tensor
            Normalized image embeddings of shape [batch_size, embed_dim].
        """
        with torch.no_grad():
            images_z = self.clip_model.encode_image(images).float()
            
        normalized_images_z = images_z / images_z.norm(dim=-1, keepdim=True)
        
        return normalized_images_z
    
    def encode_images(self, x: list) -> list[torch.Tensor]:
        """
        Extract multi-scale features from CLIP.

        Parameters
        ----------
        x : list
            Input images after passing through initial CLIP layers.

        Returns
        -------
        list[Tensor]
            Multi-scale features from different FPN layers.
        """
        with torch.no_grad():
            x = self.clip_model.visual.conv1(x)
            x = self.clip_model.visual.bn1(x)
            x = self.clip_model.visual.relu1(x)
            x = self.clip_model.visual.conv2(x)
            x = self.clip_model.visual.bn2(x)
            x = self.clip_model.visual.relu2(x)
            x = self.clip_model.visual.conv3(x)
            x = self.clip_model.visual.bn3(x)
            x = self.clip_model.visual.relu3(x)
            x = self.clip_model.visual.avgpool(x)
            l1 = self.clip_model.visual.layer1(x) # shape [batch_size, 256, 56, 56]
            l2 = self.clip_model.visual.layer2(l1) # shape [batch_size, 512, 28, 28]
            l3 = self.clip_model.visual.layer3(l2) # shape [batch_size, 1024, 14, 14]

        return [l1, l2, l3]
    
    def forward(self, batch: dict) -> dict:
        """
        Forward pass to process multi-modal inputs and predict bounding boxes.

        Parameters
        ----------
        batch : dict
            Dictionary containing batched data:
            - "texts_transformed_padded": Tensor of padded tokenized sentences.
            - "texts_padding_masks": Tensor mask for valid text tokens.
            - "imgs_transformed": Tensor of preprocessed images.
            - "imgs_dims": List of image dimensions.

        Returns
        -------
        dict
            Dictionary with intermediate features, attention masks, embeddings and bounding box predictions.
        """
        try:
            
            texts = batch["texts_transformed_padded"]
            texts_masks = batch["texts_padding_masks"]
            images = batch["imgs_transformed"]
            
            texts_z = self.encode_sentences(texts, texts_masks)  # Shape: [batch_size, num_sents, 1024]
            texts_z_mean = self.encode_sentences_mean(texts, texts_masks)
            fpn_features = self.encode_images(images)  # Shape: [batch_size, 1024, H, W]
            imgs_embeddings = self.encode_images_std(images)
                       
            # Project FPN features to common dimension
            fpn_l1 = self.fpn_l1(fpn_features[0])
            fpn_l2 = self.fpn_l2(fpn_features[1])
            fpn_l3 = self.fpn_l3(fpn_features[2])
            
            # Combine FPN features by summing
            fused_features = self.fusion([fpn_l1, fpn_l2, fpn_l3])  # Shape: [batch_size, 1024, 56, 56]
            attentions_masks = self.film_layer(fused_features, texts_z_mean)
            
            attention_masks_resized = process_map(attentions_masks,  batch["imgs_dims"])
            foregrounds, backgrounds = apply_percentile_threshold(attention_masks_resized, self.bg_threshold)

            bboxes = self.bbox_head(attentions_masks, texts_z_mean)
            
            return {
                
                "imgs_embeddings" : imgs_embeddings,
                "fpn_imgs_l1" : fpn_l1,
                "fpn_imgs_l2" : fpn_l2,
                "fpn_imgs_l3" : fpn_l3,
                "fpn_imgs_l1_raw" : fpn_features[0],
                "fpn_imgs_l2_raw" : fpn_features[1],
                "fpn_imgs_l3_raw" : fpn_features[2],
                "fused_features" : fused_features,
                "attention_masks" : attentions_masks,
                "attention_masks_resized" : attention_masks_resized,
                "foreground_masks" : foregrounds,
                "background_masks" : backgrounds,
                "proposals" : [],
                "texts_embeddings" : texts_z,
                "texts_embeddings_mean" : texts_z_mean,
                "pred_bboxes" : bboxes
            }

        except Exception as e:
            
            error_vars = {
                'texts': texts.shape if 'texts' in locals() else 'NC',
                'texts_masks': texts_masks.shape if 'texts_masks' in locals() else 'NC',
                'images': images.shape if 'images' in locals() else 'NC',
                'texts_z': texts_z.shape if 'texts_z' in locals() else 'NC',
                'texts_z_mean': texts_z_mean.shape if 'texts_z_mean' in locals() else 'NC',
                'fpn_features': [f.shape for f in fpn_features] if 'fpn_features' in locals() else 'NC',
                'imgs_embeddings': imgs_embeddings.shape if 'imgs_embeddings' in locals() else 'NC',
                'fpn_l1': fpn_l1.shape if 'fpn_l1' in locals() else 'NC',
                'fpn_l2': fpn_l2.shape if 'fpn_l2' in locals() else 'NC',
                'fpn_l3': fpn_l3.shape if 'fpn_l3' in locals() else 'NC',
                'fused_features': fused_features.shape if 'fused_features' in locals() else 'NC',
                'attentions_masks': attentions_masks.shape if 'attentions_masks' in locals() else 'NC',
                'attention_masks_resized': [m.shape for m in attention_masks_resized] if 'attention_masks_resized' in locals() else 'NC',
                'foregrounds': [f.shape for f in foregrounds] if 'foregrounds' in locals() else 'NC',
                'backgrounds': [b.shape for b in backgrounds] if 'backgrounds' in locals() else 'NC',
                'bboxes': bboxes.shape if 'bboxes' in locals() else 'NC'
            }
            print(f"Error: {e} - Internal state:\n")
            print_map(error_vars)
            raise e

        

        
        