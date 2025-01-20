import torch.nn as nn

def initialize_weights(layer) -> None:
    """
    Initializes the weights of layers of ClipVS using different initialization method.

    Parameters
    ----------
    m : nn.Module
        A module from the neural network to be initialized. This function is designed
        to handle two kind of layers: 'Conv2d' and 'Linear'.

    Initialization Details
    -----------------------
    - 'nn.Conv2d':
        - Weight: Kaiming (He) uniform initialization, which is suitable layers
          followed by ReLU activation functions to prevent vanishing/exploding gradients.
        - Bias: Constant initialization to 0 (if bias exists).
    - 'nn.Linear':
        - Weight: Xavier (Glorot) uniform initialization, good for fully connected layers.
        - Bias: Constant initialization to 0 (if bias exists).

    """
    
    if isinstance(layer, nn.Conv2d):
        
        # He initialization for convolutional layers
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
            
    elif isinstance(layer, nn.Linear):
        
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.01)