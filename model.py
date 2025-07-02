from torch import nn
import torch
import torch.nn.functional as F

# Enhanced CNN with modern improvements
class EnhancedCNN(nn.Module):
    def __init__(self, input_length, num_classes=1):
        super(EnhancedCNN, self).__init__()
        
        # Convolutional blocks with batch normalization and pooling
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  # Reduces sequence length by half
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  # Reduces sequence length by half again
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling - reduces to (batch, 128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ResNet-style CNN with residual connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class ResCNN(nn.Module):
    def __init__(self, input_length, num_classes=1):
        super(ResCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else nn.Identity()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

# Attention mechanism for sequence data
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads=8):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, x):
        N, seq_length, _ = x.shape
        
        # Split into multiple heads
        x = x.reshape(N, seq_length, self.heads, self.head_dim)
        
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, seq_length, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

class CNNWithSelfAttention(nn.Module):
    def __init__(self, input_length, num_classes=1):
        super(CNNWithSelfAttention, self).__init__()
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Self-attention
        self.attention = SelfAttention(embed_size=256, heads=8)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )
        
    def forward(self, x):
        # CNN features
        x = self.conv_layers(x)  # (batch, 256, seq_len)
        
        # Prepare for attention: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # Apply self-attention
        x = self.attention(x)
        
        # Back to (batch, features, seq_len) for pooling
        x = x.permute(0, 2, 1)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x

# Original SimpleCNN class remains the same
class SimpleCNN(nn.Module):
    def __init__(self, input_length):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.5)  # Dropout layer

        self.fc1 = nn.Linear(128 * input_length, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 1)  # Binary classification

        # Sigmoid activation for output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional layers followed by ReLU
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten the output from CNN layers
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers

        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        # x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # Sigmoid activation for binary classification
        x = self.sigmoid(x)
        return x

# Model factory function
def create_model(model_type="enhanced", input_length=100, num_classes=1, **kwargs):
    """
    Factory function to create different model types
    
    Args:
        model_type (str): Type of model ('simple', 'enhanced', 'resnet', 'attention')
        input_length (int): Length of input sequences
        num_classes (int): Number of output classes
        **kwargs: Additional arguments for specific models
    
    Returns:
        torch.nn.Module: The requested model
    """
    if model_type == "simple":
        return SimpleCNN(input_length)
    elif model_type == "enhanced":
        return EnhancedCNN(input_length, num_classes)
    elif model_type == "resnet":
        return ResCNN(input_length, num_classes)
    elif model_type == "attention":
        return CNNWithSelfAttention(input_length, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Model initialization function
def init_weights(m):
    """Initialize model weights using Xavier/He initialization"""
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Example usage and model comparison
if __name__ == "__main__":
    # Example parameters
    batch_size = 32
    input_length = 100
    num_features = 1
    
    # Create sample input
    x = torch.randn(batch_size, num_features, input_length)
    
    # Test different models
    models = {
        "Simple CNN": SimpleCNN(input_length),
        "Enhanced CNN": EnhancedCNN(input_length),
        "ResNet CNN": ResCNN(input_length),
        "CNN with Attention": CNNWithSelfAttention(input_length)
    }
    
    print("Model Comparison:")
    print("-" * 50)
    
    for name, model in models.items():
        # Initialize weights
        model.apply(init_weights)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output shape: {output.shape}")
        print()
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Scaled Dot-Product Attention Function
# def scaled_dot_product_attention(Q, K, V, mask=None):
#     d_k = Q.size(-1)
#     scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, float('-inf'))
    
#     attention_weights = F.softmax(scores, dim=-1)
#     output = torch.matmul(attention_weights, V)
#     return output, attention_weights

# # Self-Attention Module
# class SelfAttention(nn.Module):
#     def __init__(self, embed_size):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(embed_size, embed_size)
#         self.key = nn.Linear(embed_size, embed_size)
#         self.value = nn.Linear(embed_size, embed_size)

#     def forward(self, x, mask=None):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         out, _ = scaled_dot_product_attention(Q, K, V, mask)
#         return out

# # CNN Model with Self-Attention
# class CNNWithAttention(nn.Module):
#     def __init__(self, input_length):
#         super(CNNWithAttention, self).__init__()

#         # Convolutional layers
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

#         self.dropout = nn.Dropout(0.2)  

#         # Self-Attention Layer (embed_size should match last Conv1d output channels)
#         self.attention = SelfAttention(embed_size=128)

#         # Fully connected layers
#         self.fc1 = nn.Linear(128 * input_length, 128)  
#         self.fc2 = nn.Linear(128, 1)  

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))  # Shape: (batch, 128, seq_length)

#         # Reshape for Self-Attention
#         x = x.permute(0, 2, 1)  # Change shape to (batch, seq_length, 128)
#         x = self.attention(x)  # Apply self-attention

#         # Flatten before FC layers
#         x = x.view(x.size(0), -1)  
#         x = self.dropout(torch.relu(self.fc1(x)))
#         x = self.fc2(x)

#         return self.sigmoid(x)

# import torch  
# import torch.nn as nn  
# import torch.nn.functional as F  
# from typing import Optional, Tuple  

# def scaled_dot_product_attention(  
#     Q: torch.Tensor,   
#     K: torch.Tensor,   
#     V: torch.Tensor,   
#     mask: Optional[torch.Tensor] = None  
# ) -> Tuple[torch.Tensor, torch.Tensor]:  
#     """  
#     Compute scaled dot-product attention.  
    
#     Args:  
#         Q (torch.Tensor): Query tensor  
#         K (torch.Tensor): Key tensor  
#         V (torch.Tensor): Value tensor  
#         mask (Optional[torch.Tensor]): Attention mask  
    
#     Returns:  
#         Tuple of output tensor and attention weights  
#     """  
#     # Validate input dimensions  
#     assert Q.dim() == K.dim() == V.dim(), "Tensor dimensions must match"  
    
#     d_k = Q.size(-1)  
#     scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))  
    
#     if mask is not None:  
#         scores = scores.masked_fill(mask == 0, float('-inf'))  
    
#     attention_weights = F.softmax(scores, dim=-1)  
#     output = torch.matmul(attention_weights, V)  
#     return output, attention_weights  

# class SelfAttention(nn.Module):  
#     """  
#     Self-Attention Module with configurable embedding size.  
#     """  
#     def __init__(self, embed_size: int):  
#         """  
#         Initialize self-attention layers.  
        
#         Args:  
#             embed_size (int): Dimension of input embeddings  
#         """  
#         super(SelfAttention, self).__init__()  
#         self.embed_size = embed_size  
        
#         # Linear layers for query, key, and value  
#         self.query = nn.Linear(embed_size, embed_size)  
#         self.key = nn.Linear(embed_size, embed_size)  
#         self.value = nn.Linear(embed_size, embed_size)  
        
#         # Layer normalization for improved stability  
#         self.layer_norm = nn.LayerNorm(embed_size)  

#     def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  
#         """  
#         Forward pass for self-attention.  
        
#         Args:  
#             x (torch.Tensor): Input tensor  
#             mask (Optional[torch.Tensor]): Attention mask  
        
#         Returns:  
#             torch.Tensor: Attention-weighted output  
#         """  
#         # Residual connection  
#         residual = x  
        
#         # Compute query, key, and value  
#         Q = self.query(x)  
#         K = self.key(x)  
#         V = self.value(x)  
        
#         # Apply scaled dot-product attention  
#         out, _ = scaled_dot_product_attention(Q, K, V, mask)  
        
#         # Residual connection and layer normalization  
#         out = self.layer_norm(out + residual)  
#         return out  

# class CNNWithAttention(nn.Module):  
#     """  
#     CNN model with self-attention mechanism for sequence processing.  
#     """  
#     def __init__(self, input_length: int, num_features: int = 1):  
#         """  
#         Initialize CNN with attention architecture.  
        
#         Args:  
#             input_length (int): Length of input sequence  
#             num_features (int, optional): Number of input features. Defaults to 1.  
#         """  
#         super(CNNWithAttention, self).__init__()  
        
#         # Convolutional layers with increased depth  
#         self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, stride=1, padding=1)  
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  
        
#         # Batch normalization for improved training stability  
#         self.bn1 = nn.BatchNorm1d(32)  
#         self.bn2 = nn.BatchNorm1d(64)  
#         self.bn3 = nn.BatchNorm1d(128)  
        
#         # Dropout for regularization  
#         self.dropout = nn.Dropout(0.3)  
        
#         # Self-Attention Layer  
#         self.attention = SelfAttention(embed_size=128)  
        
#         # Fully connected layers  
#         self.fc1 = nn.Linear(128 * input_length, 256)  
#         self.fc2 = nn.Linear(256, 128)  
#         self.fc3 = nn.Linear(128, 1)  
        
#         # Activation functions  
#         self.relu = nn.ReLU()  
#         self.sigmoid = nn.Sigmoid()  

#     def forward(self, x: torch.Tensor) -> torch.Tensor:  
#         """  
#         Forward pass of the CNN with attention model.  
        
#         Args:  
#             x (torch.Tensor): Input tensor  
        
#         Returns:  
#             torch.Tensor: Model output  
#         """  
#         # Convolutional layers with batch normalization  
#         x = self.bn1(self.relu(self.conv1(x)))  
#         x = self.bn2(self.relu(self.conv2(x)))  
#         x = self.bn3(self.relu(self.conv3(x)))  
        
#         # Reshape for Self-Attention  
#         x = x.permute(0, 2, 1)  # Change shape to (batch, seq_length, channels)  
#         x = self.attention(x)  # Apply self-attention  
        
#         # Flatten before FC layers  
#         x = x.view(x.size(0), -1)  
        
#         # Fully connected layers with dropout  
#         x = self.dropout(self.relu(self.fc1(x)))  
#         x = self.dropout(self.relu(self.fc2(x)))  
#         x = self.fc3(x)  
        
#         return self.sigmoid(x)  

# # Optional: Model initialization and parameter check  
# def init_weights(m):  
#     """  
#     Initialize model weights.  
    
#     Args:  
#         m (nn.Module): Neural network module  
#     """  
#     if isinstance(m, (nn.Linear, nn.Conv1d)):  
#         nn.init.xavier_uniform_(m.weight)  
#         if m.bias is not None:  
#             nn.init.zeros_(m.bias)  



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AdditiveAttention(nn.Module):
#     """
#     Additive Attention Mechanism
#     """
#     def __init__(self, embed_size):
#         super(AdditiveAttention, self).__init__()
#         self.W = nn.Linear(embed_size, embed_size, bias=False)  # Learnable transformation
#         self.V = nn.Linear(embed_size, 1, bias=False)  # Scoring vector

#     def forward(self, x):
#         """
#         x: (batch, seq_length, embed_size)
#         Returns: Attention-weighted representation (batch, embed_size)
#         """
#         score = torch.tanh(self.W(x))  # Apply transformation
#         attn_weights = F.softmax(self.V(score), dim=1)  # Compute attention weights (batch, seq_length, 1)
#         attn_output = torch.sum(attn_weights * x, dim=1)  # Weighted sum over sequence
#         return attn_output

# class CNNWithAttention(nn.Module):
#     """
#     CNN model with Additive Attention for RNA feature extraction.
#     """
#     def __init__(self, input_length: int, num_features: int = 1):
#         """
#         Args:
#             input_length (int): Length of input sequence
#             num_features (int, optional): Number of input features (default: 1).
#         """
#         super(CNNWithAttention, self).__init__()

#         # Convolutional layers with batch normalization
#         self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(128)

#         self.dropout = nn.Dropout(0.3)

#         # Additive Attention
#         self.attention = AdditiveAttention(embed_size=128)

#         # Fully connected layers
#         self.fc1 = nn.Linear(128, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 1)

#         # Activation functions
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         """
#         Forward pass of the CNN with Additive Attention model.
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch, features, seq_length)
#         Returns:
#             torch.Tensor: Model output
#         """
#         x = self.bn1(self.relu(self.conv1(x)))
#         x = self.bn2(self.relu(self.conv2(x)))
#         x = self.bn3(self.relu(self.conv3(x)))

#         # Reshape for Additive Attention
#         x = x.permute(0, 2, 1)  # Change shape to (batch, seq_length, channels)
#         x = self.attention(x)  # Apply Additive Attention

#         # Fully connected layers with dropout
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)

#         return self.sigmoid(x)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple

# # Removed the scaled_dot_product_attention function as it's replaced by
# # the logic within the AdditiveAttention module.

# class AdditiveAttention(nn.Module):
#     """
#     Additive (Bahdanau-style) Self-Attention Module.
#     """
#     def __init__(self, embed_size: int, attention_dim: Optional[int] = None):
#         """
#         Initialize additive attention layers.

#         Args:
#             embed_size (int): Dimension of input embeddings (and Q, K, V).
#             attention_dim (int, optional): Dimension of the hidden layer in the
#                                            attention mechanism. Defaults to embed_size.
#         """
#         super(AdditiveAttention, self).__init__()
#         self.embed_size = embed_size
#         self.attention_dim = attention_dim if attention_dim is not None else embed_size

#         # Linear layers for query, key, and value (remains the same)
#         self.query = nn.Linear(embed_size, embed_size)
#         self.key = nn.Linear(embed_size, embed_size)
#         self.value = nn.Linear(embed_size, embed_size)

#         # Layers for the additive attention scoring mechanism
#         # Projects Query to attention_dim
#         self.W_q = nn.Linear(embed_size, self.attention_dim, bias=False)
#         # Projects Key to attention_dim
#         self.W_k = nn.Linear(embed_size, self.attention_dim, bias=False)
#         # Computes the final score (maps attention_dim to 1)
#         self.v = nn.Linear(self.attention_dim, 1, bias=False)

#         # Layer normalization for improved stability
#         self.layer_norm = nn.LayerNorm(embed_size)
#         self.tanh = nn.Tanh()

#     def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Forward pass for additive self-attention.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_size)
#             mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_length)
#                                            or broadcastable (e.g., (batch_size, 1, seq_length)
#                                            or (batch_size, seq_length, seq_length)).
#                                            Positions with 0 or False will be masked.

#         Returns:
#             torch.Tensor: Attention-weighted output of shape (batch_size, seq_length, embed_size)
#         """
#         # Residual connection
#         residual = x
#         batch_size, seq_length, _ = x.size()

#         # Compute query, key, and value
#         Q = self.query(x)  # (batch_size, seq_length, embed_size)
#         K = self.key(x)    # (batch_size, seq_length, embed_size)
#         V = self.value(x)  # (batch_size, seq_length, embed_size)

#         # --- Additive Attention Score Calculation ---
#         # Project Q and K
#         Q_proj = self.W_q(Q)  # (batch_size, seq_length, attention_dim)
#         K_proj = self.W_k(K)  # (batch_size, seq_length, attention_dim)

#         # Expand dimensions for broadcasting:
#         # Q_proj becomes (batch_size, seq_length, 1, attention_dim)
#         # K_proj becomes (batch_size, 1, seq_length, attention_dim)
#         # The sum will be (batch_size, seq_length, seq_length, attention_dim)
#         # representing the sum for each query-key pair.
#         features = Q_proj.unsqueeze(2) + K_proj.unsqueeze(1)

#         # Apply tanh activation
#         tanh_features = self.tanh(features) # (batch_size, seq_length, seq_length, attention_dim)

#         # Apply v vector (linear layer) to get scores
#         # scores shape: (batch_size, seq_length, seq_length, 1) -> (batch_size, seq_length, seq_length)
#         scores = self.v(tanh_features).squeeze(-1)

#         # Apply mask (if provided)
#         # The mask needs to be compatible with the scores shape (batch_size, seq_length, seq_length)
#         if mask is not None:
#              # Ensure mask is boolean
#             if mask.dtype != torch.bool:
#                  mask = mask == 0 # Assuming 0 means masked

#             # Expand mask if necessary (e.g., from (B, T) or (B, 1, T) to (B, T, T))
#             if mask.dim() == 2: # (B, T) -> (B, 1, T) -> (B, T, T)
#                 mask = mask.unsqueeze(1).expand(batch_size, seq_length, seq_length)
#             elif mask.dim() == 3 and mask.shape[1] == 1: # (B, 1, T) -> (B, T, T)
#                  mask = mask.expand(batch_size, seq_length, seq_length)
#             elif mask.dim() == 3 and mask.shape[2] == 1: # (B, T, 1) -> (B, T, T)
#                  mask = mask.expand(batch_size, seq_length, seq_length)
#             # else: assume mask is already (B, T, T)

#             # Mask needs to be inverted for masked_fill: True positions are *kept*, False are *filled*
#             # However, the common convention is mask=0 means ignore. So we fill where mask is 0 (or False).
#             # Ensure the mask has the same shape as scores.
#             scores = scores.masked_fill(mask, float('-inf')) # Fill masked positions with negative infinity

#         # Compute attention weights using softmax
#         # Softmax over the key dimension (last dimension of scores)
#         attention_weights = F.softmax(scores, dim=-1) # (batch_size, seq_length, seq_length)

#         # Apply attention weights to Value tensor
#         # output shape: (batch_size, seq_length, embed_size)
#         output = torch.matmul(attention_weights, V)

#         # Residual connection and layer normalization
#         out = self.layer_norm(output + residual)
#         return out # Usually we don't need the weights outside, but you could return them: return out, attention_weights

# class CNNWithAttention(nn.Module):
#     """
#     CNN model with self-attention mechanism for sequence processing.
#     Uses AdditiveAttention now.
#     """
#     def __init__(self, input_length: int, num_features: int = 1, attention_hidden_dim: Optional[int] = None):
#         """
#         Initialize CNN with attention architecture.

#         Args:
#             input_length (int): Length of input sequence
#             num_features (int, optional): Number of input features. Defaults to 1.
#             attention_hidden_dim (int, optional): Hidden dimension for additive attention.
#                                                    Defaults to the output channels of the last CNN layer (128).
#         """
#         super(CNNWithAttention, self).__init__()

#         # Convolutional layers with increased depth
#         self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

#         # Batch normalization for improved training stability
#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(128)

#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.3)

#         # --- Use AdditiveAttention Layer ---
#         # The embed_size for attention must match the output channels of the last conv layer
#         self.attention = AdditiveAttention(embed_size=128, attention_dim=attention_hidden_dim)

#         # Fully connected layers
#         # The input size depends on the output of the attention layer
#         self.fc1 = nn.Linear(128 * input_length, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 1)

#         # Activation functions
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid() # Assuming binary classification/regression in [0, 1]

#     def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Forward pass of the CNN with attention model.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, num_features, input_length)
#             mask (Optional[torch.Tensor]): Optional padding mask for the sequence.
#                                            Shape (batch_size, input_length). Positions with
#                                            0 or False indicate padding.

#         Returns:
#             torch.Tensor: Model output
#         """
#         # Convolutional layers with batch normalization
#         x = self.bn1(self.relu(self.conv1(x)))
#         x = self.bn2(self.relu(self.conv2(x)))
#         x = self.bn3(self.relu(self.conv3(x))) # Shape: (batch, 128, input_length)

#         # Reshape for Self-Attention
#         # Change shape to (batch, seq_length, channels/embed_size)
#         x = x.permute(0, 2, 1) # Shape: (batch, input_length, 128)

#         # --- Apply Additive self-attention ---
#         # Pass the mask if available
#         x = self.attention(x, mask=mask) # Shape: (batch, input_length, 128)

#         # Flatten before FC layers
#         x = x.reshape(x.size(0), -1) # Shape: (batch, input_length * 128)

#         # Fully connected layers with dropout
#         x = self.dropout(self.relu(self.fc1(x)))
#         x = self.dropout(self.relu(self.fc2(x)))
#         x = self.fc3(x)

#         # Final activation (adjust if needed for your task)
#         return self.sigmoid(x)