import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from typing import Optional, Tuple  

def scaled_dot_product_attention(  
    Q: torch.Tensor,   
    K: torch.Tensor,   
    V: torch.Tensor,   
    mask: Optional[torch.Tensor] = None  
) -> Tuple[torch.Tensor, torch.Tensor]:  
    """  
    Compute scaled dot-product attention.  
    
    Args:  
        Q (torch.Tensor): Query tensor  
        K (torch.Tensor): Key tensor  
        V (torch.Tensor): Value tensor  
        mask (Optional[torch.Tensor]): Attention mask  
    
    Returns:  
        Tuple of output tensor and attention weights  
    """  
    # Validate input dimensions  
    assert Q.dim() == K.dim() == V.dim(), "Tensor dimensions must match"  
    
    d_k = Q.size(-1)  
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))  
    
    if mask is not None:  
        scores = scores.masked_fill(mask == 0, float('-inf'))  
    
    attention_weights = F.softmax(scores, dim=-1)  
    output = torch.matmul(attention_weights, V)  
    return output, attention_weights  

class SelfAttention(nn.Module):  
    """  
    Self-Attention Module with configurable embedding size.  
    """  
    def __init__(self, embed_size: int):  
        """  
        Initialize self-attention layers.  
        
        Args:  
            embed_size (int): Dimension of input embeddings  
        """  
        super(SelfAttention, self).__init__()  
        self.embed_size = embed_size  
        
        # Linear layers for query, key, and value  
        self.query = nn.Linear(embed_size, embed_size)  
        self.key = nn.Linear(embed_size, embed_size)  
        self.value = nn.Linear(embed_size, embed_size)  
        
        # Layer normalization for improved stability  
        self.layer_norm = nn.LayerNorm(embed_size)  

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  
        """  
        Forward pass for self-attention.  
        
        Args:  
            x (torch.Tensor): Input tensor  
            mask (Optional[torch.Tensor]): Attention mask  
        
        Returns:  
            torch.Tensor: Attention-weighted output  
        """  
        # Residual connection  
        residual = x  
        
        # Compute query, key, and value  
        Q = self.query(x)  
        K = self.key(x)  
        V = self.value(x)  
        
        # Apply scaled dot-product attention  
        out, _ = scaled_dot_product_attention(Q, K, V, mask)  
        
        # Residual connection and layer normalization  
        out = self.layer_norm(out + residual)  
        return out  

class CNNWithAttention(nn.Module):  
    """  
    CNN model with self-attention mechanism for sequence processing.  
    """  
    def __init__(self, input_length: int, num_features: int = 1):  
        """  
        Initialize CNN with attention architecture.  
        
        Args:  
            input_length (int): Length of input sequence  
            num_features (int, optional): Number of input features. Defaults to 1.  
        """  
        super(CNNWithAttention, self).__init__()  
        
        # Convolutional layers with increased depth  
        # self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, stride=1, padding=1)  
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
        # self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 

        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization for improved training stability  
        # self.bn1 = nn.BatchNorm1d(32)  
        # self.bn2 = nn.BatchNorm1d(64)  
        # self.bn3 = nn.BatchNorm1d(128)  

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Dropout for regularization  
        self.dropout = nn.Dropout(0.2)  
        
        # Self-Attention Layer  
        # self.attention = SelfAttention(embed_size=128)  

        self.attention = SelfAttention(embed_size=256)
        
        # Fully connected layers  
        # self.fc1 = nn.Linear(128 * input_length, 256)  
        # self.fc2 = nn.Linear(256, 128)  
        # self.fc3 = nn.Linear(128, 1)  

        self.fc1 = nn.Linear(256 * input_length, 256)
        self.fc2 = nn.Linear(256, 1)
        
        # Activation functions  
        self.relu = nn.LeakyReLU()  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """  
        Forward pass of the CNN with attention model.  
        
        Args:  
            x (torch.Tensor): Input tensor  
        
        Returns:  
            torch.Tensor: Model output  
        """  
        # Convolutional layers with batch normalization  
        x = self.bn1(self.relu(self.conv1(x)))  
        x = self.bn2(self.relu(self.conv2(x)))  
        x = self.bn3(self.relu(self.conv3(x)))  
        
        # Reshape for Self-Attention  
        x = x.permute(0, 2, 1)  # Change shape to (batch, seq_length, channels)  
        x = self.attention(x)  # Apply self-attention  
        
        # Flatten before FC layers  
        x = x.view(x.size(0), -1)  
        
        # Fully connected layers with dropout  
        x = self.dropout(self.relu(self.fc1(x)))  
        # x = self.dropout(self.relu(self.fc2(x)))  
        # x = self.fc3(x)  
        x = self.fc2(x)
        
        return self.sigmoid(x)  

# Optional: Model initialization and parameter check  
def init_weights(m):  
    """  
    Initialize model weights.  
    
    Args:  
        m (nn.Module): Neural network module  
    """  
    if isinstance(m, (nn.Linear, nn.Conv1d)):  
        nn.init.xavier_uniform_(m.weight)  
        if m.bias is not None:  
            nn.init.zeros_(m.bias)  

# Example usage  
if __name__ == "__main__":  
    # Create model instance  
    model = CNNWithAttention(input_length=100)  
    
    # Initialize weights  
    model.apply(init_weights)  
    
    # Print model summary  
    print(model)