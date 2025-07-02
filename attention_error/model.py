# from torch import nn
# import torch

# class SimpleCNN(nn.Module):
#     def __init__(self, input_length):
#         super(SimpleCNN, self).__init__()

#         # Convolutional layers
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

#         self.dropout = nn.Dropout(0.5)  # Dropout layer

#         self.fc1 = nn.Linear(128 * input_length, 128)  # Fully connected layer
#         self.fc2 = nn.Linear(128, 1)  # Binary classification

#         # Sigmoid activation for output layer
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Convolutional layers followed by ReLU
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))

#         # Flatten the output from CNN layers
#         x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers

#         # Fully connected layers
#         x = self.dropout(torch.relu(self.fc1(x)))
#         # x = torch.relu(self.fc1(x))
#         x = self.fc2(x)

#         # Sigmoid activation for binary classification
#         x = self.sigmoid(x)
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F

# Scaled Dot-Product Attention Function
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return out

# CNN Model with Self-Attention
class CNNWithAttention(nn.Module):
    def __init__(self, input_length):
        super(CNNWithAttention, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.2)  

        # Self-Attention Layer (embed_size should match last Conv1d output channels)
        self.attention = SelfAttention(embed_size=128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * input_length, 128)  
        self.fc2 = nn.Linear(128, 1)  

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))  # Shape: (batch, 128, seq_length)

        # Reshape for Self-Attention
        x = x.permute(0, 2, 1)  # Change shape to (batch, seq_length, 128)
        x = self.attention(x)  # Apply self-attention

        # Flatten before FC layers
        x = x.view(x.size(0), -1)  
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return self.sigmoid(x)