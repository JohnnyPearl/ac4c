# CNN with Self-Attention for Sequence Classification

This repository contains a CNN model with self-attention for sequence classification. The dataset consists of encoded k-mer features stored in `.npz` files.

## Repository Structure
```
├── encoded_data/  
│   ├── train_data_kmer.npz  
│   ├── test_data_kmer.npz  
├── model.py  
├── README.md  
```

- `encoded_data/`: Contains preprocessed dataset files.  
- `model.py`: Implements the CNN model with self-attention.  
- `README.md`: Documentation for usage and structure.

## Data Description

The dataset consists of k-mer encoded features for training and testing:
- `train_data_kmer.npz`: Contains training features (`features`) and labels (`labels`).
- `test_data_kmer.npz`: Contains testing features (`features`) and labels (`labels`).

## Model Overview

The `model.py` script implements a **CNN with Self-Attention**:

### **Key Components**
1. **Scaled Dot-Product Attention**: Computes attention scores to weigh important features.
2. **Self-Attention Module**: Learns feature dependencies within the sequence.
3. **Convolutional Layers**: Extract hierarchical features from input sequences.
4. **Fully Connected Layers**: Classify the sequence based on learned representations.

### **Model Architecture**
- **Convolutional Layers**:
  - `Conv1d(1, 32, kernel_size=3)`
  - `Conv1d(32, 64, kernel_size=3)`
  - `Conv1d(64, 128, kernel_size=3)`
- **Self-Attention Layer**:
  - Learns dependencies between extracted features.
- **Fully Connected Layers**:
  - `Linear(128 * input_length, 128)`
  - `Linear(128, 1) → Sigmoid activation`

## Requirements

Ensure you have the following dependencies installed:
```bash
pip install torch numpy pandas scikit-learn
```