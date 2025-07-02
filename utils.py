from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score, matthews_corrcoef, make_scorer, confusion_matrix, precision_score
import torch
import numpy as np

# Check and Remove Duplicates in a List of Sequences
def deduplicateSequence(sequences):
    # Initialize a dictionary to store sequence counts and their indices
    sequence_positions = {}

    # Loop through the list and track occurrences and their positions
    for index, sequence in enumerate(sequences):
        if sequence in sequence_positions:
            sequence_positions[sequence].append(index)
        else:
            sequence_positions[sequence] = [index]

    # Identify duplicates and their positions
    duplicates = {seq: positions for seq, positions in sequence_positions.items() if len(positions) > 1}

    # Calculate the number of duplicates
    num_duplicates = sum(len(positions) - 1 for positions in duplicates.values())  # Total count of duplicates
    duplicate_sequences = list(duplicates.keys())  # List of unique duplicate sequences
    total_sequences = len(sequences)  # Total number of sequences

    # Calculate percentage of duplicates
    duplicate_percentage = (num_duplicates / total_sequences) * 100 if total_sequences > 0 else 0

    # Print the results
    print(f"Number of duplicate sequences: {num_duplicates}")
    print(f"Duplicate sequences: {duplicate_sequences}")
    print(f"Percentage of duplicates: {duplicate_percentage:.2f}%")
    print(f"Positions of duplicate sequences: {duplicates}")

    # Remove duplicates by keeping only the first occurrence
    unique_sequences = []
    for i, seq in enumerate(sequences):
        if seq not in duplicates:
            unique_sequences.append(seq)

    return unique_sequences

def remove_unequal_sequences(sequences, target_length=201):
    """Removes sequences that are not of the target length and returns the cleaned list, 
       the count of removed sequences, and their positions."""
    
    filtered_sequences = []
    removed_positions = []

    for i, seq in enumerate(sequences):
        if len(seq) == target_length:
            filtered_sequences.append(seq)
        else:
            removed_positions.append(i)

    removed_count = len(removed_positions)
    print(f"Removed {removed_count} sequences from train set at positions: {removed_positions}")

    return filtered_sequences


scoring = {
    'accuracy': make_scorer(accuracy_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'roc_auc': make_scorer(roc_auc_score),
    'aucpr': make_scorer(average_precision_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=0)),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'mcc': make_scorer(matthews_corrcoef)
}

# Train Function
def train(model, train_loader, criterion, optimizer, device, threshold=0.5):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for data, labels in train_loader:
        # Move data to GPU if available
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        output = model(data.unsqueeze(1))

        # Compute loss
        loss = criterion(output.squeeze(), labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimizer step

        # Store predictions and labels
        all_preds.extend((output >= threshold).cpu().numpy())  # Threshold at 0.5
        all_labels.extend(labels.cpu().numpy())

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    return float(f"{avg_loss:.4f}")

# Evaluation function
def evaluate(y_true, y_pred_prob, threshold=0.5):
    # Convert y_pred_prob to a NumPy array if it's a list of arrays
    if isinstance(y_pred_prob, list):
        y_pred_prob = np.concatenate([arr.flatten() for arr in y_pred_prob]) # Concatenate list elements

    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred_prob)
    spe = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    sen = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auroc:.4f}")
    print(f"Specificity: {spe:.4f}")
    print(f"Sensitivity: {sen:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"MCC: {mcc:.4f}")

    return {
        'accuracy': float(f"{acc:.4f}"),
        'auroc': float(f"{auroc:.4f}"),
        'specificity': float(f"{spe:.4f}"),
        'sensitivity': float(f"{sen:.4f}"),
        'precision': float(f"{precision:.4f}"),
        'f1': float(f"{f1:.4f}"),
        'balanced_accuracy': float(f"{ba:.4f}"),
        'mcc': float(f"{mcc:.4f}")
    }

# Function to get model predictions
def get_predictions(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data.unsqueeze(1))
            loss = criterion(output.squeeze(), labels)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(output.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    print(f'Loss: {avg_loss:.4f}')
    return float(f"{avg_loss:.4f}"), all_labels, all_probs