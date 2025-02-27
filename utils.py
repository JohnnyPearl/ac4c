from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score, matthews_corrcoef, make_scorer, confusion_matrix


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
def train(model, train_loader, criterion, optimizer, device):
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
        all_preds.extend((output >= 0.5).cpu().numpy())  # Threshold at 0.5
        all_labels.extend(labels.cpu().numpy())

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc

# Evaluation function
def evaluate(y_true, y_pred_prob, threshold=0.5):
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred_prob)
    sen = tp / (tp + fn) if (tp + fn) != 0 else 0
    spe = tn / (tn + fp) if (tn + fp) != 0 else 0
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    

    return acc, auroc, sen, spe, ba, f1, mcc


