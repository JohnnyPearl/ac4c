# Check and Remove Duplicates in a List of Sequences
def check_and_remove_duplicates(sequences):
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

