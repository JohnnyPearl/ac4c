def classDistVisualize(method, figsize=(8, 5)):
    scaler = StandardScaler()

    # Standardize the train and test data (both should be standardized using the same scaler)
    scaled_train_features = scaler.fit_transform(train_data)  # Fit and transform train data
    scaled_test_features = scaler.transform(test_data)  # Only transform the test data
    # Apply PCA/TSNE to both train and test data
    perplexity = 30
    projector = PCA(n_components=2) if method == "pca" else TSNE(n_components=2, perplexity = perplexity)
    projected_train_features = projector.fit_transform(scaled_train_features)
    projected_test_features = projector.transform(scaled_test_features)
    projected_features = np.concatenate([projected_train_features, projected_test_features])
    labels = np.concatenate([train_labels, test_labels])

    # Create a plot
    plt.figure(figsize=figsize)

    # Scatter plot
    sns.scatterplot(x=projected_features[:, 0], y=projected_features[:, 1], hue=labels, alpha=0.7, palette='viridis')

    # Title and labels
    plt.title(f'{method.upper()} Comparison Between Train and Test Set')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Show legend
    plt.legend(title="Dataset")

    # Display the plot
    plt.show()