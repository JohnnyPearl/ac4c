from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openTSNE import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def classDistVisualize(method, train_data, eval_data, figsize=(6, 6), perplexity = 30):
    train_features, train_labels = train_data[0], train_data[1]
    eval_features, eval_labels = eval_data[0], eval_data[1]
    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Standardize the train and test data (both should be standardized using the same scaler)
    scaled_train_features = scaler.fit_transform(train_features)  # Fit and transform train data
    scaled_eval_features = scaler.transform(eval_features)  # Only transform the test data
    # Apply PCA/TSNE to both train and test data
    projector = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, perplexity = perplexity)
    if (method == 'pca'):
        projector = PCA(n_components=2)
        projected_train_features = projector.fit_transform(scaled_train_features)
        projected_eval_features = projector.transform(scaled_eval_features)
    else:
        projector = TSNE(perplexity=perplexity, metric="euclidean", n_jobs=8, random_state=42, verbose=True)
        projected_train_features = projector.fit(scaled_train_features)
        projected_eval_features = projected_train_features.transform(scaled_eval_features)
    
    projected_features = np.concatenate([projected_train_features, projected_eval_features])
    labels = np.concatenate([train_labels, eval_labels])

    # Create a plot
    plt.figure(figsize=figsize)

    # Scatter plot
    sns.scatterplot(x=projected_features[:, 0], y=projected_features[:, 1], hue=labels, alpha=0.7, palette='viridis')

    # Set axis labels
    axis_label = ['Principal Component 1', 'Principal Component 2'] if method == 'pca' else ['tSNE 1', 'tSNE 2']

    # Title and labels
    plt.title(f'{method.upper()} Comparison Between Train and Test Set')
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.legend(title="Dataset")
    plt.show()

# Function to plot the training and validation losses
def plot_loss(loss_data_1, loss_data_2, num_epoch, figsize=(8, 6), name_data=['Training', 'Validation']):
    plt.figure(figsize=figsize)

    plt.plot(range(1, num_epoch + 1), loss_data_1, label='f{name_data[0]} loss')
    plt.plot(range(1, num_epoch + 1), loss_data_2, label='f{name_data[1]} loss')

    # Title and labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{name_data[0]} versus {name_data[1]} loss.')
    plt.legend()
    plt.show()
