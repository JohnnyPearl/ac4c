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