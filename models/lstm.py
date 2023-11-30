import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming filtered_normalized_firing_rates DataFrame is already loaded

def create_dataloaders(X, y, batch_size, sequence_length):
    """
    Converts arrays to PyTorch tensors and creates DataLoader for batch processing.

    Parameters:
    X (numpy.ndarray): Input features array.
    y (numpy.ndarray): Output labels array.
    batch_size (int): Batch size for the DataLoader.
    sequence_length (int): The length of the sequence of data.

    Returns:
    DataLoader: DataLoader object for the input data.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return loader

def encode_labels(labels):
    """
    Encodes categorical labels to integers.

    Parameters:
    labels (numpy.ndarray): Array of categorical labels.

    Returns:
    numpy.ndarray: Array of encoded integer labels.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels

def reshape_and_split_data(X, y, sequence_length, test_size, random_state):
    """
    Reshapes the data into sequences and splits it into training and test sets.

    Parameters:
    X (numpy.ndarray): Input features array.
    y (numpy.ndarray): Output labels array.
    sequence_length (int): The length of the sequence of data.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.

    Returns:
    tuple: Tuple containing reshaped and split training and test data.
    """
    num_samples = X.shape[0] // sequence_length
    num_features = X.shape[1]

    X_reshaped = X[:num_samples*sequence_length].reshape(num_samples, sequence_length, num_features)
    y_reshaped = y[:num_samples*sequence_length].reshape(num_samples, sequence_length, 1)
    y_reshaped = y_reshaped[:, -1]  # Predicting a single value for each sequence

    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_reshaped, test_size=test_size, random_state=random_state, shuffle=False
    )

    return X_train, X_test, y_train.ravel(), y_test.ravel()

class LSTMModel(nn.Module):
    """
    LSTM model for sequence classification.

    Attributes:
    input_dim (int): The number of input features.
    hidden_dim (int): The number of features in the hidden state h.
    layer_dim (int): The number of recurrent layers.
    output_dim (int): The number of output classes.
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    Trains the LSTM model.

    Parameters:
    model (LSTMModel): The LSTM model to train.
    train_loader (DataLoader): DataLoader for the training data.
    criterion (loss): The loss function.
    optimizer (optim): The optimization algorithm.
    num_epochs (int): The number of epochs to train the model.

    Returns:
    None
    """
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        correct_train_preds = 0.0
        total_train_samples = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct_train_preds += (preds == labels).float().sum()
            total_train_samples += labels.size(0)

        train_acc = correct_train_preds / total_train_samples
        print(f'Epoch {epoch+1}: Loss: {train_running_loss/len(train_loader)}, Train Acc: {train_acc}')

def evaluate_model(model, test_loader):
    """
    Evaluates the LSTM model on test data.

    Parameters:
    model (LSTMModel): The LSTM model to evaluate.
    test_loader (DataLoader): DataLoader for the test data.

    Returns:
    None
    """
    model.eval()
    correct_test_preds = 0.0
    total_test_samples = 0.0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            out = model(features)
            _, preds = torch.max(out, 1)
            correct_test_preds += (preds == labels).float().sum()
            total_test_samples += labels.size(0)

    test_acc = correct_test_preds / total_test_samples
    print(f'Test Acc: {test_acc}')

def run_lstm_model(data, sequence_length=10, test_size=0.2, random_state=42, hidden_dim=500, layer_dim=10, learning_rate=0.01, num_epochs=100, batch_size=64):
    """
    Main function to run the LSTM model with the given data and hyperparameters.

    Parameters:
    data (DataFrame): The DataFrame containing the input data.
    sequence_length (int): The length of the sequence of data.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.
    hidden_dim (int): The number of features in the hidden state h.
    layer_dim (int): The number of recurrent layers.
    learning_rate (float): The learning rate for the optimizer.
    num_epochs (int): The number of epochs to train the model.
    batch_size (int): The size of the batches for training and testing.

    Returns:
    None
    """
    y = encode_labels(data['frame'].values)
    X = data.drop(columns=['frame']).values
    X_train, X_test, y_train, y_test = reshape_and_split_data(X, y, sequence_length, test_size, random_state)
    train_loader = create_dataloaders(X_train, y_train, batch_size, sequence_length)
    test_loader = create_dataloaders(X_test, y_test, batch_size, sequence_length)

    input_dim = X_train.shape[-1]
    output_dim = len(np.unique(y))
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs)
    evaluate_model(model, test_loader)

# Example usage
# run_lstm_model(filtered_normalized_firing_rates)
