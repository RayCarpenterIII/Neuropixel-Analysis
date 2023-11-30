import torch 
import time
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torch_geometric.nn import GATConv


class AdaptiveAdjacencyLayer(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super(AdaptiveAdjacencyLayer, self).__init__()
        self.V_Adap = nn.Parameter(torch.randn(num_nodes, num_nodes))  # Changing the shape to (num_nodes, num_nodes)
        self.activation = nn.Sigmoid()
        
    def forward(self, X):
        A_Adap = self.activation(self.V_Adap) / self.V_Adap.size(0)
        return torch.matmul(X, A_Adap)  

class TrainableGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TrainableGATLayer, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.gat_conv(x, edge_index)
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X): # X: (batch_size, time_steps, flattened_nodes_and_features)
        # Pass through LSTM
        out, _ = self.lstm(X)
        # Apply the fully connected layer 
        out = self.fc(out[:, -1, :])
        return out

class STGNN(nn.Module):
    def __init__(self, spatial_in_features, num_nodes, lstm_input_dim, hidden_dim, layer_dim, output_dim, edge_index):
        super(STGNN, self).__init__()
        self.adaptive_adjacency = AdaptiveAdjacencyLayer(num_nodes, hidden_dim)
        self.gat_layer = TrainableGATLayer(spatial_in_features, spatial_in_features)
        self.num_nodes = num_nodes
        self.lstm = LSTMModel(lstm_input_dim, hidden_dim, layer_dim, output_dim)
        self.edge_index = edge_index  

    def forward(self, X, spatial_in_features):
        spatial_out = []
        B, T, NF = X.size()
        for t in range(T):
            x_t = X[:, t, :].view(B, self.num_nodes, spatial_in_features).squeeze(-1)
            x_t_adj = [self.adaptive_adjacency(x) for x in x_t]
            x_t_adj = torch.stack(x_t_adj)
            spatial_out_t = self.gat_layer(x_t_adj.view(-1, spatial_in_features), self.edge_index)  
            spatial_out_t = spatial_out_t.view(B, self.num_nodes, spatial_in_features)
            spatial_out.append(spatial_out_t)
        spatial_out = torch.stack(spatial_out, dim=1)
        spatial_out = spatial_out.view(B, T, -1)
        out = self.lstm(spatial_out)
        return out

    
def create_stgnn_model(X, y, hidden_dim, layer_dim, output_dim, device, edge_threshold=0.3):
    spatial_in_features = X.shape[3]
    num_nodes = X.shape[2]
    lstm_input_dim = spatial_in_features * num_nodes

    # Encode labels to class indices (if not already encoded)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Define a correlation-based edge index
    corr_matrix = np.corrcoef(X.reshape(-1, spatial_in_features * num_nodes).T)  # Reshape X to calculate correlations
    edges = np.argwhere(corr_matrix > edge_threshold)

    # edge_index should be a 2xN tensor where N is the number of edges, so we transpose the result
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    # Create the STGNN model
    model = STGNN(spatial_in_features, num_nodes, lstm_input_dim, hidden_dim, layer_dim, output_dim, edge_index)
    model.to(device)
    return model


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, metrics_file):
    for epoch in range(num_epochs):
        start_time = time.time()  
        model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features, labels = features.to(device), labels.to(device)
            labels = labels.squeeze().long() 

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train

        # Test the model
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                labels = labels.squeeze().long()
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test

        print(f'Epoch {epoch+1}, Loss: {np.round((running_loss / len(train_loader)),2)}, Train Acc: {np.round(train_acc, 2)}%, Test Acc: {np.round(test_acc, 2)}%')

        end_time = time.time()  
        epoch_duration = end_time - start_time
        with open(metrics_file, 'a') as file:
            file.write(f'Epoch {epoch+1}, Loss: {np.round((running_loss / len(train_loader)),2)}, Train Acc: {np.round(train_acc, 2)}%, Test Acc: {np.round(test_acc, 2)}%\n')
            file.write(f'Epoch Duration: {epoch_duration} seconds\n')

class CapturableAdam(torch.optim.Adam):
    def __init__(self, *args, **kwargs):
        super(CapturableAdam, self).__init__(*args, **kwargs)
        self.defaults['capturable'] = True

def create_optimizer_and_criterion(model, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = CapturableAdam(model.parameters(), lr=learning_rate)
    return optimizer, criterion

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size):
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def run_stgnn(X_train, y_train, X_test, y_test, spatial_in_features, hidden_dim, layer_dim, output_dim, learning_rate, num_epochs, batch_size, metrics_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the STGNN model
    model = create_stgnn_model(X_train, y_train, hidden_dim=32, layer_dim=1, output_dim=output_classes, device=device)

    # Create data loaders for training and testing
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size)

    # Initialize the optimizer and loss criterion
    optimizer, criterion = create_optimizer_and_criterion(model, learning_rate)

    # Call the train_model function to start training 
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, metrics_file)
