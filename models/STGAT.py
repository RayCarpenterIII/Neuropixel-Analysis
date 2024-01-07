import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder

'''
The purpose of this script is to be able to call a Spatio-Temporal Graph Attention Network. 
This is a type of ST-GNN with a GAT for the spatial layer and an LSTM as the temporal layer. 
This process the node and directed edge information of a graph over time.
Directed edges can be given to or trained from the model.
Note: Currently this file only trains the adjacency matrices for the edges and cannot accept pre-defined matrices. 
'''

class DynamicAdaptiveAdjacencyLayer(nn.Module):
    '''
    Creates a directed adjacency matrix for each class and updates it through backpropagation using the ST-GAT model. 
    The matrix is pushed through a sigmoid function to shrink the 

    Parameters:
        - num_classes (int): The number of unique classes in the dataset. Determines the number of different adjacency matrices to be learned.
        - num_nodes (int): The number of nodes in the graph. This specifies the size of each square adjacency matrix.
        - class_idx (Tensor): A tensor containing the indices of the classes for which the adjacency matrices are to be generated. Each index corresponds to one class.
        - edge_threshold(int): Sets the edge threshold.

    Returns:
        - Tensor: The edge index tensor for the specified class. 
    '''
    def __init__(self, num_classes, num_nodes, edge_threshold):
        super(DynamicAdaptiveAdjacencyLayer, self).__init__()
        self.V_Adap = nn.Parameter(torch.randn(num_classes, num_nodes, num_nodes))
        self.activation = nn.Sigmoid()
        self.edge_threshold = edge_threshold

    def forward(self, class_idx):
        A_Adap = self.activation(self.V_Adap[class_idx]) / self.V_Adap.size(1)
        edge_index = (A_Adap > self.edge_threshold).nonzero(as_tuple=False).t()
        return edge_index
        
class TrainableGATLayer(nn.Module):
    '''
    The spatial layer of a Spatio-Temporal Graph Neural Network (ST-GNN). This layer uses a Graph Attention Network (GAT) to update the node and vertex features of a directed graph.

    Parameters:
        - in_channels (int): The number of input features for each vertex/node in the graph.
        - spatial_hidden_dim (int): The dimension of the hidden layer within the GAT.
        - spatial_out_features (int): The number of output features for each vertex after processing.
        - x (Tensor): The node features tensor, with shape (B, N, F), where B is the batch size, N is the number of nodes, and F is the number of features per node.
        - edge_index (Tensor): The edge index tensor, representing the connections between nodes in the graph.

    Returns:
        - Tensor: A transformed node features tensor with updated spatial information. The shape of the output tensor is (B, N, spatial_out_features), where B is the batch size, N is the number of nodes, and spatial_out_features is the number of output features per node after transformation.
    '''
    def __init__(self, in_channels, spatial_hidden_dim, spatial_out_features):
        super(TrainableGATLayer, self).__init__()
        self.gat_conv = GATv2Conv(in_channels, spatial_hidden_dim)  
        self.fc = nn.Linear(spatial_hidden_dim, spatial_out_features) 

    def forward(self, x, edge_index):
        B, N, F = x.size()
        x_reshaped = x.reshape(-1, F)
        edge_index_repeated = edge_index.repeat(1, B)
        output = self.gat_conv(x_reshaped, edge_index_repeated) 
        output_reduced = self.fc(output) 
        return output_reduced.reshape(B, N, -1)
    
class LSTMModel(nn.Module):
    '''
    A LSTM (Long Short-Term Memory) model for processing temporal data.

    Parameters:
        - input_dim (int): The number of input features.
        - hidden_dim (int): The size of the LSTM hidden layer.
        - layer_dim (int): The number of LSTM layers in the network.
        - output_dim (int): The size of the output layer.
        - x (tensor): The output from the spatial layer. X should be of the shape (batch_size, time_steps, flattened_nodes_and_features)

    Returns:
        - out(tensor): Prediction per interval.
    '''
    ### Future me, try adding a dropout layer.
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X): 
        out, _ = self.lstm(X)
        out = self.fc(out[:, -1, :])
        return out 

class STGAT(nn.Module):
    '''
    A type of spatio-temporal graph neural network that uses a graph attention network as the spatial layer and an LSTM as the temporal layer.
    
    Parameters:
        - spatial_in_features(int): Number of features in the spatial layer. 
        - spatial_hidden_dim (int): The dimension of the hidden layer within the GAT.
        - spatial_out_features (int): The number of output features for each vertex after processing.
        - num_classes (int): The number of unique classes in the dataset. Determines the number of different adjacency matrices to be learned.
        - num_nodes (int): The number of nodes in the graph. This specifies the size of each square adjacency matrix.
        - edge_threshold(int): Sets the edge threshold.
        - temporal_input_layer (int): The number of input features for the temporal layer. 
        - temporal_hidden_layer (int): The size of the temporal hidden layer.
        - temporal_layer_dimension (int): Number of layers in the temporal model. In this case it is our LSTM
        - temporal_output_dim (): The size of the output layer for the temporal dimension. In this case it is our model's output. It should be of length Y. 
        - x (Tensor): The node features tensor, with shape (B, N, F), where B is the batch size, N is the number of nodes, and F is the number of features per node.
        - class_idx (Tensor): A tensor containing the indices of the classes for which the adjacency matrices are to be generated. Each index corresponds to one class.
    
    Returns: 
        - A prediction per interval. (Tensor)
    '''
    def __init__(self, spatial_in_features, spatial_hidden_dim, spatial_out_features, num_classes, num_nodes, edge_threshold, temporal_input_layer, temporal_hidden_layer, temporal_layer_dimension, temporal_output_dim):
        super(STGAT, self).__init__()
        self.spatial_in_features = spatial_in_features  
        self.dynamic_adjacency = DynamicAdaptiveAdjacencyLayer(num_classes, num_nodes, edge_threshold)
        self.gat_layer = TrainableGATLayer(self.spatial_in_features, spatial_hidden_dim, spatial_out_features)
        self.num_nodes = num_nodes
        self.lstm = LSTMModel(temporal_input_layer, temporal_hidden_layer, temporal_layer_dimension, temporal_output_dim)

    def forward(self, X, class_idx):
        spatial_out = []
        B, T, N, F = X.size()  
        X_reshaped = X.view(B, T, -1)  # Reshape to (B, T, N*F)

        edge_index = self.dynamic_adjacency(class_idx)
        for t in range(T):
            x_t = X_reshaped[:, t, :]  
            x_t = x_t.view(B, self.num_nodes, -1)  # Reshape back to (B, N, NF)
            spatial_out_t = self.gat_layer(x_t, edge_index)
            spatial_out.append(spatial_out_t)

        spatial_out = torch.stack(spatial_out, dim=1)
        spatial_out = spatial_out.view(B, T, -1)
        out = self.lstm(spatial_out)
        return out