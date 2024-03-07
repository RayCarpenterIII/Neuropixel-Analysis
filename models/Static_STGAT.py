import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch_geometric.nn import GATv2Conv
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import dense_to_sparse
from math import ceil
'''
The purpose of this script is to be able to call a Spatio-Temporal Graph Attention Network. 
This is a type of ST-GNN with a GAT for the spatial layer and an LSTM as the temporal layer. 
This process the node and directed edge information of a graph over time.
Directed edges can be given to or trained from the model.
Note: Currently this file only trains the adjacency matrices for the edges and cannot accept pre-defined matrices. 
'''

class StaticAdaptiveAdjacencyLayer(nn.Module):
    '''
    Creates a single directed adjacency matrix for the whole system and updates it through backpropagation using the ST-GAT model. 
    The matrix is pushed through a sigmoid function to shrink the 

    Parameters:
        - num_classes (int): The number of unique classes in the dataset. Determines the number of different adjacency matrices to be learned.
        - num_nodes (int): The number of nodes in the graph. This specifies the size of each square adjacency matrix.
        - class_idx (Tensor): A tensor containing the indices of the classes for which the adjacency matrices are to be generated. Each index corresponds to one class.
        - edge_threshold(int): Sets the edge threshold.

    Returns:
        - Tensor: The edge index tensor for the whole system.
    '''
    def __init__(self, num_nodes, edge_threshold):
        super(StaticAdaptiveAdjacencyLayer, self).__init__()
        self.activation = nn.Sigmoid()
        self.edge_threshold = edge_threshold

    def forward(self, V_Adap):
        A_Adap = self.activation(V_Adap)
        batch_size = A_Adap.shape[0]
        edge_index_list = []
        edge_attr_list = []
        for i in range(batch_size):
            edge_index, edge_attr = dense_to_sparse(A_Adap[i])
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list)
        #print("edge_attr:", edge_attr)
        #print("Edge attribute shape:", edge_attr.shape)
        return edge_index, edge_attr

    def get_edge_sliver(self, V_Adap, sliver_size=5):
        A_Adap = self.activation(V_Adap)
        edge_weights = A_Adap[A_Adap > self.edge_threshold]
        edge_indices = (A_Adap > self.edge_threshold).nonzero(as_tuple=False)
        sliver_indices = edge_indices[:sliver_size, :2]
        sliver_weights = edge_weights[:sliver_size]
        return sliver_indices, sliver_weights



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
        self.gat_conv = GATv2Conv(in_channels, spatial_hidden_dim, edge_dim=1)
        self.fc = nn.Linear(spatial_hidden_dim, spatial_out_features)

    def forward(self, x, edge_index, edge_attr):
        B, N, F = x.size()
        x_reshaped = x.reshape(B * N, F)  # Reshape to (B*N, F)
        output = self.gat_conv(x_reshaped, edge_index, edge_attr=edge_attr)
        output_reshaped = output.reshape(B, N, -1)  # Reshape back to (B, N, H)
        output_reduced = self.fc(output_reshaped)
        return output_reduced
    
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

class Static_STGAT(nn.Module):
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
    def __init__(self, spatial_in_features, spatial_hidden_dim, spatial_out_features, num_classes, num_nodes, edge_threshold, temporal_hidden_dim, temporal_layer_dimension, graph_batch_size=32):
        super(Static_STGAT, self).__init__()
        self.V_Adap = nn.Parameter(torch.full((1, num_nodes, num_nodes), 0.2))  # Add batch dimension
        self.dynamic_adjacency = StaticAdaptiveAdjacencyLayer(num_nodes, edge_threshold)
        self.gat_layer = TrainableGATLayer(spatial_in_features, spatial_hidden_dim, spatial_out_features)
        self.num_nodes = num_nodes
        self.lstm = LSTMModel(spatial_out_features * num_nodes, temporal_hidden_dim, temporal_layer_dimension, num_classes)
        self.graph_batch_size = graph_batch_size

    def forward(self, X):
        spatial_out = []
        B, T, N, F = X.size()
        X_reshaped = X.view(B, T, -1)
        
        num_graph_batches = ceil(N / self.graph_batch_size)
        for t in range(T):
            x_t = X_reshaped[:, t, :]
            x_t = x_t.view(B, N, -1)
            spatial_out_t = []
            for i in range(num_graph_batches):
                start_idx = i * self.graph_batch_size
                end_idx = min((i + 1) * self.graph_batch_size, N)
                x_t_batch = x_t[:, start_idx:end_idx, :]
                V_Adap_batch = self.V_Adap[:, start_idx:end_idx, start_idx:end_idx]
                edge_index, edge_attr = self.dynamic_adjacency(V_Adap_batch)
                spatial_out_batch = self.gat_layer(x_t_batch, edge_index, edge_attr)
                spatial_out_t.append(spatial_out_batch)
            spatial_out_t = torch.cat(spatial_out_t, dim=1)
            spatial_out.append(spatial_out_t)
        
        spatial_out = torch.stack(spatial_out, dim=1)
        spatial_out = spatial_out.view(B, T, -1)
        out = self.lstm(spatial_out)
        return out