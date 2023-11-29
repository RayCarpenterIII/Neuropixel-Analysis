import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

def prepare_data(df, edge_threshold=0.3, train_ratio=0.8):
    """
    Prepares input data for training and testing a model.

    Args:
        df (pandas.DataFrame): The input dataframe containing the data.
        edge_threshold (float, optional): The correlation threshold for determining edges. Defaults to 0.3.
        train_ratio (float, optional): The ratio of data to be used for training. Defaults to 0.8.

    Returns:
        train_data (torch_geometric.data.Data): The training data.
        test_data (torch_geometric.data.Data): The testing data.
        num_classes (int): The number of classes in the data.
    """
    y = df['frame'].values
    X = df.drop(columns=['frame']).values
    num_classes = len(np.unique(y))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    corr_matrix = np.corrcoef(X, rowvar=False)
    edges = np.argwhere(corr_matrix > edge_threshold)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)
    train_edges = edge_index[:, perm[:int(train_ratio * num_edges)]]
    test_edges = edge_index[:, perm[int(train_ratio * num_edges):]]

    num_samples = X.shape[0]
    perm = torch.randperm(num_samples)
    X_train = X[perm[:int(train_ratio * num_samples)], :]
    y_train = y_encoded[perm[:int(train_ratio * num_samples)]]
    X_test = X[perm[int(train_ratio * num_samples):], :]
    y_test = y_encoded[perm[int(train_ratio * num_samples):]]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_data = Data(x=X_train_tensor, y=y_train_tensor, edge_index=train_edges)
    test_data = Data(x=X_test_tensor, y=y_test_tensor, edge_index=test_edges)

    train_data.num_classes = num_classes
    test_data.num_classes = num_classes

    return train_data, test_data, num_classes

def define_model(num_features, num_classes, conv_layers=[128, 64]):
    """
    Defines a graph neural network model.

    Args:
        num_features (int): Number of input features.
        num_classes (int): Number of output classes.
        conv_layers (list, optional): List of integers representing the number of output channels for each convolutional layer. Defaults to [128, 64].

    Returns:
        torch.nn.Module: The defined graph neural network model.
    """
    class GNN(torch.nn.Module):
        def __init__(self, num_features, num_classes, conv_layers):
            super(GNN, self).__init__()
            layers = []
            in_channels = num_features
            for out_channels in conv_layers:
                layers.append(GCNConv(in_channels, out_channels))
                in_channels = out_channels
            self.convs = torch.nn.ModuleList(layers)
            self.classifier = Linear(in_channels, num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
            out = F.log_softmax(self.classifier(x), dim=1)
            return out

    model = GNN(num_features, num_classes, conv_layers)
    return model

def train_model(model, train_data, test_data, device, lr=0.01, epochs=50):
    """
    Trains input model using inputed training data and evaluates performance on the test data.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_data (torch.Tensor): The training data.
        test_data (torch.Tensor): The test data.
        device (torch.device): The device to be used for training and evaluation.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        epochs (int, optional): The number of training epochs. Defaults to 50.

    Returns:
        tuple: A tuple containing the trained model and a list of test accuracies at each epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.to(device))
        loss = F.nll_loss(out, train_data.y.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        out_test = model(test_data.to(device))
        _, preds_test = torch.max(out_test, dim=1)
        correct_test = (preds_test == test_data.y.to(device)).sum().item()
        accuracy_test = correct_test / test_data.y.size(0)
        test_accuracies.append(accuracy_test)

    return model, test_accuracies

def evaluate_model(model, train_data, test_data, device):
    """
    Evaluate the performance of a model on test data
    using Precision, Recall, F1 Score, Test Accuracy,
    and ROC AUC.

    Args:
        model (torch.nn.Module): The model to evaluate.
        train_data (torch.Tensor): The training data.
        test_data (torch.Tensor): The test data.
        device (torch.device): The device to run the evaluation on.

    Returns:
        dict: A dictionary containing the evaluation metrics, including precision, recall,
              F1 score, test accuracy, and ROC AUC.
    """
    model.eval()
    out_test = model(test_data.to(device))
    _, preds_test = torch.max(out_test, dim=1)
    out_prob_test = torch.exp(out_test)
    out_prob_test_np = out_prob_test.cpu().detach().numpy()
    y_test_np = test_data.y.cpu().numpy()

    precision = precision_score(y_test_np, preds_test.cpu().numpy(), average='macro')
    recall = recall_score(y_test_np, preds_test.cpu().numpy(), average='macro')
    f1 = f1_score(y_test_np, preds_test.cpu().numpy(), average='macro')
    accuracy_test = (preds_test == test_data.y.to(device)).sum().item() / test_data.y.size(0)
    roc_auc = roc_auc_score(y_test_np, out_prob_test_np, multi_class='ovr', average='macro')

    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Test Accuracy': accuracy_test,
        'ROC AUC': roc_auc
    }

    return metrics

def run_gnn(df, edge_threshold=0.3, train_ratio=0.8, conv_layers=[128, 64], lr=0.01, epochs=50):
    """
    Runs a graph neural network (GNN) on the given dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe containing the graph data.
        edge_threshold (float, optional): The threshold for edge weights. Defaults to 0.3.
        train_ratio (float, optional): The ratio of training data to total data. Defaults to 0.8.
        conv_layers (list, optional): The number of units in each convolutional layer. Defaults to [128, 64].
        lr (float, optional): The learning rate for training the model. Defaults to 0.01.
        epochs (int, optional): The number of training epochs. Defaults to 50.

    Returns:
        dict: A dictionary containing the evaluation metrics of the trained model.
    """
    train_data, test_data, num_classes = prepare_data(df, edge_threshold, train_ratio)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model(train_data.num_features, num_classes, conv_layers).to(device)
    trained_model, test_accuracies = train_model(model, train_data, test_data, device, lr, epochs)
    metrics = evaluate_model(trained_model, train_data, test_data, device)
    return metrics
