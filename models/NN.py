import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class CustomNet(nn.Module):
    def __init__(self, input_dim, num_layers, num_classes):
        super(CustomNet, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            output_dim = num_classes if i == num_layers - 1 else input_dim // (2 ** (i + 1))
            self.layers.append(nn.Linear(input_dim if i == 0 else input_dim // (2 ** i), output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation function except for the last layer
                x = torch.relu(x)
        return x

class NeuralNetwork:
    def __init__(self, input_dim, num_layers, num_classes, batch_size, epochs):
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = CustomNet(self.input_dim, self.num_layers, self.num_classes)

    def fit(self, X_train, y_train, X_test, y_test):
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_data = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(self.epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Neural Network Test Accuracy: {accuracy:.2f}%')

# Example Usage:
# nn_model = NeuralNetwork(input_dim=X_train.shape[1], num_layers=3, num_classes=10, batch_size=64, epochs=30)
# nn_model.fit(X_train, y_train, X_test, y_test)
