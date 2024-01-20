import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class DataSplitter:
    '''
    The purpose of this class is to process the data from a table of predictors over time and where the last column is the class identifier.
    '''
    def __init__(self, dataframe, seq_len=10, batch_size=32, device='cpu'):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.dataframe = dataframe
        self.X, self.y = self._prepare_data()
        self.X = self.remove_0_columns(self.X)
        self._create_loaders()
    
    def _prepare_data(self):
        X = self.dataframe.iloc[:, :-1].values
        y = self.dataframe.iloc[:, -1].values
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        return X, y_encoded
        
    def remove_0_columns(self, X):
        std_devs = np.std(X, axis=0)
        zero_std_columns = np.where(std_devs == 0)[0]
        if len(zero_std_columns) > 0:
            X = np.delete(X, zero_std_columns, axis=1)
        return X
        
    def _split_and_reshape(self, X, y):
        num_samples = X.shape[0] // self.seq_len
        num_features = X.shape[1]  # This should be the total number of features (N * F)
        X = X[:num_samples * self.seq_len].reshape(num_samples, self.seq_len, num_features, 1)
        y = y[:num_samples * self.seq_len].reshape(num_samples, self.seq_len, 1)[:, -1]
        return X, y

    def _create_loaders(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=False)

        X_train, y_train = self._split_and_reshape(X_train, y_train)
        X_test, y_test = self._split_and_reshape(X_test, y_test)
        X_val, y_val = self._split_and_reshape(X_val, y_val)

        # Convert numpy arrays to PyTorch tensors and reshape
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)
        self.y_val = torch.tensor(y_val, dtype=torch.long).to(self.device)


        # Create DataLoaders
        self.train_loader = DataLoader(list(zip(self.X_train, self.y_train)), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(list(zip(self.X_test, self.y_test)), batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(list(zip(self.X_val, self.y_val)), batch_size=self.batch_size, shuffle=False)
        
        print(f"X_val shape: {self.X_val.shape}")
        print(f"y_val shape: {self.y_val.shape}")
        print(" ")
        print("X (array): Matrix of number of batch_size, time_steps_per_frame, num_nodes, and number of features per node.")
        print("X.shape = (B, T, N, F)")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"X_train type: {type(self.X_train)}")
        print(" ")
        print(f"X_val shape: {self.X_val.shape}")
        print(f"y_val shape: {self.y_val.shape}")
        print(" ")
        print(f"y_shape = [batch_size, unique_frames_shown_per_{self.seq_len}_timesteps]")  # use self.seq_len
        print(f"y_train shape: {self.y_train.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        print(f"y_test type: {type(self.y_test)}")
        print(" ")

    def compute_correlation_matrix(self):
        corr_matrix = np.corrcoef(self.X, rowvar=False)
        return corr_matrix