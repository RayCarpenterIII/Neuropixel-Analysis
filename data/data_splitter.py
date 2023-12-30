import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class DataSplitter:
    '''
    The purpose of this class is to process the data from a table of predictors over time and where the last column is the class identifier.
    '''
    def __init__(self, spike_df, seq_len=10, batch_size=32):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.spike_df = spike_df[spike_df['frame'] != -1]
        self.X, self.y = self._prepare_data()
        self.X = self.remove_0_columns(self.X)
        self.train_loader, self.test_loader, self.val_loader = self._create_loaders()
    
    def _prepare_data(self):
        X = self.spike_df.drop(columns=['frame']).values
        y = self.spike_df['frame'].values
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        return X, y_encoded
        
    def remove_0_columns(self, X):
        print('Shape of X before =', np.shape(X))
        print('...Checking and removing columns with a standard deviation of 0...')

        std_devs = np.std(X, axis=0)
        zero_std_columns = np.where(std_devs == 0)[0]
        has_zero_std_columns = len(zero_std_columns) > 0

        if has_zero_std_columns:
            X = np.delete(X, zero_std_columns, axis=1)
            print('Deleted columns with zero standard deviation.')
        else:
            print('No columns with zero standard deviation found.')

        print('Shape of X after =', np.shape(X))
        return X # Returns X without 0 columns.
        
    def _split_and_reshape(self, X, y):
            num_samples = X.shape[0] // self.seq_len
            num_features = X.shape[1]

            X = X[:num_samples * self.seq_len]
            X = X.reshape(num_samples, self.seq_len, num_features)

            y = y[:num_samples * self.seq_len]
            y = y.reshape(num_samples, self.seq_len, 1)
            y = y[:, -1]

            return X, y

    def _create_loaders(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=False)
    
        X_train, y_train = self._split_and_reshape(X_train, y_train)
        X_test, y_test = self._split_and_reshape(X_test, y_test)
        X_val, y_val = self._split_and_reshape(X_val, y_val)
    
        train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=self.batch_size, shuffwle=False)  # Set shuffle to False

        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(" ")
        print("X (array): Matrix of number of batch_size, time_steps_per_frame, num_nodes, and number of features per node.")
        print("X.shape = (B, T, N, F)")
        print(f"X_train type: {type(X_train)}")
        print(" ")
        print(f"y_shape = [batch_size, unique_frames_shown_per_{self.seq_len}_timesteps]")
        print(" ")   
        
        return train_loader, test_loader, val_loader

    def compute_correlation_matrix(self):
        corr_matrix = np.corrcoef(self.X, rowvar=False)
        return corr_matrix  