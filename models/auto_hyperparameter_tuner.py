import random
import numpy as np
from filelock import FileLock
import tempfile
import ray
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.train import Checkpoint
from ray.tune.search.hebo import HEBOSearch
import torch
import torch.optim as optim
import wandb
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import ray
import os
import pickle
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch import nn
import time
from torch_geometric.utils import add_self_loops
from torch.cuda.amp import GradScaler, autocast

from models.STGAT import * 
from models.PCR import * 
from models.NN import *
from models.STTR import *

class ModelTrainer:
    def __init__(self, param_space):
        self.param_space = param_space
        self.num_samples = param_space['num_samples']
        self.wandb_project = param_space['wandb_project']
        self.wandb_api_key = param_space['wandb_api_key']
        self.computed_params = {}

    def initialize_model(self, config, devic):
        model_name = config.get("Architecture")
        params = self.computed_params
        num_nodes, num_classes, spatial_in_features, lstm_input_dim, temporal_hidden_dim, temporal_layer_dimension, num_epochs, temporal_output_dim = (
            params['num_nodes'], params['num_classes'], params['spatial_in_features'], 
            params['lstm_input_dim'], params['temporal_hidden_dim'], params['temporal_layer_dimension'], 
            params['num_epochs'], params['temporal_output_dim']
        )
        
        if model_name == 'ST-GAT':
            model = STGAT(spatial_in_features, config["spatial_hidden_dim"],    config["spatial_out_features"], 
                      num_classes, num_nodes, config["edge_threshold"], lstm_input_dim, 
                      temporal_hidden_dim, temporal_layer_dimension, temporal_output_dim)
            return model
        elif model_name == 'LSTM':
            model = LSTM(input_dim, config["spatial_hidden_dim"], layer_dim, ouput_dim)
            return model

        elif model_name == 'PCR':
            model = PCRModel(n_components=config.get("n_components", 10))
            return model
        
        elif model_name == 'NN':
            model = NNModel(input_dim, config["hidden_dim"], config["output_dim"])
            return model
        
        elif model_name == 'Transformer':
            if model_name == 'Transformer':
                model = TransformerTemporalLayer(
                    original_input_dim=config["original_input_dim"],
                    transformer_input_dim=config["transformer_input_dim"],  
                    num_heads=config["num_heads"],
                    num_layers=config["num_layers"],
                    output_dim=config["output_dim"]
                )
        return model
        
        # We can add more elif blocks for other models
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")
        
    def train_model(self, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb = setup_wandb(config, project=self.wandb_project, api_key=self.wandb_api_key)
        should_checkpoint = config.get("should_checkpoint", False)

        from data_processors.data_splitter import DataSplitter
        
        mouse_number = config['mouse_number']
        timesteps = config['timesteps']

        spike_trains_file_path = config['file_path']
        #spike_trains_file_path = '/proj/STOR/pipiras/Neuropixel/output/spike_trains_with_stimulus_session_732592105_10.pkl'

        #spike_trains_file_path = f'/nas/longleaf/home/rayrayc/output/spike_trains_with_stimulus_session_719161530_10.pkl'

        with open(spike_trains_file_path, 'rb') as f:
                spike_df = pickle.load(f)
                print(f"Loaded spike trains dataset: {type(spike_df)}")

        # Split data into train, test, and validation sets. 
        DataSplitter = DataSplitter(dataframe = spike_df, batch_size = config['batch_size'])

        # Access the data loaders.
        ### Redo the code to combine the trains into the loader.
        X_train = DataSplitter.X_train
        X_test = DataSplitter.X_test
        #X_val = DataSplitter.X_val
        y_train = DataSplitter.y_train
        y_test = DataSplitter.y_test
        #y_val = DataSplitter.y_val
        train_loader = DataSplitter.train_loader
        test_loader = DataSplitter.test_loader
        #val_loader = DataSplitter.val_loader
        X = DataSplitter.X
        y = DataSplitter.y

        # Early stopping parameters
        patience = config['early_stop_patience']
        min_delta = config['early_stop_delta']
        best_val_acc = 0
        epochs_no_improve = 0

        # Parameters given through data.
        self.computed_params['num_nodes'] = np.shape(X)[1]
        self.computed_params['num_classes'] = len(np.unique(y_train))
        self.computed_params['num_time_steps'] = X_train.shape[1]
        self.computed_params['spatial_in_features'] = X_train.shape[3]
        self.computed_params['lstm_input_dim'] = self.computed_params['spatial_in_features'] * self.computed_params['num_nodes']
        self.computed_params['edge_index'] = torch.tensor([[i, j] for i in range(self.computed_params['num_nodes']) for j in range(self.computed_params['num_nodes']) if i != j], dtype=torch.long).t().contiguous().to(device)
        print("Shape of edge_index:", self.computed_params['edge_index'].shape)
        print("Content of edge_index:", self.computed_params['edge_index'])

        self.computed_params['temporal_output_dim'] = int(len(np.unique(y_train)))
        self.computed_params['temporal_hidden_dim'] = int(config["temporal_hidden_dim"])
        self.computed_params['temporal_layer_dimension'] = int(config["temporal_layer_dimension"])
        self.computed_params['num_epochs'] = int(config['num_epochs'])
        self.computed_params['label_encoder'] = LabelEncoder().fit(y_train.ravel())
        
        params = self.computed_params
        num_nodes, num_classes, spatial_in_features, lstm_input_dim, temporal_hidden_dim, temporal_layer_dimension, num_epochs, temporal_output_dim = (
            params['num_nodes'], params['num_classes'], params['spatial_in_features'], 
            params['lstm_input_dim'], params['temporal_hidden_dim'], params['temporal_layer_dimension'], 
            params['num_epochs'], params['temporal_output_dim']
        )


        model = self.initialize_model(config, device)
        model.to(device)
    
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=config["lr"])
    
        scaler = GradScaler()  # Initialize the GradScaler for AMP
    
        # Training Loop
        highest_test_acc = 0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0  # Initialize running_loss before the loop
            correct_train = 0
            total_train = 0
        
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                labels = labels.squeeze().long()
        
                # Transform labels to class indices
                class_idx = self.computed_params['label_encoder'].transform(labels.cpu().numpy())
                class_idx = torch.tensor(class_idx, dtype=torch.long).to(device)
        
                # Forward pass with class_idx
                outputs = model(features, class_idx)
        
                # Compute loss, backward pass, and optimization
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        
                running_loss += loss.item()  # Update running_loss
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                features.cpu()
                labels.cpu()
        
            train_acc = 100 * correct_train / total_train

            model.eval()  # Set the model to evaluation mode
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    labels = labels.squeeze().long()
                    class_idx = self.computed_params['label_encoder'].transform(labels.cpu().numpy())[0] # Transform labels to class indices
                    class_idx = torch.tensor(class_idx, dtype=torch.long).to(device)
                    outputs = model(features, class_idx) # Call the model with the features and class index
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                    
                    features.cpu()
                    labels.cpu()

            test_acc = 100 * correct_test / total_test

            # Update the highest test accuracy
            if test_acc > highest_test_acc:
                highest_test_acc = test_acc

            end_time = time.time()  # End time of the epoch
            epoch_duration = end_time - start_time
            wandb.log({
                "Epoch": epoch, 
                "Loss": running_loss, 
                "Train Accuracy": train_acc, 
                "Test Accuracy": test_acc, 
                "Epoch Duration": epoch_duration,
                "Architecture": config['Architecture']}
            )

            # Print Results
            print(f'Epoch {epoch+1}, Duration: {epoch_duration:.2f}s, Loss: {np.round((running_loss / len(train_loader)),2)}, Train Acc: {np.round(train_acc, 2)}%, Test Acc: {np.round(test_acc, 2)}%')

            # Early stopping logic
            if test_acc - best_val_acc > min_delta:
                best_val_acc = test_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f'Highest Test Accuracy: {highest_test_acc}%')
        train.report({'test_acc': highest_test_acc})  # Report the highest test accuracy
    
    def execute_tuning(self):
        scheduler = ASHAScheduler(max_t=100, grace_period=5, reduction_factor=2, brackets=3)
        resources_per_trial = {"cpu": 16, "gpu": 1}    #Pass as parameter instead 
        hebo = HEBOSearch(metric="test_acc", mode='max')

        tuner = tune.Tuner(
            tune.with_resources(self.train_model, resources=resources_per_trial),
            tune_config=tune.TuneConfig(
                search_alg=hebo,
                scheduler=scheduler,
                metric="test_acc",
                mode="max",
                num_samples=self.num_samples
            ),
            run_config=train.RunConfig(
                name="exp",
                stop={"training_iteration": 5},
                callbacks=[WandbLoggerCallback(project=self.wandb_project, api_key=self.wandb_api_key)]
            ),
            param_space=self.param_space
        )

        results = tuner.fit()
        print("Best config is:", results.get_best_result().config)

'''
### Example usage
param_space = {
    "wandb_project": "Predicting Visual Stimulus",
    "wandb_api_key": "7c8d251196fd96d2a93bfb6ffd0005ac030ce42b",
    "num_epochs": 50,
    "lr": tune.uniform(.0001,.001),
    "temporal_hidden_dim": tune.randint(100, 2000),
    "spatial_hidden_dim": tune.randint(5, 1000),
    "edge_threshold": tune.uniform(0, 0.75),
    "early_stop_patience": 10,
    "early_stop_delta": 0.01,
    "batch_size": 32,
    "temporal_layer_dimension": 1,
    "spatial_out_features": 1,
    "mouse_number": 721123822,
    "timesteps": 10,
    "Architecture": 'ST-GAT',
    "num_samples": 100
    }

trainer = ModelTrainer(param_space)
trainer.execute_tuning()
'''
