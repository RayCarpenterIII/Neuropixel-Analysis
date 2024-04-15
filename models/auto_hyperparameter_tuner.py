import random
import numpy as np
import tempfile
import ray
import torch
import torch.optim as optim
import wandb
import ray
import os
import pickle
import numpy as np
import torch
import time
import re

from filelock import FileLock

from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.train import Checkpoint
from ray.tune.search.hebo import HEBOSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch import nn
from torch_geometric.utils import add_self_loops
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from data_processors.data_splitter import DataSplitter    


from models.STGAT import * 
from models.PCR import * 
from models.NN import *
from models.STTR import *
from models.Static_STGAT import *
from models.MultiLayerNN import *
from models.GAT import *

def get_highest_existing_accuracy(directory):
    highest_accuracy = 0.0
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            match = re.search(r'_(\d+\.\d+)\.pth$', filename)
            if match:
                accuracy = float(match.group(1))
                highest_accuracy = max(highest_accuracy, accuracy)
    return highest_accuracy

class ModelTrainer:
    def __init__(self, param_space):
        self.param_space = param_space
        self.num_samples = param_space['num_samples']
        self.wandb_project = param_space['wandb_project']
        self.wandb_api_key = param_space['wandb_api_key']
        self.computed_params = {}
        self.best_model_path = ""
        self.highest_test_acc = 0
    
    def initialize_model(self, config, device, num_nodes, X_train=None):
        model_name = config.get("Architecture")
        params = self.computed_params
        num_classes, spatial_in_features, lstm_input_dim, temporal_hidden_dim, temporal_layer_dimension, num_epochs = (
            params['num_classes'], params['spatial_in_features'], 
            params['lstm_input_dim'], params['temporal_hidden_dim'], params['temporal_layer_dimension'], 
            params['num_epochs']
        )
    
        if model_name == 'Static_STGAT':
            use_auto_corr_matrix = config.get("use_auto_corr_matrix", False)
            
            if use_auto_corr_matrix:
                if X_train is None:
                    raise ValueError("X_train must be provided when use_auto_corr_matrix is True")
                
                # Reshape X_train before computing the auto-correlation matrix
                X_train_reshaped = X_train.reshape(-1, num_nodes)
                
                # Convert X_train_reshaped to a numeric array
                X_train_reshaped = np.asarray(X_train_reshaped, dtype=np.float32)
                
                # Replace all NaN values with zeros
                X_train_reshaped[np.isnan(X_train_reshaped)] = 0
                
                # Compute the auto-correlation matrix using absolute values
                auto_corr_matrix = np.abs(np.corrcoef(X_train_reshaped.T))
                
                # Replace NaN values in the auto-correlation matrix with zeros
                auto_corr_matrix[np.isnan(auto_corr_matrix)] = 0
                
                # Print the auto-correlation matrix right after computation
                #print("Auto-correlation matrix after computation:")
                #print(auto_corr_matrix)
                
                # Check for NaNs in the auto-correlation matrix
                if np.isnan(auto_corr_matrix).any():
                    raise ValueError("Auto-correlation matrix contains NaNs")
                
                # Check the shape of the auto-correlation matrix
                if auto_corr_matrix.shape != (num_nodes, num_nodes):
                    raise ValueError(f"Auto-correlation matrix should have shape (num_nodes, num_nodes), but got {auto_corr_matrix.shape}")
                
                model = Static_STGAT(spatial_in_features, config["spatial_hidden_dim"], config["spatial_out_features"],
                                     num_classes, num_nodes, config["edge_threshold"], temporal_hidden_dim, 
                                     temporal_layer_dimension, config["graph_batch_size"], auto_corr_matrix)
            else:
                model = Static_STGAT(spatial_in_features, config["spatial_hidden_dim"], config["spatial_out_features"],
                                     num_classes, num_nodes, config["edge_threshold"], temporal_hidden_dim, 
                                     temporal_layer_dimension, config["graph_batch_size"])
            
            self.graph_optimizer = Adam([model.V_Adap], lr=config["graph_lr"])  # Define the graph_optimizer attribute
            return model
        
        elif model_name == 'LSTM':
            model = LSTM(input_dim, config["spatial_hidden_dim"], layer_dim, ouput_dim)
            return model
        elif model_name == 'GAT':
            model = GAT(spatial_in_features, config["spatial_hidden_dim"], num_classes, num_nodes, config["edge_threshold"], auto_corr_matrix)
            return model

        elif model_name == 'PCR':
            model = PCRModel(n_components=config.get("n_components", 10))
            return model
        
        elif model_name == 'NN':
            model = NNModel(input_dim, config["hidden_dim"], config["output_dim"])
            return model
        
        elif model_name == 'MLNN':
            input_dim = num_nodes * spatial_in_features
            hidden_dim = config["hidden_dim"]
            num_layers = config["num_layers"]
            output_dim = num_classes
            model = MLNNModel(input_dim, hidden_dim, num_layers, output_dim)
            return model
        
        elif model_name == 'Transformer':
            model = TransformerTemporalLayer(
            original_input_dim=config["original_input_dim"],
            transformer_input_dim=config["transformer_input_dim"],  
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            output_dim=config["output_dim"]
            )
            return model

        elif model_name == "ST-TR":
            use_auto_corr_matrix = config.get("use_auto_corr_matrix", False)

            auto_corr_matrix = None
            if use_auto_corr_matrix:
                if X_train is None:
                    raise ValueError("X_train must be provided when use_auto_corr_matrix is True")

                # Reshape X_train before computing the auto-correlation matrix
                X_train_reshaped = X_train.reshape(-1, num_nodes)

                # Convert X_train_reshaped to a numeric array and replace NaNs with zeros
                X_train_reshaped = np.asarray(X_train_reshaped, dtype=np.float32)
                X_train_reshaped[np.isnan(X_train_reshaped)] = 0

                # Compute the auto-correlation matrix and handle NaNs
                auto_corr_matrix = np.abs(np.corrcoef(X_train_reshaped.T))
                auto_corr_matrix[np.isnan(auto_corr_matrix)] = 0

                # Ensure the auto-correlation matrix is valid
                if np.isnan(auto_corr_matrix).any() or auto_corr_matrix.shape != (num_nodes, num_nodes):
                    raise ValueError("Invalid auto-correlation matrix.")

            model = Static_STTR(
                spatial_in_features=spatial_in_features, 
                spatial_hidden_dim=config["spatial_hidden_dim"], 
                spatial_out_features=config["spatial_out_features"],
                num_classes=num_classes, 
                num_nodes=num_nodes, 
                edge_threshold=config["edge_threshold"],
                original_input_dim=config["original_input_dim"],
                transformer_input_dim=config["transformer_input_dim"], 
                num_heads=config["num_heads"], 
                num_layers=config["num_layers"], 
                output_dim=config["output_dim"],
                auto_corr_matrix=auto_corr_matrix 
            )
            self.graph_optimizer = Adam([model.V_Adap], lr=config["graph_lr"])

            return model

        else:
            raise ValueError(f"Unrecognized model architecture: {model_name}")

    def save_model_weights(self, model, test_acc, config):
            if test_acc > 80:
                model_name = config['Architecture']
                checkpoint_dir = f"checkpoints/{model_name}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_{test_acc:.2f}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model weights saved to {checkpoint_path}")

        
    def train_model(self, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = config.get("Architecture")
        
        wandb = setup_wandb(config, project=self.wandb_project, api_key=self.wandb_api_key)
        should_checkpoint = config.get("should_checkpoint", False)
        
        mouse_number = config['mouse_number']
        timesteps = config['timesteps']

        spike_trains_file_path = config['file_path']
        #spike_trains_file_path = '/proj/STOR/pipiras/Neuropixel/output/spike_trains_with_stimulus_session_732592105_10.pkl'

        #spike_trains_file_path = f'/nas/longleaf/home/rayrayc/output/spike_trains_with_stimulus_session_719161530_10.pkl'

        with open(spike_trains_file_path, 'rb') as f:
                spike_df = pickle.load(f)
                print(f"Loaded spike trains dataset: {type(spike_df)}")

        # Split data into train, test, and validation sets. 
        data_splitter = DataSplitter(dataframe = spike_df, batch_size = config['batch_size'])

        # Access the data loaders.
        ### Redo the code to combine the trains into the loader.
        X_train = data_splitter.X_train
        X_test = data_splitter.X_test
        #X_val = data_splitter.X_val
        y_train = data_splitter.y_train
        y_test = data_splitter.y_test
        #y_val = data_splitter.y_val
        train_loader = data_splitter.train_loader
        test_loader = data_splitter.test_loader
        #val_loader = data_splitter.val_loader
        X = data_splitter.X
        y = data_splitter.y

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
        #print("Shape of edge_index:", self.computed_params['edge_index'].shape)
        #print("Content of edge_index:", self.computed_params['edge_index'])

        self.computed_params['temporal_output_dim'] = int(len(np.unique(y_train)))
        self.computed_params['temporal_hidden_dim'] = int(config["temporal_hidden_dim"])
        self.computed_params['temporal_layer_dimension'] = int(config["temporal_layer_dimension"])
        self.computed_params['num_epochs'] = int(config['num_epochs'])
        self.computed_params['label_encoder'] = LabelEncoder().fit(y_train.ravel())
        config.update(self.computed_params)
        num_nodes = X_train.shape[2]  # Assuming X_train has shape (batch_size, num_timesteps, num_nodes, num_features)
        #print(f"num_nodes: {num_nodes}")
        #print(f"X_train shape: {X_train.shape}")
        use_auto_corr_matrix = config.get("use_auto_corr_matrix", False)

        if use_auto_corr_matrix:
            X_train_reshaped = X_train.reshape(-1, num_nodes)
            X_train_reshaped = np.asarray(X_train_reshaped, dtype=np.float32)
            
            # Replace NaN columns with zeros
            nan_columns = np.where(np.isnan(X_train_reshaped).any(axis=0))[0]
            X_train_reshaped[:, nan_columns] = 0
            
            model = self.initialize_model(config, device, num_nodes, X_train_reshaped)
        else:
            model = self.initialize_model(config, device, num_nodes)
        model.to(device)
    
        criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizers conditionally
        if model_name == "ST-TR":
            stgat_optimizer = Adam(model.static_stgat.parameters(), lr=config["stgat_lr"])
            transformer_optimizer = Adam(model.transformer.parameters(), lr=config["transformer_lr"])
        else:
            optimizer = Adam(model.parameters(), lr=config["lr"])

        # Initialize learning rate scheduler for the non-STTR optimizer
        if model_name != "ST-TR":
            scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

        
        scaler = GradScaler()
    
        
        from torch_geometric.data import NeighborSampler

        highest_test_acc = 0
        num_epochs = int(config['num_epochs'])
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0  # Initialize running_loss before the loop
            correct_train = 0
            total_train = 0
            batch_count = 0  # Initialize batch_count before the loop
            start_time = time.time()  # Initialize start_time at the beginning of each epoch
        
            total_batches = len(train_loader)
            update_interval = max(1, total_batches // 10)  # Update every 10% or at least every batch
        
            # Initialize tqdm with manual control by setting mininterval to a large number
            progress_bar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}', mininterval=1e9)
        
            accumulation_steps = config["accumulation_steps"]  # Number of steps to accumulate gradients

            #Reset zero gradients based on architecture 
            if model_name == "ST-TR":
                stgat_optimizer.zero_grad()
                transformer_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)
                labels = labels.squeeze().long()
        
                # Generate the dynamic adjacency matrix for the current batch
                V_Adap_batch = model.V_Adap.expand(features.shape[0], -1, -1)  # Expand V_Adap to match batch size
                edge_index, edge_attr = model.dynamic_adjacency(V_Adap_batch)
                edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)   
                # Transform labels to class indices
                class_idx = self.computed_params['label_encoder'].transform(labels.cpu().numpy())
                class_idx = torch.tensor(class_idx, dtype=torch.long).to(device)
        
                # In the training loop, pass the edge_index to the model
                outputs = model(features)
                
                #print(f"Output shape: {outputs.shape}, Target shape: {labels.shape}")
                outputs_last_step = outputs[:, :]  # shape: [batches, 119]
                
                loss = criterion(outputs_last_step, labels)
                    
                loss = loss / accumulation_steps  # Scale the loss to account for accumulation
                
                
                self.graph_optimizer.zero_grad()
                
                
                loss.backward()
                
                if model_name == "ST-TR":
                    torch.nn.utils.clip_grad_norm_(model.static_stgat.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(model.transformer.parameters(), max_norm=1.0)
                    stgat_optimizer.step()
                    transformer_optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    self.graph_optimizer.step()
                    optimizer.step()
                
        
                running_loss += loss.item()  # Update running_loss
                _, predicted = torch.max(outputs_last_step.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                features.cpu()
                labels.cpu()
                batch_count += 1
        
                # Manually update the progress bar every 10% or at the end of the epoch
                '''
                if batch_idx % update_interval == 0 or batch_idx == total_batches - 1:
                    progress_bar.update(batch_idx - progress_bar.n)  # Update progress bar to the current batch index
                    progress_bar.set_postfix({'Loss': running_loss / (batch_count + 1), 'Train Acc': correct_train / total_train * 100})
                    sliver_indices, sliver_weights = model.dynamic_adjacency.get_edge_sliver(model.V_Adap)
                    edge_indices, edge_weights = model.dynamic_adjacency(model.V_Adap)
                    num_connections = edge_indices.size(1)
                    #print(f"Total connections: {num_connections}")
                    #print("Weights:", sliver_weights)
                '''
            #print(f'output.shape = {np.shape(outputs_last_step)}')
            #print(f'output = {outputs_last_step}')
            #print(f'predicted_indices.shape: {np.shape(predicted)}')
            #print(f'predicted_indices = {predicted}')
            #print(f'labels shape = {np.shape(labels)}')
            #print(f'labels = {labels}')
            
            progress_bar.set_postfix({'Loss': running_loss / (batch_count + 1), 'Train Acc': correct_train / total_train * 100})

            #### Close the progress bar at the end of the epoch
            progress_bar.close()
        
            train_acc = 100 * correct_train / total_train
            model.eval()  # Set the model to evaluation mode
            correct_test = 0
            total_test = 0
            correct_test = 0
            total_test = 0
            total_batches = len(test_loader)
            update_interval = max(1, total_batches // 4)  # Update every 25% or at least every batch
            
            # Initialize tqdm with manual control by setting mininterval to a large number
            test_progress_bar = tqdm(test_loader, desc=f'Test Epoch {epoch+1}/{num_epochs}', mininterval=1e9)
            
            with torch.no_grad():
                for batch_idx, (features, labels) in enumerate(test_loader):
                    features, labels = features.to(device), labels.to(device)
                    labels = labels.squeeze().long()
                    class_idx = self.computed_params['label_encoder'].transform(labels.cpu().numpy())  # Use all class indices
                    class_idx = torch.tensor(class_idx, dtype=torch.long).to(device)
                    outputs = model(features)  # Call the model with the features and class index
                    predicted_indices = torch.argmax(outputs, dim=1)
                    total_test += labels.size(0)
                    #print(f'output.shape = {np.shape(outputs)}')
                    #print(f'output = {outputs}')
                    #print(f'predicted_indices.shape: {np.shape(predicted_indices)}')
                    #print(f'predicted_indices = {predicted_indices}')
                    #print(f'labels shape = {np.shape(labels)}')
                    #print(f'labels = {labels}')

                    correct_test += (predicted_indices == labels).sum().item()
                    
                    features.cpu()
                    labels.cpu()
            
                    # Manually update the progress bar every 25% or at the end of the epoch
                    '''
                    if batch_idx % update_interval == 0 or batch_idx == total_batches - 1:
                        test_progress_bar.update(batch_idx - test_progress_bar.n)  # Update progress bar to the current batch index
                        test_progress_bar.set_postfix({'Test Acc': correct_test / total_test * 100})
                    '''
            
            # Close the progress bar at the end of the test loop
            test_progress_bar.close()
            
            test_acc = 100 * correct_test / total_test
            


            best_model_path = ""
            
            # Update highest test accuracy
            if test_acc > self.highest_test_acc:
                self.highest_test_acc = test_acc
            
                base_saved_models_dir = "/proj/STOR/pipiras/Neuropixel/saved_models"

                # Append model architecture to directory path
                saved_models_dir = os.path.join(base_saved_models_dir, config['Architecture'])

                os.makedirs(saved_models_dir, exist_ok=True)

                os.chmod(saved_models_dir, 0o755)  # Grant read, write, and execute permissions
                os.chown(saved_models_dir, os.getuid(), os.getgid())  # Change ownership to current user
            
                # Check for existing models in directory
                existing_models = [f for f in os.listdir(saved_models_dir) if f.endswith(".pth")]
            
                # Get highest accuracy from existing models
                highest_existing_acc = 0.0
                for model_file in existing_models:
                    match = re.search(r'_(\d+\.\d+)\.pth$', model_file)
                    if match:
                        acc = float(match.group(1))
                        highest_existing_acc = max(highest_existing_acc, acc)
            
                # Save model if the current accuracy higher 
                if test_acc > highest_existing_acc:
                    best_model_path = f"best_model_{config['Architecture']}_{config['mouse_number']}_{test_acc:.2f}.pth"
                    model_save_path = os.path.join(saved_models_dir, best_model_path)
                    torch.save(model.state_dict(), model_save_path)
                    print(f"New highest test accuracy: {test_acc:.2f}%. Model saved to {model_save_path}")
                else:
                    print(f"Test accuracy {test_acc:.2f}% is not higher than the existing highest accuracy {highest_existing_acc:.2f}%. Model not saved.")
                    
                    
            if model_name != "ST-TR":
                    scheduler.step()
                    
            end_time = time.time()  # End time of the epoch
            epoch_duration = end_time - start_time
            wandb.log({
                "Epoch": epoch, 
                "Loss": running_loss, 
                "Train Accuracy": train_acc, 
                "Test Accuracy": test_acc, 
                "Epoch Duration": epoch_duration}
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
            # After training, print and save a sliver of the edge weights
            edge_indices, edge_weights = model.dynamic_adjacency(model.V_Adap)
            num_connections = edge_indices.size(1)
            print(f"Total connections: {num_connections}")
            sliver_indices, sliver_weights = model.dynamic_adjacency.get_edge_sliver(model.V_Adap)
            print("Edge sliver weights:", sliver_weights)
            

        
        print(f'Highest Test Accuracy: {self.highest_test_acc}%')
        # After training, print and save a sliver of the edge weights.
        sliver_indices, sliver_weights = model.dynamic_adjacency.get_edge_sliver(model.V_Adap)
        print("Edge sliver indices:", sliver_indices)
        print("Edge sliver weights:", sliver_weights)
                
        print(f'Highest Test Accuracy: {self.highest_test_acc}%')
        self.save_model_weights(model, self.highest_test_acc, config)
        train.report({'test_acc': self.highest_test_acc})  # Report the highest test accuracy
            
    def execute_tuning(self):
        scheduler = ASHAScheduler(max_t=100, grace_period=5, reduction_factor=2, brackets=3)
        resources_per_trial = {"cpu": 256, "gpu": 1}    
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
        
    
    def load_model(self, model_path, config, num_nodes, num_classes, spatial_in_features):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.computed_params['num_classes'] = num_classes
        self.computed_params['spatial_in_features'] = spatial_in_features
        self.computed_params['num_nodes'] = num_nodes

        self.computed_params.setdefault('lstm_input_dim', spatial_in_features * num_nodes)
        self.computed_params.setdefault('temporal_hidden_dim', config.get('temporal_hidden_dim', 128))
        self.computed_params.setdefault('temporal_layer_dimension', config.get('temporal_layer_dimension', 1))
        self.computed_params.setdefault('num_epochs', config.get('num_epochs', 10))

        model = self.initialize_model(config, device, num_nodes)

        # Load the state dictionary from the checkpoint
        state_dict = torch.load(model_path, map_location=device)

        # Filter the state dictionary to keep only matching keys
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}

        # Load the filtered state dictionary into the model
        model.load_state_dict(filtered_state_dict, strict=False)

        model.to(device)
        return model
    
    def test_loaded_model(self, model, new_mouse_loader, num_nodes, num_classes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()  
        correct = 0
        total = 0

        print(f"Testing the loaded model on new mouse data...")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of classes: {num_classes}")
        print(f"Input data shape: {next(iter(new_mouse_loader))[0].shape}")

        with torch.no_grad():
            for data, labels in new_mouse_loader:
                data, labels = data.to(device), labels.to(device)
                labels = labels.squeeze().long()

                # print(f"Input data shape: {data.shape}")
                # print(f"Labels shape: {labels.shape}")

                outputs = model(data)

                # print(f"Output shape: {outputs.shape}")

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # print(f"Predicted labels: {predicted}")
                # print(f"True labels: {labels}")
                # print(f"Correct predictions: {(predicted == labels).sum().item()} out of {labels.size(0)}")
                # print("---")

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the new mouse data: {accuracy:.2f}%')

    
        

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
