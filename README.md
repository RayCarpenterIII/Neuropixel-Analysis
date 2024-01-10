
# Neuropixel Analysis 
This project aims to predict the visual stimulus presented to mice based on the firing rates of their neurons.

By utilizing a Spatial-Temporal Graph Neural Network we will train an adjacency matrix that may pick up on the directed functional Connectomics between the single neurons when the mouse is shown different images.

To test this we will compare the prediction accuracies of different models. Some do not utilize spatial information, some use a correlation matrix as the spatial information, and our proposed model utilizes adjacency matrices produced to show the direction of connectivity between each neuron for each image.

## Project Structure 
```
├── README.md                  <- description of project 
├── model_predictions.ipynb    <- build and run model predictions
├── visualize_data.ipynb       <- visualize neural activity data
├── neural_image_testing.ipynb <- test creation of neural images
├── data                       <- directory for project data processing
    ├── load_processed_data.py <- load pre-processed data
    ├── processed              <- pull data from AllenSDK and pre-process
├── models
  ├── stgnn.py                 <- train and evaluate stgnn on input data
  ├── static_stgnn.py          <- train, evaluate static stgnn    
├── test                       <- function testing directory 
```

## Data 
The open data is from the Ecephys Project, which employs Neuropixel technology to record large-scale neural activity in mice. The program contained is built to pull the data directly from the Ecephys project, process the data, create visualizations, and run the models.

- **Raw Data:** https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
