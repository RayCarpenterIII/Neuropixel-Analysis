
# Neuropixel Analysis 
This project aims to predict the visual stimulus presented to mice based on the firing rates of their neurons and produce adjacency matrices that describe the directed functional connectomics between single neurons.

To do this, we test an array of statistical, ML, and DL algorithms to test which best predicts a mouse's visual stimuli given open-sourced Neuropixel data pulled from the API by the Allen Brain Institute.

Using a Spatial-Temporal Graph Neural Network, we can train adjacency matrices that may pick a representation of the directed functional Connectomics between single neurons at a specific class, interval, or across time. This model takes in temporal graph data with node and edge information, makes predictions, and can update the edge information through backpropagation. This model can be used on brain data at the micro, meso, and macro level. It can also be used on data that has a graph structure and temporal components. This can be implemented with traffic data and social network data. 

Our current models are able to achieve a prediction accuracy of up to 95%. We are implementing Ray Tune and Pytorch for automatic model optimization and parallel computing while recording our run data at Wandb.ai. 

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
