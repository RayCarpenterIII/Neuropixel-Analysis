
# Neuropixel Analysis 
Our project aims to predict visual stimuli presented to mice based on the firing rates of single neurons within their visual cortex by surveying how different machine and deep learning models predict stimuli based on neural activity at the micro-scale. This project is currently being worked on in an OnDemand Jupyter notebook connected to UNC Research computing's longleaf cluster. Changes on the Jupyter Notebook are periodically updated to Git.

We created a pipeline that pulls the data from an open-sourced project by the Allen Brain Observatory, visualizes it, automatically tunes deep learning models on the data, and builds an
adjacency matrix representing the functional connectomics between single neurons. 

We've benchmarked several statistical, machine learning, and deep learning techniques to predict what the mouse sees. Given 118 different images of natural scenes and around 2,000 neurons per mouse, we have been able to predict the correct scene with 95% accuracy using a type of Spatio-Temporal Graph Neural Network (ST-GNN) called a Spatio-Temporal Graph Attention Network(ST-GAT). In addition to having the highest prediction accuracy, this model can produce a directed adjacency matrix between nodes that represents how one neuron may affect another through backpropagation. The ST-GAT architecture and its variants are outlined in the figure below. The code also allows for the leveraging of a distributed system. We are currently using the Longleaf Computing Cluster at UNC.

<p align="center">
  <img src="https://github.com/RayCarpenterIII/Neuropixel-Analysis/assets/106690201/6207ea50-61c8-44c7-8da7-29b8fa157cf0" width="60%">
</p>

We are implementing Ray Tune and Pytorch for automatic model optimization and parallel computing while recording our run data at Wandb.ai. 

## Project Structure 
```
├── data_processors            <- directory for project data processing
    ├── data_splitter.py       <- process tables of predictors
    ├── load_processed_data.py <- load pre-processed data
    ├── pull_and_process_data  <- pull data from AllenSDK and pre-process
├── models
  ├── stgnn.py                 <- train and evaluate stgnn on input data
  ├── static_stgnn.py          <- train, evaluate static stgnn
├── pictures                   <- flow charts for ST-GAT and ST-GNN  
├── test                       <- function testing directory
├── README.md                  <- description of project 
├── model_predictions.ipynb    <- build and run model predictions
├── neural_image_testing.ipynb <- test creation of neural images
├── requirements.txt           <- install dependencies (wip)
├── visualize_data.ipynb       <- visualize neural activity data
```

## Data 
The open data is from the Ecephys Project, which employs Neuropixel technology to record large-scale neural activity in mice. The program contained is built to pull the data directly from the Ecephys project, process the data, create visualizations, and run the models.

- **Raw Data:** https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
