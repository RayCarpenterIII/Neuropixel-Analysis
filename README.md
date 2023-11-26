
# Neuropixel Analysis 
  The goal of this project is to predict the visual stimulus presented to mice based on the firing rates of their neurons. 
  
  


- Mapping a Representation of Directed Functional Connectomics between Single Neurons to Predict Visual Perception in Mice
  
  By utilizing a Spatial-Temporal Graph Neural Network we will train an adjacency matrix that may pick up on the directed functional Connectomics between the single neurons when the mouse is shown different images. To test this we will compare the prediction accuracies of different models. Some do not utilize spatial information, some use a correlation matrix as the spatial information, and our proposed model utilizes a set of adjacency matrices that are produced to show the direction between each neuron for each image. The open data is from the Ecephys Project, which employs Neuropixel technology to record large-scale neural activity in mice. The program contained is built to pull the data directly from the Ecephys project, process the data, create visualizations, and run the models. This is still a work in progress. A paper will be produced and linked here when it is complete.


# Project Structure 
```
├── README.md                  <- description of project 
├── stimulus_predictions.ipynb <- build and run model predictions
├── neural_image_testing.ipynb <- test creation of neural images
├── ProcessData                <- directory for project data processing
    ├── load_processed_data.py <- load pre-processed data
    ├── processed              <- pull data from AllenSDK and pre-process
├── test                       <- function testing directory 
```

- **Raw Data:** https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
- **Program in works:** https://github.com/RayCarpenterIII/Neuropixel-Analysis/blob/refactor-repository/stimulus_predictions.ipynb