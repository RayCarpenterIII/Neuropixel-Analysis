
# Neuropixel Analysis 
Our study sought to survey and identify machine learning models and deep learning architectures
capable of predicting visual stimuli based on the spike patterns of single neurons. Drawing
insights from neuronal encoding is integral for understanding the neural mechanisms underlying
cognitive processes, aiding the development of brain-computer interfaces, and providing models
that may be useful for classifying disease or predicting functionalities. We worked with
Neuropixel data from the Allen Brain Observatory [1,2] consisting of the firing rates from single
neurons, referred to as units, from several male mice's visual cortex, thalamus, and hippocampus.
Each recording involved around 2,000 separate units, which were used as the input variables for
all models. The mice were shown 118 different natural images of predators, foliage, and other
scenes from their natural habitat at random in repetition and for 250 ms each. The image
numbers associated with them were used as the variables to predict. Three of the 32 mice with
data recorded were chosen for initial model selection, benchmarking, and hyper-parameter range
searches. Models that scored the highest were tested across the rest of the 32 mice utilizing the
found hyper-parameter ranges.

We assessed the prediction performance on test data for various machine and deep learning
architectures built on training data. A random guess was associated with a baseline test accuracy
of 1/118 = 0.85%. Multivariate logistic regression(MVLR) returned an accuracy of 91.76%.
Next, Principal Component Analysis was performed, followed by MVLR (PC-MVLR) on the
components. The number of principal components was equal to the number of original variables.
This model achieved the second-highest accuracy of 95.29%. Support Vector Classifiers with
linear, polynomial, and radial kernels achieved an accuracy of 92.52%. A single-hidden layer
neural network (NN) on aggregate firing rates over the length of a visual stimulus resulted in a
test accuracy of 93%. To test the utility of spatial modeling, a single-layer Graph Convolution
Network (GCN) and a graph attention (GAT) network were tried with 48% and 91.9%
accuracies, respectively. Each spatial model was tested with three configurations: a fully
connected graph, a graph created from cross-correlation with a 25% threshold, and a
backpropagation-optimized graph starting with full connections. Multiple models were tested to
consider the temporal aspect of the data, whereby the firing rates were presented in a sequence
over time bins during a visual stimulus. Consisting of a Long-Short Term Memory (LSTM)
model, Spatio-Temporal Graph Attention Network, and a Transformer. The LSTM produced the
highest test accuracy at 97.73%, the ST-GAT achieved 92.4%, and the Transformer achieved
93% test accuracy. We tested each temporal model by shifting each 250ms interval into bins of 1,
2, 3, 5, 10, and 20. Gains in accuracy are maximized at three to five bins. Based on the mouse,
results would range from 51 - 97.7% accuracy when implementing the LSTM.

Our results provide insights into the utility of different learning architectures in prediction tasks
using neural data at the micro-level. We have found that deep learning architectures with fewer
hidden layers consistently demonstrated higher test accuracies for all models. The success of
PC-MVLR and the LSTM demonstrates how dimensionality reduction and temporal information
can improve model performance.

References
1. Allen Brain Observatory. Neuropixel[Poster.pdf](https://github.com/user-attachments/files/16473154/Poster.pdf)
s Visual Coding.
https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels
2. Paulk AC, Kfir Y, Khanna AR et al. Large-scale recording with single neuron resolution using
Neuropixels probes in human cortex. Nature Neuroscience, 2022, 25: 252-263.
3. Wein S, Schuller A, Tome AM, et al. Forecasting brain activity based on models of
spatiotemporal brain dynamics: A comparison of graph neural network architectures. Network
Neuroscience, 2022, 6 (3): 665–701.
4. Ray Tune: Hyperparameter Tuning. https://docs.ray.io/


<p align="center">
  <img src="https://github.com/RayCarpenterIII/Neuropixel-Analysis/assets/106690201/6207ea50-61c8-44c7-8da7-29b8fa157cf0" style="width: 45%; height: auto; display: inline-block;">
  <img src="https://github.com/RayCarpenterIII/Neuropixel-Analysis/raw/main/pictures/Poster.jpeg" style="width: 45%; height: auto; display: inline-block;">
</p>

[Poster PDF](https://github.com/RayCarpenterIII/Neuropixel-Analysis/raw/main/pictures/Poster.pdf)

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
├── test_across.ipynb          <- test models on separate mice
├── visualize_data.ipynb       <- visualize neural activity data
```

## Data 
The open data is from the Ecephys Project, which employs Neuropixel technology to record large-scale neural activity in mice. The program contained is built to pull the data directly from the Ecephys project, process the data, create visualizations, and run the models.

- **Raw Data:** https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
