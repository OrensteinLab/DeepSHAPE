# DeepSHAPE 
A deep neural network for predicting SHAPE reactivity scores from high-throughput RNA probing data.

Requirements:
  - Python version 3.6.8
  - TensorFlow version 1.13.1
  - keras version 2.2.4
  - numpy version 1.16.2

Command line for training and testing:<br />
python main.py <train/test>
<sequences/RNAplfold/both> <binary_crossentropy/mse> <in_vitro/in_vivo>
<training/testing sequences file> <training/testing RNAplfold file>
<training/testing annotation file><br />

Inputs:
  - \<train\test\> - Train or test the network
    - "train" - Train DeepSHAPE network.
    - "test" - Predict icSHAPE rectivity scores using DeepSHAPE trained netork.
  - \<sequences\\RNAplfold\\both\> - Controls the input data type
    - "sequences" - Select only sequence data as an input to the network
    - "RNAplfold" - Select only RNAplfold data as an input to the network
    - "both"      - Select both sequence data and RNAplfold data as an input to the network
  - \<binary_crossentropy\mse\> - Controls loss function selection and and its corresponding activation function of the last layer
    - "binary_crossentropy" - Binary cross entropy loss function and sigmoid activation function of the last layer.
    - "mse" - Mean squared error loss function and linear activation function of the last layer.
  - \<in_vitro\in_vivo\> - Controls dataset selection 
    - "in_vitro" - In vitro dataset
    - "in_vitro" - In vivo dataset 

  - \<training/testing sequences file\> - Path to training sequences file
  - \<training/testing RNAplfold file\> - Path to training RNAplfold file
  - \<training/testing annotation file\> - Path to training SHAPE file <br />
Note - 	It is required to provide three valid input files regardless of the chosen input data type.<br /> <br /> 

Outputs:
  - Training Outputs stored in outputs/saved_models/
    - log.txt
    - .hdf5 - Trained network for each epoch
    - acc_history.txt - Accuracy value after each epoch
    - loss_history.txt - Loss value after each epoch
  - Testing Outputs stored in outputs/test_results/
    - log.txt
    - predictions_.txt - SHAPE predictions (one file per epoch)
    - performance.txt - Performance        
Note - The outputs will be stored in a path that indicates the chosen training configuration under 'outputs' folder, i.e - (1)Dataset selection (2) input data type selection (3) loss function selection
