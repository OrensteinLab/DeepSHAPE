# DeepSHAPE 
Command line for training and testing<br />
python main.py train \<(1,1)\\(1,0)\\(0,1)\> \<binary_crossentropy\mse\> \<in_vitro\in_vivo\> \<training sequences file\>  \<training RNAplfold file\> \<training annotation file\><br />
python main.py predict \<(1,1)\\(1,0)\\(0,1)\> \<binary_crossentropy\mse\> \<in_vitro\in_vivo\> \<testing sequences file\>  \<testing RNAplfold file\> \<testing annotation file\><br />

Inputs:
  - \<train\predict\> - Train network or predict by trained network
    - "train" - Train DeepSHAPE network.
    - "predict" - Predict icSHAPE rectivity scores using DeepSHAPE trained netork.
  - \<(1,1)\\(1,0)\\(0,1)\> - Controls x data selection (add_sequences, add_RNAplfold)
    - "(1,1)" - Select both sequence data and RNAplfold data as an input to the network
    - "(1,0)" - Select only sequence data as an input to the network
    - "(0,1)" - Select only RNAplfold data as an input to the network
  - \<binary_crossentropy\mse\> - Controls loss function selection and and its corresponding activation function of the last layer
    - "binary_crossentropy" - Binary cross entropy loss function and sigmoid activation function of the last layer.
    - "mse" - Mean squared error loss function and linear activation function of the last layer.
  - \<in_vitro\in_vivo\> - Controls dataset selection 
    - "in_vitro" - In vitro dataset
    - "in_vitro" - In vivo dataset <br />

For training the model:<br />
  - \<training sequences file\> - Path to training sequences file
  - \<training RNAplfold file\> - Path to training RNAplfold file
  - \<training annotation file\> - Path to training SHAPE file <br /> 

For testing the model:<br />
  - \<testing sequences file\> - Path to testing sequences file
  - \<testing RNAplfold file\> - Path to testing RNAplfold file
  - \<testing annotation file\> - Path to testing icSHAPE file

Outputs:
- The trained models will be saved under the directory results/saved_models
-  The network details, SHAPE predictions and performances will be saved under the directory results/net_details_predictions_performances
