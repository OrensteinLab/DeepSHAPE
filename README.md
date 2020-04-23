# DeepSHAPE 
Command line for training and testing<br />
python main.py train \<(1,1)\\(1,0)\\(0,1)\> \<binary_crossentropy\mse\> \<in_vitro\in_vivo\> \<training sequences file\>  \<training RNAplfold file\> \<annotation file\><br />
python main.py predict \<(1,1)\\(1,0)\\(0,1)\> \<binary_crossentropy\mse\> \<in_vitro\in_vivo\> \<path to testing sequences file\>  \<path to testing RNAplfold file\> \<path to testing annotation file\><br />

Inputs:
- \<(1,1)\\(1,0)\\(0,1)\> - Controls x data selection (add_sequences, add_RNAplfold)
  - "(1,1)" - Select both sequence data and RNAplfold data as an input to the network
  - "(1,0)" - Select only sequence data as an input to the network
  - "(0,1)" - Select only RNAplfold data as an input to the network
- \<binary_crossentropy\mse\> - Controls lost function selection and and its corresponding activation function of the last layer
  - "binary_crossentropy" (and sigmoid function)
  - "mse" (and linear function)
- \<in_vitro\in_vivo\> - Controls dataset selection 
  - "in_vitro"- In vitro dataset
  - "in_vitro" - In vivo dataset
- \<training sequences file\> - Path + name of training sequences file
- \<training RNAplfold file\> - Path + name of RNAplfold file
 - \<annotation file\> - Path + name of SHAPE file

Outputs:
- The trained models will be saved under the directory results/saved_models
-  The network details, SHAPE predictions and performances will be saved under the directory results/net_details_predictions_performances
