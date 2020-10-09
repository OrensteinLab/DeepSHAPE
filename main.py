#Example script to run this program
#"train" "both" "binary_crossentropy" "in_vitro" "C:\Users\Yifat\Desktop\deepSHAPE\Thesis_Part1_Oct2020\partial_data\3_invitro\train\sequences3.txt" "C:\Users\Yifat\Desktop\deepSHAPE\Thesis_Part1_Oct2020\partial_data\3_invitro\train\sequences3.txt.RNAplfold" "C:\Users\Yifat\Desktop\deepSHAPE\Thesis_Part1_Oct2020\partial_data\3_invitro\train\annotations3.txt"
#"test" "both" "binary_crossentropy" "in_vitro" "C:\Users\Yifat\Desktop\deepSHAPE\Thesis_Part1_Oct2020\partial_data\3_invitro\test\sequences3.txt" "C:\Users\Yifat\Desktop\deepSHAPE\Thesis_Part1_Oct2020\partial_data\3_invitro\test\sequences3.txt.RNAplfold" "C:\Users\Yifat\Desktop\deepSHAPE\Thesis_Part1_Oct2020\partial_data\3_invitro\test\annotations3.txt"
###################################################################################################################################################################################################################################################################################
from utility_funcs import *
from model import *
import time
import glob
import sys
import os
import numpy as np
import tensorflow as tf
############################################################################## Use these imports for local ####################################################################
#  - Python version 3.6.8
#  - TensorFlow version 1.13.1
#  - keras version 2.2.4
#  - numpy version 1.16.2
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
from tensorflow.keras.optimizers import *
# Random generator initializers
tf.set_random_seed(1)
np.random.seed(1)
##############################################################################  Use these imports for intel cloud ############################################################
#from keras.models import load_model
#from keras import losses
#from keras.optimizers import *
##Random generator initializers
#tf.random.set_seed(1)
#np.random.seed(1)
###############################################################################################################################################################
# Constants
NEIGHBOURS_VEC = np.array([20,40,80,120,160]) # Number of neighbors from both flanks symmetrically ( (window size-1) )
EPOCHS_        = 3
LR_            = 0.001
DECAY_         = 1e-5
OPTIMIZER_     = Adam(lr = LR_, decay = DECAY_)
BATCH_         = 16
###############################################################################################################################################################
# Read the user inputs
if len(sys.argv) != 8:
    sys.exit(printUsage())
directive       = sys.argv[1] # Directive selection <train/predict>
x_data_         = sys.argv[2] # Input data type selection <sequences/RNAplfold/both>
loss_           = sys.argv[3] # Loss function selection <binary_crossentropy/mse>
in_vivo_vitro   = sys.argv[4] # Dataset selection <in_vivo/in_vitro>
seq_file        = sys.argv[5] # x_data file (sequences)
RNAplfold_file  = sys.argv[6] # x_data file (RNAplfold)
struct_file     = sys.argv[7] # y_data file (structures)
###############################################################################################################################################################
# Set parameters according to the user selections.
#1. Input data setting
if x_data_ == 'both':
    select_x_data = (1,1);
    input_dir = "input_seq_rnaplfold"
elif x_data_ == 'sequences':
    select_x_data = (1,0);
    input_dir = "input_seq"
elif x_data_ == 'RNAplfold':
    select_x_data = (0,1);
    input_dir = "input_rnaplfold"
else:
    print("Unknown input type selection")
    sys.exit(printUsage())
#2.  Loss function setting 
loss_dir = loss_
if loss_ == 'binary_crossentropy':
    loss_ = losses.binary_crossentropy
    activation_last_layer = 'sigmoid'
elif loss_ == 'mse':
    activation_last_layer = 'linear'
else:
    print("Unknown loss function selection")
    sys.exit(printUsage())
#3. Dataset setting         
if ((in_vivo_vitro != "in_vivo") and (in_vivo_vitro != "in_vitro")): 
    print("Unknown dataset selection")
    sys.exit(printUsage())
neighbours_vec_str = str(NEIGHBOURS_VEC)
###############################################################################################################################################################
# Set path to save train and test outputs
current_path = os.getcwd()
filepath_models = current_path + '/outputs/' + '/saved_models/' + in_vivo_vitro +'/' + input_dir +'/' + loss_dir + '/' # path that indicates the chosen training configuration under 'outputs' folder, i.e - (1)Dataset selection (2) input data type selection (3) loss function selection
filepath_results = current_path + '/outputs/' + '/test_results/' + in_vivo_vitro +'/' + input_dir +'/' + loss_dir + '/'
###############################################################################################################################################################
# Read data
dataset= read_data(seq_file, RNAplfold_file, struct_file)
sequences_uncoded, sequences, RNAplfolds, lengths, structures = dataset
###############################################################################################################################################################
#Train the network
if directive == "train":
    filepath = filepath_models # path to save train outputs
    if not os.path.exists(filepath):
        os.makedirs(filepath)    
    for neighbours in list(NEIGHBOURS_VEC):        
        x_data, y_data, num_features = arange_data(sequences, structures, RNAplfolds, select_x_data, neighbours)        
        model_fc = model(neighbours,num_features,loss_,OPTIMIZER_,activation_last_layer)
#        print('Network architecture:')
#        model_fc.summary()       
        # Save the models under the current working directory -> outputs -> saved_model -> ... -> model_neighbours*_epoch*.hdf5
        model_name = 'model_neighbours' + str(neighbours) + '_'
        filename = filepath + model_name + 'epoch{epoch:02d}.hdf5'  
        print(model_name)                
        # Record loss and accuracy history at the end of every epoch
        callbacks_ = getCallbacks(filename)   
        #Fit the model
        model_fc.fit(x_data, y_data, epochs = EPOCHS_, batch_size = BATCH_, callbacks = callbacks_, verbose=2) # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        #Write loss and accuracy outputs to files
        LossAcc(callbacks_, filepath, model_name)


#Test the network        
elif directive == "test":
    filepath = filepath_results # path to save test outputs
    if not os.path.exists(filepath):
        os.makedirs(filepath)     
    for neighbours in list(NEIGHBOURS_VEC):
        x_data, y_data,num_features = arange_data(sequences, structures, RNAplfolds, select_x_data, neighbours)
        for i in range(1, EPOCHS_+1):     
            # Load trained model
            model_name = 'model_neighbours' + str(neighbours) + '_epoch' + '0' + str(i) + '.hdf5'
            model_fc = load_model(filepath_models + model_name)  
            # Predict
            print('Predict: Number of neighbours: ''{}'' Number of epochs: ''{}\n' .format(neighbours,i))
            y_pred_prob= model_fc.predict(x_data,batch_size=BATCH_, verbose=2) # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            y_pred_prob = y_pred_prob.ravel()
            y_true = y_data
            # Save predictions under the current working directory -> outputs -> test_results -> ... -> predictions_*.txt
            predictions(filepath, neighbours, i, sequences_uncoded, y_pred_prob)
            # Calculate and save performances under the current working directory -> outputs -> test_results-> ... -> performance_*.txt
            performance_output_file = performance(filepath, neighbours, model_name, i, y_pred_prob, y_true)
    performance_output_file.close() # close performance file

# Wrong directive input selection
else: 
    print("Unknown directive")
    sys.exit(printUsage()) 

# Write log
output_file_net_details = log(directive, in_vivo_vitro, seq_file, RNAplfold_file, struct_file, sequences_uncoded,
                        lengths,BATCH_, x_data_, loss_dir, activation_last_layer, OPTIMIZER_, LR_, DECAY_, filepath, NEIGHBOURS_VEC) 