###################################################################################################################################################################################################################################################################################
from utility_funcs import *
import time
import glob
import sys
import os
import numpy as np

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import *
#from tensorflow.keras.optimizers import *
#from tensorflow.keras import losses
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.callbacks import LambdaCallback
#from tensorflow.keras.callbacks import Callback

from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
from keras.optimizers import *
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback


import tensorflow as tf
#"train" "(1,1)" "binary_crossentropy" "in_vitro" "C:\Users\Yifat\Desktop\deepSHAPE\partial_data\3_in_vitro\train\sequences3.txt" "C:\Users\Yifat\Desktop\deepSHAPE\partial_data\3_in_vitro\train\sequences3.txt.RNAplfold" "C:\Users\Yifat\Desktop\deepSHAPE\partial_data\3_in_vitro\train\annotations3.txt"
###############################################################################################################################################################
neighbours_vec=[80]#[10,20,40,60,80] #Look at the 2*neighbours neighbours(from front and  back symmetrically)
epochs_ = 3
lr_ = 0.001
decay_ = 1e-5
optimizer_= Adam(lr = lr_, decay = decay_)
batch_ = 16
return_x_data_as_list_or_array = "as_array"
###############################################################################################################################################################
# Random generator initializers
#tf.random.set_seed(1)
tf.set_random_seed(1)
np.random.seed(1)


if len(sys.argv) != 8:
    print("Usage: python myProg.py <train/predict> <(1,1)/(1,0)/(0,1)> <binary_crossentropy/mse> <in_vivo/in_vitro> <seq_file> <RNAplfold_file> <struct_file>")
    exit()

directive = sys.argv[1] # Set directive <train/predict>
select_x_data = sys.argv[2];# Select x data <(1,1)/(1,0)/(0,1)> - stands for (add_sequences, add_RNAplfold)
loss_ = sys.argv[3]; # <binary_crossentropy/mse>
in_vivo_vitro = sys.argv[4]; # <in_vivo/in_vitro>
seq_file = sys.argv[5]; # x_data (sequences)
RNAplfold_file = sys.argv[6]; # x_data (RNAplfold)
struct_file = sys.argv[7]; # y_data (structures)


if select_x_data == '(1,1)':
    select_x_data = (1,1);
    input_dir = "input_seq_rnaplfold"
elif select_x_data == '(1,0)':
    select_x_data = (1,0);
    input_dir = "input_seq"
elif select_x_data == '(0,1)':
    select_x_data = (0,1);
    input_dir = "input_rnaplfold"

loss_dir = loss_
if loss_ == 'binary_crossentropy':
    loss_ = losses.binary_crossentropy
    activation_last_layer = 'sigmoid'
elif loss_ == 'mse':
    activation_last_layer = 'linear'
    
neighbours_vec_str='neighbours'
for neighbours in neighbours_vec:
    curr = '_' + str(2*neighbours)
    neighbours_vec_str =neighbours_vec_str +curr

current_path = os.getcwd()
filepath_models = current_path + '/results/' + '/saved_models/' + in_vivo_vitro +'/' + input_dir +'/' + loss_dir + '/'
filepath_results = current_path + '/results/' + '/net_details_predictions_performances/' + in_vivo_vitro +'/' + input_dir +'/' + loss_dir + '/'

# Read data
dataset= read_data(seq_file, RNAplfold_file, struct_file)
sequences_uncoded, sequences, RNAplfolds, lengths, structures = dataset

#Learning
if directive == "train":
    

    if not os.path.exists(filepath_models):    
            os.makedirs(filepath_models) 
       
    # Print network details
    output_file_net_details = write_net_details(directive, in_vivo_vitro, seq_file, RNAplfold_file, struct_file,sequences_uncoded,
                            lengths,batch_,select_x_data, loss_, activation_last_layer, optimizer_, lr_, decay_, filepath_results, neighbours_vec_str) 
    output_file_net_details.close()
    
    for neighbours in neighbours_vec:
        model_name = 'model_'+'Neighbours'+str(2*neighbours)+'_'
        print(model_name)
        print('\n', 'number of neighbours:', str(2*neighbours))
        x_data, y_data, num_features = arange_data(sequences, structures, RNAplfolds, select_x_data,neighbours,return_x_data_as_list_or_array)
  
        model=Sequential()
        model.add(Flatten(input_shape=(2*neighbours+1,num_features)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation = activation_last_layer))
        model.compile(optimizer = optimizer_, loss = loss_ ,metrics=['accuracy'])
		  
        # Save models under current working directory -> saved_models
        filename= filepath_models + model_name+'epoch{epoch:02d}.hdf5'           
        
        # Record loss and accuracy history at the end of every epoch
        class LossAccHistory(Callback):
            def on_train_begin(self, logs={}):
                self.losses = []
                self.acc = []
            def on_epoch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
                self.acc.append(logs.get('acc'))
                
        #Save the model after every epoch and record loss and accuracy history
        callbacks_ = [ModelCheckpoint(filename, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='max', period=1), #Save the model after every epoch
                      LossAccHistory()] # Record loss and accuracy history         
     
        #Fit model
        model.fit(x_data, y_data, epochs = epochs_, batch_size = batch_, callbacks = callbacks_, verbose=2) # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
		
        #Write loss and accuracy outputs to files                   
        loss_history = callbacks_[1].losses
        np_loss_history = np.array(loss_history)
        np.savetxt(filepath_models + model_name + "loss_history.txt", np_loss_history, delimiter=",")
        acc_history = callbacks_[1].acc
        np_acc_history = np.array(acc_history)
        np.savetxt(filepath_models + model_name + "acc_history.txt", np_acc_history, delimiter=",")
        
        print('Network architecture:')
        model.summary()

#Predicting             
if directive == "predict":

    
    # Write network details under current working directory -> results -> net_details_predictions_performances -> ... -> performance_.txt
    output_file_performance = write_net_details(directive, in_vivo_vitro, seq_file, RNAplfold_file, struct_file,sequences_uncoded,
                            lengths,batch_,select_x_data, loss_, activation_last_layer, optimizer_, lr_, decay_, filepath_results, neighbours_vec_str) 
        
    for neighbours in neighbours_vec:
        x_data, y_data,num_features = arange_data(sequences, structures, RNAplfolds, select_x_data, neighbours,return_x_data_as_list_or_array)
        for i in range(1, epochs_+1):    
            # Load trained model
            model_name = 'model_Neighbours'+str(2*neighbours)+'_epoch'+'0'+str(i)+'.hdf5'
            model = load_model(filepath_models + model_name)           
            
            # Predict
            y_pred_prob= model.predict(x_data,batch_size=batch_, verbose=2) # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            y_pred_prob = y_pred_prob.ravel()
            y_true = y_data
            
            # Save predictions under current working directory -> results -> net_details_predictions_performances -> ... -> predictions_.txt
            write_predictions(filepath_results, neighbours, i, sequences_uncoded, y_pred_prob)
            # Calculate and save performances under current working directory -> results -> predictions_and_performances-> ... -> performance_.txt
            output_file_performance = calc_and_write_net_performance(output_file_performance, neighbours, model_name, i, y_pred_prob, y_true)
           
    output_file_performance.close()
    
   