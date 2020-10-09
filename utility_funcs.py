import random
import numpy as np
#import tensorflow as tf
import datetime
import os
import re
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
############################################################################## Use these imports for local ####################################################################
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
##############################################################################  Use these imports for intel cloud ############################################################
#from keras.callbacks import ModelCheckpoint
#from keras.callbacks import Callback
###############################################################################################################################################################
MIN_SEQ_LEN = 10
MAX_SEQ_LEN = 30000

def process_transcriptome_header(header):
    '''
    Process transcriptome header line.
    Args:
        header: header string of the form >CHOROMOSOM:START_INDEX-END_INDEX.
    Returns:
        chromosome  : the chromosome number
        start_index : the transcriptome start index
        end_index   : the transcriptome end index
    '''
    assert(header[0] == '>')
    chromosome_end_index = header.find(":")
    indices_sep_index = header.find("-")
    chromosome = header[1:chromosome_end_index]
    start_index = header[chromosome_end_index+1:indices_sep_index]
    end_index = header[indices_sep_index+1:]
    return chromosome, start_index, end_index


def process_probs(annotations_file):
    '''
    Process next transcriptome and extract meta data and SHAPE/RNAplfold values.
    Args:
        annotations_file   : handler to a SHAPE/RNAplfold file.
                            Each transcriptome data is stored in the following format:
                            (-) Header line.
                            (-) SHAPE/RNAplfold values      
    Returns:
        transcriptome_data : tuple with transcriptome chromosome, start endex and end index.
        structure_npArray  : 1-d numpy array of of a single SHAPE/RNAplfold trancript values, of size [trancript_length X 1]
        
        if no data is left to be read from the file, None value is returned.
    '''      
    data = annotations_file.readline()
    if not data:
        return None
    header_line = data.strip()
    transcriptome_data = process_transcriptome_header(header_line) 
    structure_str = annotations_file.readline().strip()
    structure_list = re.split(',|\s|\t',structure_str) 
    structure_list = list(filter(lambda x: x != "", structure_list)) #remove empty strings from a list of strings
    structure_mat = [float(elem) for elem in structure_list]
    structure_npArrayT = np.array([structure_mat])
    structure_npArray = np.transpose(structure_npArrayT)
    return (transcriptome_data, structure_npArray)       
    
def process_sequences(sequences_file):
    '''
    Process next transcriptome and extract meta data and sequence data.

    Args:
        sequences_file: handler to a sequence file
            Each transcriptome data is stored in the following format:
            (-) Header line.
            (-) Sequence line.
    Returns:
        transcriptome_data : tuple with transcriptome chromosome, start index and end index.
        seq_matrix         : a 2d-numpy array of a single one-hot encoded transcript, of size [transcript_length X alphabet_size].
        seq                : a string of a single uncoded transcript
        
        if no data is left to be read from the file, None value is returned.
    '''
    data = sequences_file.readline()
    if not data:
        return None
    header_line = data.strip() # returns a copy of the string in which all chars have been stripped from the beginning and the end of the string (default whitespace characters).
    transcriptome_data = process_transcriptome_header(header_line)
    seq = sequences_file.readline().strip()
    seq_matrix = np.empty((len(seq),4), int)
    idx = 0
    for base in seq:
        base = base.upper()
        if base == 'A':
            base_encoding = np.array([[1.0, 0.0, 0.0, 0.0]])
        elif base == 'C':
            base_encoding = np.array([[0.0, 1.0, 0.0, 0.0]])
        elif base == 'G':
            base_encoding = np.array([[0.0, 0.0, 1.0, 0.0]])
        elif base == 'U' or base == 'T':
            base_encoding = np.array([[0.0, 0.0, 0.0, 1.0]])
        elif base == 'N':
            base_encoding = np.array([[1.0, 1.0, 1.0, 1.0]])
        else:
            raise ValueError("Base is " + base)
        seq_matrix[idx] = base_encoding
        idx = idx + 1
    seq_matrix = np.byte(seq_matrix) 
    return (transcriptome_data, seq_matrix, seq)


def read_data(sequences_file, RNAplfolds_file, annotations_file):
    '''
    Read icSHAPE data for training and testing (sequence, SHAPE and RNAplfold information)

    Args:
        sequences_file   : file containing transcriptome sequence information.
        RNAplfolds_file  : file containing transcriptome RNAplfold information.
        annotations_file : file containing transcriptome SHAPE information.
        
    Returns:
        sequences_uncoded : Python list, each element is a string of a single uncoded transcript
        sequences         : Python list, each element is a 2d-numpy array of a single one-hot encoded transcript, each of size [transcipt_length X alphabet_size].
        RNAplfolds        : Python list, each element is a numpy array of a single transcript RNAplfold values, each of size [transcipt_length X 1].
        lengths           : Python list, each element is an integer indicating the transcript length.
        structures        : Python list, each element is a numpy array of a single transcript SHAPE values, each of size [transcipt_length X 1].
    '''

    with open(sequences_file) as seq_data, open(RNAplfolds_file) as RNAplfold_data, open(annotations_file) as annot_data:
        sequences_uncoded, sequences, RNAplfolds, lengths, structures =list(), list(), list(), list(),list()
        while True:
            # Process next sample information:  sequence and structure annotations.
            data_seq       = process_sequences(seq_data)
            data_RNAplfold = process_probs(RNAplfold_data)
            data_struct    = process_probs(annot_data)
            # No more samples to process - return entire data
            if not data_seq or not data_struct or not data_RNAplfold:
                return sequences_uncoded, sequences, RNAplfolds, lengths, structures
            transcriptome_data_1, seq_matrix, seq  = data_seq
            transcriptome_data_2, RNAplfold_matrix = data_RNAplfold
            transcriptome_data_3, struct_matrix    = data_struct
            # Validate transcriptome identity
            assert(transcriptome_data_1 == transcriptome_data_2 and transcriptome_data_2 == transcriptome_data_3)
            # Compute sequnce length
            curr_seq_len = len(seq_matrix)
            # Skip too short and too long transcriptomes
            if curr_seq_len < MIN_SEQ_LEN or curr_seq_len > MAX_SEQ_LEN:
                continue
            # Aggregate new transcriptome data 
            lengths.append(curr_seq_len)
            sequences.append(seq_matrix)
            sequences_uncoded.append(seq)
            RNAplfolds.append(RNAplfold_matrix)
            structures.append(struct_matrix)
        assert(False)
 
      
def arange_data(sequences, structures, RNAplfolds, select_x_data,L):
    '''
    Arange icSHAPE data for training and testing (sequence, SHAPE and RNAplfold information)
    Denote by 'l' the length of the current RNA sequence and by '2L' the windows size.
    The RNA sequence is padded with n zero-vectors at both ends of the sequence. Using a sliding window with stride 1, every sequence is divided
    into 'l' subsequences of length L+1 each. Every subsequence represents a nucleotide and its L adjacent neighbors (from both flanks symmetrically).
    These one-hot encoded subsequences and their corresponding base-pairing probabilities are then concatenated along the column dimension to form 
    'l' data points, each represented as a matrix of size (L + 1) X 5. Finally, these steps are applied to every RNA sequence in the dataset.
        
    Args:
        sequences     : Python list, each element is a 2d-numpy array of a single one-hot encoded transcript, each of size [transcipt_length X alphabet_size].
        structures    : Python list, each element is a numpy array of a single transcript SHAPE values, each of size [transcipt_length X 1].
        RNAplfolds    : Python list, each element is a numpy array of a single transcript RNAplfold values, each of size [transcipt_length X 1].
        select_x_data : A tuple of one of the following values <(1,1)/(1,0)/(0,1)> - stands for (add_sequences, add_RNAplfold)
        L             : number of neighbour.
        
    Returns:        
        all_x_data   : Numpy array. All numpy arrays from all transcripts are concateneated together along the row axis 
                       to form one numpy array of size  [total number of nucleotides in all transcripts X (L + 1) X (alphabet_size+1)]
                       each row slice represents a nucleotide and its L adjacent neighbors (one hot encoded) together with their RNAplfolds probabilities .
        all_y_data   : One long numpy array containing all icSHAPE probabilities (of size: total number of nucleotides in all transcripts).
        num_features : Number of dimensions which represent one nucleotide
    ''' 
    # concatenate structures
    all_y_data = np.concatenate(structures).ravel() #Flatten the structure list into numpy array
    add_sequences, add_RNAplfold= select_x_data
    
    
    if add_sequences == 1 and add_RNAplfold == 1:   
        sequences_Plus_RNAplfolds_list = []
        for i in range(0,len(sequences)):
            sequences_Plus_RNAplfolds= np.concatenate((sequences[i], RNAplfolds[i]),axis=1) # concatenate RNAplfold to sequence (along the column)                        
            sequences_Plus_RNAplfolds_list.append(sequences_Plus_RNAplfolds) # concatenate every RNA sequence and maching RNAplfolds to the rest of the sequences (along the rows)
        x_data = sequences_Plus_RNAplfolds_list
    elif add_sequences == 1 and add_RNAplfold == 0:
        sequences_list = []
        for i in range(0,len(sequences)):
            sequences_list.append(sequences[i]) # concatenate every RNA sequence to the rest of the sequences (along the rows)
        x_data = sequences_list
    elif add_sequences == 0 and add_RNAplfold == 1:
        plfold_list = []
        for i in range(0,len(RNAplfolds)):
            plfold_list.append(RNAplfolds[i]) # concatenate every RNAplfold value to the rest of the RNAplfolds (along the rows)
        x_data = plfold_list
    else: 
        assert(False)
                         
    rows, num_features = x_data[0].shape
    
    n_nucleotides = 0
    for x in x_data:
        n_nucleotides = n_nucleotides + int((x.size / num_features))
    
    npZeros = np.zeros((int(L/2), num_features),dtype='float32')
    # zero padding L/2 bottom lines 
    x_data_bottom_padded = [np.append(np.asarray(x, dtype=np.float32), npZeros, axis=0) for x in x_data]
    # zero padding L/2 top lines
    x_data_bottom_top_padded = [np.append(npZeros, x, axis=0) for x in x_data_bottom_padded]
    # for each nucleotide, Look at the L neighbours (from front and  back symmetrically) and store this information along a new dimension
    list_list_Lneighbours = [[x[i:i+L+1] for i in range(0,len(x)-(L))] for x in x_data_bottom_top_padded] # Python list, each element* represents a single trancript. [number_of_transcripts X 1]
            #Each element* contain a 'l'-long list of 2-d numpy arrays** (one numpy array for every nucleotide (and its L adjacent neighbors) in the transcript).
            #Each numpy array** is a one-hot encoded subsequence concatenated by its corresponding base-pairing probabilities (a middle nucleotide and its L adjacent neighbors). 
            #Each numpy array** is of size [(L + 1) X (alphabet_size+1)]   
    all_x_data = np.concatenate(list_list_Lneighbours)

    return(all_x_data, all_y_data,num_features)
    
def predictions(filepath_results, neighbours, i, sequences_uncoded, y_pred_prob):
    '''
    Write network predictions to a text file. 
    Each transcriptome data is written in the following format:
        (-) transcriptome string
        (-) SHAPE predictions

    Args:
        filepath_results : Path to the directory which will store the output file  
        neighbours       : Number of adjacent neighbors (from both flanks symmetrically)
        i                : Number of epochs
        sequences_uncoded: Python list, each element is a string of a single uncoded transcript
        y_pred_prob      : One long numpy array containing all icSHAPE predictions (of size: total number of nucleotides in all transcripts).
        
    '''
    predictions_filename = (filepath_results + 'predictions' +'_neighbours_'+str(neighbours)+'_epoch_'+ str(i)+'.txt');
    if not os.path.exists(os.path.dirname(predictions_filename)):
        os.makedirs(os.path.dirname(predictions_filename))
    f = open(predictions_filename,"w+")
    seq_count = 0
    for seq in sequences_uncoded:
        f.write('>' + seq + '\n')
        seq_y_pred =  y_pred_prob[seq_count:seq_count+len(seq)]
        for prob in seq_y_pred:
            f.write('\t' + str(prob))
        f.write('\n')
        seq_count = seq_count + len(seq)
    assert(y_pred_prob[-1] == seq_y_pred[-1])
    f.close();


def calc_performance(y_pred_prob, y_true):
    '''
    Calculate network performance (Person corelation, mean squered error and mean absolute error between the network SHAPE predictions and the true SHAPE lables)
    Args:
        y_pred_prob : One long numpy array containing all icSHAPE predictions (of size: total number of nucleotides in all transcripts).
        y_true      : One long numpy array containing all icSHAPE true lables (of size: total number of nucleotides in all transcripts).        
    Returns:
        pearson_corr1 : Person corelation between the network SHAPE predictions and the true SHAPE lables (over all nucleotides)
        mse           : mean squered error between the network SHAPE predictions and the true SHAPE lables (over all nucleotides)
        mae           : mean absolute error between the network SHAPE predictions and the true SHAPE lables (over all nucleotides)
        
    '''
    #Calculate network performance 
    pearson_corr1 = np.corrcoef(y_pred_prob, y_true)[0,1]
    mse =  mean_squared_error(y_pred_prob, y_true)
    mae = mean_absolute_error(y_pred_prob, y_true)
    return (pearson_corr1, mse, mae)

def performance(filepath, neighbours, model_name, i, y_pred_prob, y_true):
    '''
    Write network performance to a text file
    Args:
        filepath    : Path to the directory which will store the output file  
        neighbours  : Number of current adjacent neighbors (from both flanks symmetrically)
        model_name  : A string indicating the number of neighboures and the epoch number
        i           : Number of current epoch
        y_pred_prob : One long numpy array containing all icSHAPE predictions (of size [total number of nucleotides in all transcripts]).
        y_true      : One long numpy array containing all icSHAPE true lables (of size [total number of nucleotides in all transcripts]).
    Returns:
        output_file : A file object which can be used to read, write and modify the file.
    '''
    # Calculate network performance 
    pearson_corr1, mse, mae = calc_performance(y_pred_prob, y_true)  
     
    
    # Write network performance
    if not os.path.exists(filepath):    
        os.makedirs(filepath)
    name = "performance"
    output_file = open(filepath + name + ".txt", "a+") 
    if i == 1: # First epoch
        output_file.write('\n>Number of neighbours:''{}\n\n'.format(neighbours)) 
    output_file.write('Model: ''{}\n'.format(model_name))
    output_file.write('Number of epochs: ''{}\n'.format(i))
    output_file.write('Pearson Correlation between y_pred and y_true: ''{}\n'.format(pearson_corr1))
    output_file.write('Mean Squared Error between y_pred and y_true: ''{}\n'.format(mse))
    output_file.write('Mean Absolute Error between y_pred and y_true: ''{}\n'.format(mae))
    return(output_file)

def log(directive, in_vivo_vitro, seq_file, RNAplfold_file, struct_file,sequences_uncoded,
                            lengths,batch_, x_data_ , loss_, activation_last_layer, optimizer_, lr_, decay_, filepath, NEIGHBOURS_VEC ):
    '''
    Write network details to a text file 
    Args:
        directive             : A string indicating the current directive (train or predict)
        in_vivo_vitro         : A string indicating the current data-set (in vitro data or in vivo data)
        seq_file              : Sequence file (path + file name)
        RNAplfold_file        : RNAplfold file (path + file name)
        struct_file           : SHAPE file (path + file name)
        sequences_uncoded     : Python list, each element is a string of a single uncoded transcript
        lengths               : Python list, each element is an integer indicating the transcript length.
        batch_                : A network parameter - batch size
        x_data_               : A string. Input data type selection <sequences/RNAplfold/both>
        loss_                 : A network parameter - loss function
        activation_last_layer : A network parameter - the activation function of the last layer
        optimizer_            : A Network parameter - optimizer
        lr_                   : A network parameter - learning rate
        decay_                : A network parameter - decay
        filepath              : A path to the directory which will store the output file  
        NEIGHBOURS_VEC        : A numpy array indicating the number of neighbours

    Returns:
        output_file           : A file object which can be used to read, write and modify the file.
    '''
    name = "log"
    if not os.path.exists(filepath):    
        os.makedirs(filepath) 
    output_file = open(filepath + name + ".txt", "w") 
    output_file.write(directive + ' the network\n')
    output_file.write('Number of neighbors from both flanks symmetrically: ''{}\n'.format(str(NEIGHBOURS_VEC)))
    output_file.write('Dataset type selection: ''{}\n'.format(in_vivo_vitro))
    output_file.write('Input data type selection: ''{}\n'.format(x_data_))
    output_file.write('Loss function selection: ''{}\n'.format(loss_))
    output_file.write('Activation function of the last layer: ''{}\n'.format(activation_last_layer))
    output_file.write('Batch size: ''{}\n'.format(batch_))
    output_file.write('Optimizer: ''{}\n'.format(optimizer_))
    output_file.write('Learning rate: ''{}\n'.format(lr_))
    output_file.write('Decay: ''{}\n'.format(decay_))
    output_file.write('Total number of RNA sequences in the ' + directive + ' set: ''{}\n'.format(len(sequences_uncoded)))
    output_file.write('Total number of nucleotides in the ' + directive + ' set: ''{}\n'.format(sum(lengths)))
    output_file.write('Sequences file: ''{}\n'.format(seq_file))
    output_file.write('RNAplfold file: ''{}\n'.format(RNAplfold_file))
    output_file.write('Structure file: ''{}\n'.format(struct_file))
    output_file.write('************************************************************\n\n')
    return ()

def getCallbacks(filename):
    '''
    Save the model after every epoch and record loss and accuracy history
    Args:
        filename  : A string indicating a folder path + name of the model    
    Returns:
        callbacks_: An object
    '''
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
    return(callbacks_)
        
def LossAcc(callbacks_, filepath, model_name): 
    '''
    Write loss and accuracy outputs to files
    Args:
        callbacks_ : An object
        filepath   : A string indicating a folder path to save the Loss and accuracy files        
        model_name : A string indicating the file name
    '''        
    loss_history = callbacks_[1].losses
    np_loss_history = np.array(loss_history)
    np.savetxt(filepath + model_name + "loss_history.txt", np_loss_history, delimiter=",")
    acc_history = callbacks_[1].acc
    np_acc_history = np.array(acc_history)
    np.savetxt(filepath + model_name + "acc_history.txt", np_acc_history, delimiter=",")

def printUsage():
    '''
    Wrong input. Print correct format. 
    '''
    print("Usage: python myProg.py <train/test> <sequences/RNAplfold/both> <binary_crossentropy/mse> <in_vivo/in_vitro> <path to seq_file> <path to RNAplfold_file> <path to struct_file>")