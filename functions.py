'''
This file contains all the functions useful for carrying out certain tasks. They are put in the form of a function
because these will appear more times in the code or simply to have a cleaner and more readable code.

The functions are divided into the following macro groups:
    - DATASET FUNCTIONS: Functions for reducing and removing dirty data from the training dataset.
    - GENERIC FUNCTIONS: Functions used during the execution of the code and with multiple uses.
    - NORMALIZATION FUNCTIONS: Functions for normalizing input and target data
    - SPLIT FUNCTIONS: Functions for splitting in input tensor and target tensor, and functions for splitting in the three sets.
    - SAVE AND PLOT FUNCTIONS: Functions for saving nets and plotting results.
'''

#Import needed libraries, classes and functions
import torch
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
'''
            ############################################################################
            #                           DATASET FUNCTIONS
            ############################################################################
'''

def dataset_reduction(dataframe, *args:str, num_col = 6): #Reduce dataset
    '''
    It reduces the dimensionality of the dataset to only take the columns we are interested.
    Args:
        dataframe: entire dataset.
        args: all the column names we want to select.
    Returns:
        The reduced dataset consists of all the rows and selected columns which were passed as arguments to the function.
    '''
    #Control if the number of columns which we want is correct
    if len(args) != num_col:
        raise ValueError('Check the number of columns you want to select')

    #Selects columns specified by names in *args from the dataframe
    columns_to_select = [col for col in args if col in dataframe.columns]
    #Reduction of the dataset
    dataframe_reduced = dataframe.loc[:, columns_to_select]
    
    return dataframe_reduced

def cleaning_dataset_function(dataframe): #Clean reduced dataset
    '''
    Provides dataset cleaning by removing all those records where there is at least one NULL/NAN/undetected value.

    Args:
        dataframe: reduced dataset.
    Returns:
        The reduced dataset cleaned.
    '''
    ### Remove records with NaN or null or invalid values ###
    #Created an explicit copy of the DataFrame to ensure that changes are made on the original (copy) object so avoiding the "SettingWithCopyWarning" error
    dataframe_copy = dataframe.copy()
    #Invalid data removal: NAN/NULL/""
    dataframe_copy.dropna(inplace=True) 

    ### Convert the columns to the float datatype ###
    for col in dataframe_copy.columns:
        #Check if the column contains string values before using the .str accessor
        if dataframe_copy[col].dtype == 'object': #Assuming the string columns have 'object' dtype
            dataframe_copy[col] = dataframe_copy[col].str.replace(',', '.').astype(float)

    ### Verify that the data are different from -200 ###
    #Create a boolean array with True for each row where all columns are different from -200
    condition = (dataframe_copy != -200).all(axis=1)
    
    #Remove records that do not meet both conditions
    dataframe = dataframe_copy[condition]

    #Re-index the rows otherwise keep the previous index without the deleted record indexes
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe

def PCA_fun(X): #Dimensional reduction Principal Component Analysis
    #Use PCA to reduce dimensionality to 1 dimension
    pca = PCA(n_components=1)
    X_pca_numpy = pca.fit_transform(X)

    #Convert the NumPy array to a PyTorch tensor
    X_pca = torch.from_numpy(X_pca_numpy)
    
    return X_pca
'''
            ############################################################################
            #                           GENERIC FUNCTIONS
            ############################################################################
'''            
def create_file_csv(dataframe, filename): #Create CSV file
    '''
    Provides dataframe and a path to save this dataframe to a csv file.

    Args:
        dataframe: reduced dataset.
        filename: path where you want to save the dataframe.
    Returns:
        The csv file.
    '''
    #Save my dataframe to a new .csv file.
    dataframe.to_csv(filename, index=False)

def createDirectory(nameDirectory: str): #Create new directory
    """
    This function is used to create folders, taking as input the absolute path of the folder we want to create (path/foldername). In our case, the path is passed
    through the yaml file. The folders that will be created are those for saving training results, and during the evaluation phase, the folder
    where the images produced by the generator will be saved.
    """
    if not os.path.exists(f'{nameDirectory}'):  #checks if the folder with that path already exists
        os.mkdir(f'{nameDirectory}')  #if it does not exist creates it

def load_csv_file(filename): #Load CSV file in X tensor(used in autoencoder.py)
    '''
    Function that loads the dataset, by the filename, into a pytorch tensor and saves it into the variable X.
    
    Args:
        filename: path of the reduced cleaned dataset.
    Return:
        Pytorch tensor which contain the data.

    '''

    #Read the dataset and append it in the dataframe
    dataframe = pd.read_csv(filename)
    #Convert the dataframe to a torch tensor
    X = torch.tensor(dataframe.values) 
    return X

'''
            ############################################################################
            #                         NORMALIZATION FUNCTIONS
            ############################################################################
'''
def restrict_input_data(X, min_val=None, max_val=None): #Restrict input [-1,1]
    '''
    Provides a method for restrict input pytorch tensors to a pytorch tensor of values between -1 and 1.

    Args:
        X: input tensor not restricted.
        min_val: min value for restriction.
        max_val: max value for restriction.
    Returns:
        Restriction input pytorch tensor.
    '''
    #Calculate of min and max tensor value
    if min_val is None:
            min_val = X.min()
    if max_val is None:
            max_val = X.max()        
    # Adjust the range to -1 to 1 --> Restriction
    normalized_X = 2 * (X - min_val) / (max_val - min_val) - 1

    return normalized_X

def restrict_output_data(y, min_val=None, max_val=None): #Restrict output [0,1]
    '''
    Provides a method for restrict output pytorch tensors to a pytorch tensor of values between 0 and 1.

    Args:
        y: output tensor not normalize.
        min_val: min value for normalization.
        max_val: max value for normalization.
    Returns:
        Normalized output pytorch tensor.
    '''
    #Calculate of min and max tensor value
    if min_val is None:
        min_val = y.min()
    if max_val is None:
        max_val = y.max()
    #Adjust the range to 0 to 1 --> Normalization
    normalized_y = (y - min_val) / (max_val - min_val)
    
    return normalized_y

def real_norm_input(X, mean=None, std=None): #Normalization input
    '''
    Provides a method for normalizing input pytorch tensors to a pytorch tensor.

    Args:
        X: input tensor not restricted.
        mean: mean value for restriction.
        std: std value for restriction.
    Returns:
        Normalize input pytorch tensor.
    '''    
    #Calculate of mean and std tensor value
    if mean is None:
        mean = torch.mean(X)
    if std is None:
        std = torch.std(X)
    #Normalization
    X_normalized = (X - mean) / std

    return X_normalized, mean, std

'''
            ############################################################################
            #                             SPLIT FUNCTIONS
            ############################################################################
'''
def load_data_from_file(filename, num_col = 6):
    '''
    Provides a path and the number of columns of your dataset and split in input and output pytorch tensors.

    Args:
        filename: path where you want to save the dataframe.
        num_col: number of column of your dataset
    Returns:
        Two tensors, the input tensor with the first columns and the output tensor with the last column.
    '''

    #Load data by the csv file in a DataFrame using pandas
    data = pd.read_csv(filename)

    #Splitting data in inputs and targets
    X = torch.tensor(data.iloc[:, :(num_col-1)].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

    return X, y

def create_autoencoder_splits_unbalanced(X, train_frac, valid_frac, randomize=True):
    """
    Creates (randomized) training, validation, test data splits.

    Args:
        X: dataset (one example per row).
        train_frac: the fraction of data to be used in the training set.
        valid_frac: the fraction of data to be used in the validation set.
        input_norm: function for normalizing X.
        randomize (optional): randomize the data before splitting them.

    Returns:
        The PyTorch tensors X, y (data and targets) for training, validation, and test sets, and normalize respectively in a compact way.
    """
    #Randomize data
    if randomize:
        #Shuffle the indexes
        indices = torch.randperm(len(X))
        #Sheffle the data
        X = X[indices]

    #Total samples number
    n = len(X)

    #Calculate the number of samples that were in training and validation
    train_size, valid_size = int(n * train_frac), int(n * valid_frac)

    return X[:train_size], X[train_size:train_size+valid_size], X[train_size+valid_size:]

def create_splits_unbalanced(X, y, train_frac, valid_frac, randomize=True, max_min_into_training = True):
    """
    Creates (randomized) training, validation, test data splits. Esnure that the max and min value are putted
    into the training set.

    Args:
        X: dataset (one example per row).
        y: ground truth labels (vector).
        train_frac: the fraction of data to be used in the training set.
        valid_frac: the fraction of data to be used in the validation set.
        randomize (optional): randomize the data before splitting them.
        max_min_into_training(optional): True if you want to include the max and min into the training set.
    Returns:
        The PyTorch tensors X, y (data and targets) for training, validation, and test sets.
    """
    #Randomize data
    if randomize:
        # Shuffle the indexes
        indices = torch.randperm(len(X))
        # Shuffle the data
        X, y = X[indices], y[indices]

    #Total samples number
    n = len(X)

    #Find the minimum and maximum values
    min_val = torch.min(X)
    max_val = torch.max(X)

    #Calculate the number of samples that were in training and validation
    train_size, valid_size = int(n * train_frac), int(n * valid_frac)

    #Split in training and remaining data
    train_set_X, train_set_y = X[:train_size], y[:train_size]
    remaining_X, remaining_y = X[train_size:], y[train_size:]

    #Ensure minimum and maximum values are included in training set if it is true.
    if max_min_into_training:
        #Check if min_val is not already in the training set
        if min_val not in train_set_X:
            # Find the first occurrence of min_val in the remaining data
            min_idx = torch.where(remaining_X == min_val)[0][0]
            # Add the min_val and its corresponding label to the training set
            train_set_X = torch.cat([train_set_X, remaining_X[min_idx].unsqueeze(0)], dim=0)
            train_set_y = torch.cat([train_set_y, remaining_y[min_idx].unsqueeze(0)], dim=0)
            #Remove min_val from the remaining data
            remaining_X = torch.cat([remaining_X[:min_idx], remaining_X[min_idx+1:]], dim=0)
            remaining_y = torch.cat([remaining_y[:min_idx], remaining_y[min_idx+1:]], dim=0)

        #Check if max_val is not already in the training set
        if max_val not in train_set_X:
            # Find the first occurrence of max_val in the remaining data
            max_idx = torch.where(remaining_X == max_val)[0][0]
            # Add the max_val and its corresponding label to the training set
            train_set_X = torch.cat([train_set_X, remaining_X[max_idx].unsqueeze(0)], dim=0)
            train_set_y = torch.cat([train_set_y, remaining_y[max_idx].unsqueeze(0)], dim=0)
            # Remove max_val from the remaining data
            remaining_X = torch.cat([remaining_X[:max_idx], remaining_X[max_idx+1:]], dim=0)
            remaining_y = torch.cat([remaining_y[:max_idx], remaining_y[max_idx+1:]], dim=0)

    #The rest of the data goes into the validation and test sets
    valid_set_X, valid_set_y = remaining_X[:valid_size], remaining_y[:valid_size]
    test_set_X, test_set_y = remaining_X[valid_size:], remaining_y[valid_size:]


    return train_set_X, train_set_y, valid_set_X, valid_set_y, test_set_X, test_set_y, min_val, max_val

'''
            ############################################################################
            #                         SAVE AND PLOT FUNCTIONS
            ############################################################################
'''

def save_net(model, epoch, epoch_loss_train, epoch_loss_val, mean_value, std_value, r2, path_pth, path_txt):
    '''
    Save the "model state" of the net to a .pth file in the specified path
    Args:
        -
        -
        -
        -
        -
        -

    Return:
        Two file with the model state and a text file with some info.
    
    '''
    #Save the model
    torch.save(model.state_dict(), path_pth)
    #Write a .txt file to the specified path and writes information regarding the epoch and the loss to which
    #the best trained net belongs
    with open(path_txt, "w") as f:
        print(f"Checkpoint net:\n\n\tEPOCH:\t{epoch}\n\n\tLOSS TRAIN:\t{epoch_loss_train}\n\n\tLOSS VALIDATION:\t{epoch_loss_val}\n\n\tMEAN VALIDATION:\t{mean_value}\n\n\tSTD VALIDATION:\t{std_value}\n\n\tR2 VALIDATION:\t{r2}", file=f)

def save_net_aut(model, epoch, epoch_loss_train, epoch_loss_val, path_pth, path_txt):
    '''
    Save the "model state" of the net to a .pth file in the specified path
    Args:
        -
        -
        -
        -
        -
        -

    Return:
        Two file with the model state and a text file with some info.
    
    '''
    #Save the model
    torch.save(model.state_dict(), path_pth)
    #Write a .txt file to the specified path and writes information regarding the epoch and the loss to which
    #the best trained net belongs
    with open(path_txt, "w") as f:
        print(f"Checkpoint net:\n\n\tEPOCH:\t{epoch}\n\n\tLOSS TRAIN:\t{epoch_loss_train}\n\n\tLOSS VALIDATION:\t{epoch_loss_val}", file=f)

def plot_loss(n_epochs, net_train_losses, net_val_losses, result_path):

    #Plot of the losses for each epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), net_train_losses, label='Training Loss')
    plt.plot(range(1, n_epochs + 1), net_val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    #Save the image in the result path
    plt.savefig(result_path)

    #Show the image
    plt.show()



