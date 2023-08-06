'''
                    ############################################################

                                            FUNCTIONS

                    ############################################################

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

def dataset_reduction(dataframe, *args:str, num_col:int = 6): #Reduce dataset
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
    '''
    Given a multidimensional tensor returns a one-dimensional tensor.

    Args:
        X: input tensor.
    Returns:
        The monodimensional tensor.
    '''
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
def create_file_csv(dataframe, filename:str): #Create CSV file
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

def createDirectory(nameDirectory:str): #Create new directory
    """
    This function is used to create folders
    
    Args:
        nameDirectory: absolute path of the folder which want to create (path/foldername).
    Returns:
        The directory will be created with the path specified.
    """
    if not os.path.exists(f'{nameDirectory}'):  #checks if the folder with that path already exists
        os.mkdir(f'{nameDirectory}')  #if it does not exist creates it

def load_csv_file(filename:str): #Load CSV file in X tensor(used in autoencoder.py)
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
def restrict_input_data(X, min_val:float=None, max_val:float=None): #Restrict input [-1,1] (not used in our code, but a possible solution)
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

def real_norm_input(X, mean:float=None, std:float=None): #Normalization input
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

def restrict_output_data(y, min_val:float=None, max_val:float=None): #Restrict output [0,1]
    '''
    Provides a method for restrict output pytorch tensors to a pytorch tensor of values between 0 and 1.

    Args:
        y: output tensor not normalize.
        min_val: min value for normalization.
        max_val: max value for normalization.
    Returns:
        Restricred [0,1] output pytorch tensor.
    '''
    #Calculate of min and max tensor value
    if min_val is None:
        min_val = y.min()
    if max_val is None:
        max_val = y.max()
    #Adjust the range to 0 to 1 --> Normalization
    normalized_y = (y - min_val) / (max_val - min_val)
    
    return normalized_y

'''
            ############################################################################
            #                             SPLIT FUNCTIONS
            ############################################################################
'''
def load_data_from_file(filename:str, num_col:int = 6): #Divide dataset in input and target
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

def create_autoencoder_splits_unbalanced(X, train_frac:float, valid_frac:float, randomize:bool=True): #Create splits for autoencoder
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

def create_splits_unbalanced(X, y, train_frac:float, valid_frac:float, randomize:bool=True, max_min_into_training:bool = True): #Create splits for Net
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

def save_net(model, epoch:int, epoch_loss_train:float, epoch_loss_val:float, path_pth:str, path_txt:str, r2:float=None, mean_value:float=None, std_value:float=None ,net:str='net'): #Save model for Net
    '''
    Save the "model state" of the Net to a .pth file in the specified path and text file.    
    Args:
        - model: the network.
        - epoch: last epoch.
        - epoch_loss_train: mean loss of the last epoch in training phase.
        - epoch_loss_val: mean loss of the last epoch in validation phase.
        - mean_value: mean of validation output.
        - std_value: std of validation output.
        - r2: score of the consistency of the model. (0 not ok, 1 perfect)
        - path_pth: path where the code will save model parameters/characteristics.
        - path_txt: path where the code will save various characteristics.

    Return:
        Two file with the model state and a text file with some info.
    '''
    #Save the model
    torch.save(model.state_dict(), path_pth)
    #Write a .txt file to the specified path and writes information regarding the epoch and the loss to which
    #the best trained net belongs
    with open(path_txt, "w") as f:
        print(f"Checkpoint {net}:\n\n\tEPOCH:\t{epoch}\n\n\tLOSS TRAIN:\t{epoch_loss_train}\n\n\tLOSS VALIDATION:\t{epoch_loss_val}\n\n\tMEAN VALIDATION:\t{mean_value}\n\n\tSTD VALIDATION:\t{std_value}\n\n\tR2 VALIDATION:\t{r2}", file=f)

def plot_loss(n_epochs:int, net_train_losses:list, net_val_losses:list, result_path:str): #Plot of the train and validation losses
    '''
    Save and show the plot of the losses of the training and validation against epochs.
    
    Args:
        - n_epochs: number of the epochs.
        - net_train_losses: list of the losses of the training phase.
        - net_val_losses: list of the losses of the validation phase.
        - result_path: the path of the Results directory.

    Return:
        A PNG file with an image of the graph of the trend of losses against epochs.    
    '''
    #Plot of the losses for each epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), net_train_losses, label='Training Losses')
    plt.plot(range(1, n_epochs + 1), net_val_losses, label='Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    #Save the image in the result path
    plt.savefig(result_path)

    #Show the image
    plt.show()

def plot_box(data_list:list, labels:list, title:str, x_label:str, y_label:str, save_path:str): #BoxPlot to compare pdf
    '''
    Save and show the plot of the pdf.
    
    Args:
        - data_list: list of data which we want to plot its pdf.
        - labels: list of the labels of each box.
        - title: title of the figure.
        - x_label: x axis label.
        - y_label: y axis label.
        - save_path: figure path.

    Return:
        A PNG file with an image of the box plot.    
    '''
    #Figure
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_list, labels=labels, meanline=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def plot_pdf(target_data, output_data, label1:str, label2:str, title:str, path:str): #Plot to compare pdf
    '''
    Save and show the plot of the pdf.
    
    Args:
        - target_data: data of the target or real data.
        - output_data: data of the output or artificial data
        - label1: target/real label.
        - label2: output/artificial label.
        - title: title of the figure.
        - path: figure path.

    Return:
        A PNG file with an image of the pdf curves plot.    
    '''
    #Move the output_data tensor from GPU to CPU
    target_data_np = target_data.detach().cpu().numpy()
    output_data_np = output_data.detach().cpu().numpy()
    
    #Calculate histogram to estimate PDF
    hist_target, bins_target = np.histogram(target_data_np, bins='auto', range=(target_data_np.min(), target_data_np.max()))
    hist_out, bins_out = np.histogram(output_data_np, bins='auto', range=(output_data_np.min(), output_data_np.max()))

    #Normalize the histograms to get the respective PDFs
    pdf_target = hist_target / hist_target.sum()
    pdf_out = hist_out / hist_out.sum()

    #Calculate midpoints between bins to get PDF curves
    bin_centers_target = (bins_target[1:] + bins_target[:-1]) / 2
    bin_centers_out = (bins_out[1:] + bins_out[:-1]) / 2

    #Plot the two PDF curves with different colors
    plt.plot(bin_centers_target, pdf_target, color='b', label=label1)
    plt.plot(bin_centers_out, pdf_out, color='r', label=label2)
    plt.xlabel('Data values')
    plt.ylabel('Pdf')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    #Save
    plt.savefig(path)      
    #Show the figure
    plt.show()

def plot_initial_data_afterDR(x_values, y_values, title:str, path:str, reduce_number_data:str = '0', sample_interval:int=100): #Plot to visualize data after preprocessing
    '''
    Save and show the plot of the preprocessed data.
    
    Args:
        - x_values: preprocessed input.
        - y_values: preprocessed output.
        - title: title of the figure.
        - path: figure path.
        - reduce_number_data: if it is 1, we plot not all data but a reduced number of them.
        - sample_interval: the number of samples for which only 1 is plotted.

    Return:
        A PNG file with an image of the plot-points of the preprocessed data.    
    '''
    #Convert x_values and y_values to numpy arrays
    x_values = x_values.clone().detach().cpu().numpy()
    y_values = y_values.clone().detach().cpu().numpy()
    
    #Print
    if reduce_number_data == '1':
        #Consider every sample_interval sample
        x_values = x_values[::sample_interval]
        y_values = y_values[::sample_interval]
    
    #Figure
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, s=0.5, marker='o', label='Points')
    plt.xlabel('Preprocessed Input')
    plt.ylabel('Preprocessed Target')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    #Save figure
    plt.savefig(path)
    #Show
    plt.show()
