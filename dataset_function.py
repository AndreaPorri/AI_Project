import torch
import pandas as pd
import numpy as np 
import os

def dataset_reduction(dataframe, *args:str, num_col = 6):
    '''
    It reduces the dimensionality of the dataset to only the columns we are interested in.

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

def cleaning_dataset_function(dataframe):
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
        # Check if the column contains string values before using the .str accessor
        if dataframe_copy[col].dtype == 'object': # Assuming the string columns have 'object' dtype
            dataframe_copy[col] = dataframe_copy[col].str.replace(',', '.').astype(float)

    ### Verify that the fallout data is different from -200 ###
    #Create a boolean array with True for each row where all columns are different from -200
    condition = (dataframe_copy != -200).all(axis=1)
    
    #Remove records that do not meet both conditions
    dataframe = dataframe_copy[condition]

    #Re-index the rows otherwise I would have "holes" in indexing due to dropping records
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe

def load_data_from_file(filename, num_col = 6):
    #Load data by the .csv file that we have specially preprocessed
    data = load_csv_file(filename)

    #Splitting data in inputs and targets
    X = torch.tensor(data.iloc[:, :(num_col-1)].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

    return X, y

def createDirectory(nameDirectory: str):
    """
    This function is used to create folders, taking as input the absolute path of the folder we want to create (path/foldername). In our case, the path is passed
    through the yaml file. The folders that will be created are those for saving the generator/discriminator training, and during the evaluation phase, the folder
    where the images produced by the generator will be saved.
    """
    if not os.path.exists(f'{nameDirectory}'):  #checks if the folder with that path already exists
        os.mkdir(f'{nameDirectory}')  #if it does not exist creates it

def check_dataset(dataframe):
    #Numero righe e colonne dataset
    print(dataframe.shape)
    #Lista nomi colonne
    lista_colonne= list(dataframe.columns)
    print(f"Nomi delle colonne 1:{lista_colonne}\n")
    #Stampa delle prime 10 righe dei dataset
    print(f"Dataset first lines:\n{dataframe.head(10)}\n\n")  

def load_csv_file(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

def create_file_csv(dataframe, filename):
    # Salvo il mio dataframe in un nuovo file .csv
    dataframe.to_csv(filename, index=False)
    
def create_splits_unbalanced(X, y, train_frac, valid_frac, randomize=True):
    """
    Creates (randomized) training, validation, test data splits.

    Args:
        X: dataset (one example per row).
        y: ground truth labels (vector).
        train_frac: the fraction of data to be used in the training set.
        valid_frac: the fraction of data to be used in the validation set.
        randomize (optional): randomize the data before splitting them.

    Returns:
        The PyTorch tensors X, y (data and targets) for training, validation, and test sets, respectively.
    """

    if randomize:
        # Convert to PyTorch tensors and randomize data
        indices = torch.randperm(len(X))
        X, y = torch.tensor(X)[indices], torch.tensor(y)[indices]

    n = len(X)
    train_size, valid_size = int(n * train_frac), int(n * valid_frac)

    return X[:train_size], y[:train_size], X[train_size:train_size+valid_size], y[train_size:train_size+valid_size], X[train_size+valid_size:], y[train_size+valid_size:]

def create_encoder_splits_unbalanced(X, train_frac, valid_frac, randomize=True):
    """
    Creates (randomized) training, validation, test data splits.

    Args:
        X: dataset (one example per row).
        y: ground truth labels (vector).
        train_frac: the fraction of data to be used in the training set.
        valid_frac: the fraction of data to be used in the validation set.
        randomize (optional): randomize the data before splitting them.

    Returns:
        The PyTorch tensors X, y (data and targets) for training, validation, and test sets, respectively.
    """

    if randomize:
        # Convert to NumPy array and randomize data
        X = X.sample(frac=1).to_numpy()

    n = len(X)
    train_size, valid_size = int(n * train_frac), int(n * valid_frac)

    X_train = torch.tensor(X[:train_size])
    X_val = torch.tensor(X[train_size:train_size+valid_size])
    X_test = torch.tensor(X[train_size+valid_size:])

    return X_train, X_val, X_test

def normalize_input_data(X, min_val=None, max_val=None):   
    # Adjust the range to -1 to 1
    if min_val is None:
            min_val = X.min()
    if max_val is None:
            max_val = X.max()        
    #Normalization
    normalized_X = 2 * (X - min_val) / (max_val - min_val) - 1

    return normalized_X, max_val, min_val

def normalize_output_data(y, min_val=None, max_val=None):
    #Calcolo del minimo e massimo
    if min_val is None:
        min_val = y.min()
    if max_val is None:
        max_val = y.max()
    
    normalized_y = (y - min_val) / (max_val - min_val)
    return normalized_y



