import torch
import pandas as pd
import numpy as np 
import os

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

def dataset_reduction(dataframe, OssidoAzoto, SensoreMonossCarbonio, Temp, UmiditàRel, SensCompOrg, MonossidoCarbonio):
    # Seleziona le colonne 13 e 14 dal dataframe
    dataframe_reduced = dataframe.loc[:, [f"{OssidoAzoto}",f"{SensoreMonossCarbonio}",f"{Temp}",f"{UmiditàRel}",f"{SensCompOrg}",f"{MonossidoCarbonio}"]]
    
    return dataframe_reduced

def load_csv_file(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

def create_file_csv(dataframe, filename):
    # Salvo il mio dataframe in un nuovo file .csv
    dataframe.to_csv(filename, index=False)

def cleaning_dataset_function(dataframe):
    ### Rimuovo i record con valori NaN o null o non validi ###
    #Creando una copia esplicita del DataFrame, puoi assicurarti che le modifiche vengano apportate sull'oggetto originale e non su una copia, evitando così l'errore "SettingWithCopyWarning"
    dataframe_copy = dataframe.copy()
    #rimozione dati non validi
    dataframe_copy.dropna(inplace=True) 

    ### Converto le colonne nel datatype float ###
    dataframe_copy.loc[:, "NOx(GT)"] = dataframe_copy.loc[:, "NOx(GT)"].astype(float)
    dataframe_copy.loc[:, "PT08.S1(CO)"] = dataframe_copy.loc[:, "PT08.S1(CO)"].astype(float)
    dataframe_copy.loc[:, "T"] = dataframe_copy.loc[:, "T"].str.replace(',', '.').astype(float)
    dataframe_copy.loc[:, "RH"] = dataframe_copy.loc[:,"RH"].str.replace(',', '.').astype(float)
    dataframe_copy.loc[:, "PT08.S2(NMHC)"] = dataframe_copy.loc[:, "PT08.S2(NMHC)"].astype(float)
    dataframe_copy.loc[:, "CO(GT)"] = dataframe_copy.loc[:,"CO(GT)"].str.replace(',', '.').astype(float)

    ### Verifico che i dati ricadano siano diversi da -200
    condizione1 = (dataframe_copy["NOx(GT)"] != -200) & (dataframe_copy["PT08.S1(CO)"] != -200) & (dataframe_copy["T"] != -200) \
        & (dataframe_copy["RH"] != -200) & (dataframe_copy["PT08.S2(NMHC)"] != -200) & (dataframe_copy["CO(GT)"] != -200)

    # Rimuovo i record che non soddisfano entrambe le condizioni
    dataframe = dataframe_copy[condizione1]

    #Rindicizzo le righe sennò avrei dei "buchi" nelle indicizzazioni dovuto al fatto che ho droppato dei record
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe
    
def load_data_from_file(filename):
    # Load data dal file .csv che abbiamo appositamente preprocessato
    data = load_csv_file(filename)

    # splitting data in inputs and targets
    X = data.iloc[:, :5].to_numpy().astype(np.float32) 
    y = data.iloc[:, -1].to_numpy().astype(np.float32)

    return X, y

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

def normalize_input_data(X, mean=None, std=None, min_val=None, max_val=None):   
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



