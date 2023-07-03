import torch
import numpy as np
from main import load_csv_file

def dataset_reduction(dataframe, colonna_input, colonna_target):
    # Seleziona le colonne 13 e 14 dal dataframe
    dataframe_reduced = dataframe.loc[:, [f"{colonna_input}",f"{colonna_target}"]]
    
    return dataframe_reduced



def create_file_csv(dataframe, filename):
    # Salvo il mio dataframe in un nuovo file .csv
    dataframe.to_csv(filename, index=False)


def check_dataset_format(dataframe):
    ### Rimuovo i record con valori NaN o null o non validi ###
    #Creando una copia esplicita del DataFrame, puoi assicurarti che le modifiche vengano apportate sull'oggetto originale e non su una copia, evitando così l'errore "SettingWithCopyWarning"
    dataframe_copy = dataframe.copy()
    #rimozione dati non validi
    dataframe_copy.dropna(inplace=True) 

    ### Controllo che il formato delle colonne restanti sia corretto ###
    #Definisco il formato desiderato utilizzando un'espressione regolare
    formato_desiderato = r'^(-?\d+,\d+)$'

    # Verifico che tutti i valori delle due colonne rispettino il formato desiderato
    is_valid = dataframe_copy["T"].str.match(formato_desiderato) & dataframe_copy["RH"].str.match(formato_desiderato)

    # Rimuovo i record che non rispettano il formato desiderato
    dataframe_copy = dataframe_copy[is_valid]


    ### Converto le colonne nel datatype float ###
    dataframe_copy.loc[:, "T"] = dataframe_copy.loc[:, "T"].str.replace(',', '.').astype(float)
    dataframe_copy.loc[:, "RH"] = dataframe_copy.loc[:,"RH"].str.replace(',', '.').astype(float)


    ### Verifico che i dati ricadano in un range ragionevole, altrimenti verranno eliminati ###
    # Verifico che la prima colonna sia compresa tra -100° e 60°
    condizione1 = (dataframe_copy["T"] >= -100) & (dataframe_copy["T"] <= 60)

    # Verifico che la seconda colonna sia compresa tra 0% e 100%
    condizione2 = (dataframe_copy["RH"] >= 0) & (dataframe_copy["RH"] <= 100)

    # Rimuovo i record che non soddisfano entrambe le condizioni
    dataframe = dataframe_copy[condizione1 & condizione2]

    #Rindicizzo le righe sennò avrei dei "buchi" nelle indicizzazioni dovuto al fatto che ho droppato dei record
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe
    


def load_data_from_file(filename):
    # Load data dal file .csv che abbiamo appositamente preprocessato
    data = load_csv_file(filename)

    # splitting data in inputs and targets
    X = data.iloc[:, 0].to_numpy().reshape((-1, 1)).astype(np.float32) 
    y = data.iloc[:, 1].to_numpy().reshape((-1, 1)).astype(np.float32)

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
        The NumPy tensors X,y (data and targets) for training, validation, and test sets, respectively.
    """

    # Randomize data
    if randomize:
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]

    # Calculate split sizes
    n = X.shape[0]
    train_size = int(n * train_frac)
    valid_size = int(n * valid_frac)

    # Split data
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+valid_size], y[train_size:train_size+valid_size]
    X_test, y_test = X[train_size+valid_size:], y[train_size+valid_size:]

    return torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_val), torch.from_numpy(y_val), torch.from_numpy(X_test), torch.from_numpy(y_test)



def normalize_input_data(X, mean=None, std=None):
    #Calcolo della media e deviazione standard
    if mean is None:
            mean = X.mean()
    if std is None:
           std = X.std()
    # Adjust the range to -1 to 1
    min_val = X.min()
    max_val = X.max()
    normalized_X = 2 * (X - min_val) / (max_val - min_val) - 1

    return normalized_X, mean, std, max_val, min_val

def normalize_input_data_100(X, mean=None, std=None):
    #Calcolo della media e deviazione standard
    if mean is None:
            mean = X.mean()
    if std is None:
           std = X.std()
    # Adjust the range to -1 to 1
    min_val = -100
    max_val = 100
    normalized_X = 2 * (X - min_val) / (max_val - min_val) - 1

    return normalized_X, mean, std, max_val, min_val


def normalize_output_data(y, min_val=None, max_val=None):
    #Calcolo del minimo e massimo
    if min_val is None:
        min_val = X.min()
    if max_val is None:
        max_val = X.max()
    normalized_X = (X - min_val) / (max_val - min_val)
    return normalized_X



