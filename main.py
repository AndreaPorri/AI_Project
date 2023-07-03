import pandas as pd
from dataset_function import *
import torch
import torch.nn as nn
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])

    def add_hidden_layer(self, hidden_size):
        self.layers.insert(-1, nn.Linear(self.layers[-1].out_features, hidden_size))
        self.layers.insert(-1, nn.Tanh())

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and not isinstance(layer, nn.Tanh):
                x = torch.sigmoid(x)  # Applica sigmoid all'ultimo layer prima dell'output
        return x




if __name__ == "__main__":

                                    ### PREPROCESS DATASET ###

    #Caricamento dataframe da file .csv
    filename = "C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/AirQualityUCI.csv"
    dataframe = pd.read_csv(filename, sep=";")
    #print(dataframe.columns)

    #Dataset check caratteristiche:
    #check_dataset(dataframe)

    #Ridimensionamento dataset
    dataset_reduced = dataset_reduction(dataframe,"T","RH")
    #print('Il dataset senza le eliminazioni è:\n ',dataset_reduced)
    #print('la dimensionalità è: ',dataset_reduced.shape)
    
    #Pulitura dataset ridotto
    dataset_reduced = check_dataset_format(dataset_reduced)
    #print('Il dataset con le eliminazioni è:\n ',dataset_reduced)
    #print('la dimensionalità è: ',dataset_reduced.shape)

    #Salvo il dataset ridotto
    create_file_csv(dataset_reduced,"C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/dataset_reduced.csv")
    
    #Device configuration
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        ### DEFINIZIONE DELLA MLP ###

    #Carichiamo i dati dal nuovo file .csv e dividiamo in input e output
    data_X, data_y = load_data_from_file("C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/dataset_reduced.csv")
    
    #Splitting data in training, validation e test sets
    data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test = create_splits_unbalanced(data_X, data_y, 0.7, 0.15)
    '''
    ### CONTEGGIO NEGATIVI PRE NORMALIZZAZIONE ###
    # Converti il tensore Torch in un tensore booleano con valori True per i numeri negativi
    negative_mask = (data_X_train < 0)

    # Conta il numero di valori True nel tensore booleano
    num_negatives = torch.sum(negative_mask).item()

    print("Numero di numeri negativi nel tensore:", num_negatives)
    '''    
    # normalizing data input
    data_X_train, m, s, max, min= normalize_input_data(data_X_train)
    data_X_val, _, _, _, _ = normalize_input_data(data_X_val, m, s)
    data_X_test, _, _, _, _ = normalize_input_data(data_X_test, m, s)

    
    
    #Stampe forma dati
    #print("DATI:\n",data_X_train)
    #print(f"MEDIA:{m}, STD:{s}, MAX:{max}, MIN:{min}")
    
    '''
    ### CONTEGGIO NEGATIVI POST NORMALIZZAZIONE + DOMANDA ###
    # Converti il tensore Torch in un tensore booleano con valori True per i numeri negativi
    negative_mask = (data_X_train < 0)

    # Conta il numero di valori True nel tensore booleano
    num_negatives = torch.sum(negative_mask).item()

    print("Numero di numeri negativi nel tensore:", num_negatives)
    
    #Domanda: è normale che io abbia più numeri negativi post normalizzazione? In teoria questa dovrebbe mantenere la distribuzione relativa dei dati nel proprio intervallo.
    #        Oppure conviene fissare max = 100 e min = -100 e quindi viene una distribuzione sbilanciata ma che mantiene i segni? normalize_input_data_100
    '''
    
    
    #Ensure that we keep track of the mean and std used to normalize the data
    torch.save([m, s], 'C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/normalizers_hyperparam.pth')

    
 
    
    

      
   