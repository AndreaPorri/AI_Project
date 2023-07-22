import pandas as pd
import torch
import torch.nn as nn
from dataset_function import *
from encoder import *
from MLP import *
from data_generation import *
from time import sleep


torch.manual_seed(42)


if __name__ == "__main__":

                                    ### PREPROCESS DATASET ###
    #Creo la cartella risultati se non esiste gia
    #Caricamento dataframe da file .csv
    filename = "C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/AirQualityUCI.csv"
    dataframe = pd.read_csv(filename, sep=";")
    #print(dataframe.columns)


    #Ridimensionamento dataset
    dataset_reduced = dataset_reduction(dataframe,"NOx(GT)","PT08.S1(CO)","T","RH","PT08.S2(NMHC)","CO(GT)")
    #print('Il dataset senza le eliminazioni è:\n ',dataset_reduced)
    print('la dimensionalità è: ',dataset_reduced.shape)
    #print(dataset_reduced.columns)
    #sleep(120)
    
    #Pulitura dataset ridotto
    dataset_reduced = cleaning_dataset_function(dataset_reduced)
    #print('Il dataset con le eliminazioni è:\n ',dataset_reduced)
    print('la dimensionalità ridotta è: ',dataset_reduced.shape)
    sleep(5)

    #Salvo il dataset ridotto
    create_file_csv(dataset_reduced,"C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/dataset_reduced.csv")
    
    #Device configuration
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        ### DEFINIZIONE DELLA MLP ###

    #Carichiamo i dati dal nuovo file .csv e dividiamo in input e target
    data_X, data_y = load_data_from_file("C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/dataset_reduced.csv")
    
    #Splitting data in training, validation e test sets
    data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test = create_splits_unbalanced(data_X, data_y, 0.7, 0.15)
    
    # normalizing data input
    data_X_train, _, _ = normalize_input_data(data_X_train)
    data_X_val, _, _ = normalize_input_data(data_X_val)
    data_X_test, _, _ = normalize_input_data(data_X_test)
    
    #Normalizziamo pure gli output
    data_y_train = normalize_output_data(data_y_train)
    data_y_val = normalize_output_data(data_y_val)
    data_y_test = normalize_output_data(data_y_test)

    ################################################################################ 
                                #INIZIALIZZO LA RETE
    ################################################################################ 

    #########################
            #ENCODER
    #########################
    # Per caricare il modello
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('D:/Results/Autoencoder/net_ep_300.pth'))

    # Definisci una nuova rete con solo la porzione 5-3-1
    # Puoi ottenere i primi tre layer tramite list slicing
    encoder = autoencoder.encoder
    encoder.to(my_device)

    # Supponiamo che il tuo modello si chiami 'net'
    # Assicurati di averlo definito e inizializzato correttamente prima di utilizzare summary

    input_features = 5

    print(f'\n\n\t\t- Encoder Architecture:')
    print(encoder)

    ##########################################################
                            # MLP
    ##########################################################

    #Neuroni dei miei layer
    input_size = 1
    output_size = 1
    hidden_layers = [4, 2]

    #Creo l'architettura:
    mlp = Net(input_size, output_size,hidden_layers)
    
    #Inizializzo i pesi
    mlp.initialize_weights()

    # Sposto il modello sulla GPU
    mlp.to(my_device)

    print(f'\n\n\t\t- MLP Architecture:')
    print(mlp)

    ##################################################################################
                                #Neural Netwoek
    ##################################################################################

    net = nn.Sequential(encoder,mlp)
    
    # Sposto il modello sulla GPU
    net.to(my_device)

    #Stampo l'architettura:
    print(f'\n\n\t\t- NN Architecture:')
    print(net)

    #################################################################

                        #### TRAINING PHASE ####

    #################################################################

    # Sposto i dati di addestramento sulla GPU
    data_X_train = data_X_train.to(my_device)
    data_y_train = data_y_train.to(my_device)
    data_X_val = data_X_val.to(my_device)
    data_y_val = data_y_val.to(my_device)
    data_X_test = data_X_test.to(my_device)
    data_y_test = data_y_test.to(my_device)

    
    # training the network
    Net_training.training(net, data_X_train, data_X_val, data_y_train, data_y_val, n_epochs=400, lr=0.001, minibatch_size=64, loss_function_choice="mse", optimizer_choice="sgd")
    
    
    #Combine input and output tensors
    test_dataset = torch.Tensor(data_X_test)
    print("Dimensioni del tensore Test dataset:", len(test_dataset))
    # Dataloader
    dataloader_test = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)
    #Stampa della pdfy della mia NN
    outputs = Net_training.predict(net, dataloader_test)
    print("Dimensioni del tensore output:", outputs.shape)

    #Stampo la PDFY
    Net_training.plot_pdfy(outputs)

    ##### Genero samples da GMM #####

    # Example usage:
    mixing_parameters = [91,35,51,55,26,58,82,86,44,89,100,27,6,85,47,18,37,98,65,32,13,8,21,54,74,64,2,78,1,95,29,88]  # Mixing parameters of Gaussian components
    means = [0,49,50,8,68,69,23,34,28,41,56,77,88,16,95,31,7,96,19,87,98,58,89,37,18,12,14,43,25,3,10,60]                 # Means of Gaussian components
    std_deviations = [26,0,87,74,85,56,14,75,98,65,82,77,24,46,67,53,55,18,31,63,71,83,78,99,10,72,40,50,20,39,37,21]          # Standard deviations of Gaussian components
    n_samples = 5000                    # Number of new artificial samples to generate

    #Genero:
    new_samples = generate_gaussian_mixture_samples(mixing_parameters, means, std_deviations, n_samples)
    #Print the generated samples
    print(new_samples)
    #Normalizzo:
    new_samples = normalize_input_data(new_samples)
    print(new_samples)
    #Dataloader
    dataloader_artificial = DataLoader(new_samples, batch_size=8, shuffle=False, drop_last=False)

    ###### TESTO ARTIFICIAL INPUT ######

    # Assuming we already have 'net' defined and trained
    trained_mlp = net[1]
    print(trained_mlp)

    #Vado in inferenza
    outputs_artificial = Net_training.predict(trained_mlp, new_samples)

    #Stampo la PDFY artificiale
    Net_training.plot_pdfy(outputs_artificial)



    

     