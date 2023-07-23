import pandas as pd
import torch
import torch.nn as nn
import argparse
import yaml
from dataset_function import *
from encoder import *
from MLP import *
from data_generation import *
from time import sleep


torch.manual_seed(42)

def parse_command_line_arguments():
    """
    Function required for using additional terminal commands necessary for selecting specific parameters needed for the correct execution of DCGAN.
    The main commands are:

        - --pathConfiguratorYaml: takes in input the yaml file path as a string (mandatory command)

        - --config: takes in input "train" or "eval" for selecting the training or the evaluation phase, if not specified the default value is "train" (optional command).
                    Choices: 
                    - train ---> training phase
                    - eval ---> evaluation evaluation

        - --print: takes in input an integer parameter (0 or 1) which tell if you want to print some checks
    """

    #Creates an ArgumentParser object from the argparse library used for analyzing the passed arguments from the command line:
    parser = argparse.ArgumentParser(description='Process some command line arguments.') 

    #Add the arguments to the ArgumentParser object:
    parser.add_argument('--pathConfiguratorYaml', type=str, required=True,
                        help='Insert the path of Yaml file containing all the parameters of the project')
    parser.add_argument('--config', type=str, default='train', choices=['train', 'eval'],
                        help='You have to chose the training or evaluation configuration')
    parser.add_argument('--print', type=int, default=1, choices=[1,0],
                        help='You have to chose if you want to print some controls and checks')
        
   
    args = parser.parse_args() #This line of code reads the arguments passed from the command line and compares them with the options defined using the 
                                #add_argument method of the ArgumentParser object. parse_args returns a Namespace object containing the values of the
                                #arguments specified by the user on the command line.
    return args

def yamlParser(path_yaml_file: str):  #takes in input the yaml file path
    """
    Function required for reading a YAML file and saving its content into a variable. The input will be passed via the terminal and will be the path
    of the .yaml file.
    """
    
    with open(path_yaml_file, "r") as stream: #.yaml file is opened in read mode and called "stream"
        yaml_parser = yaml.safe_load(stream) #saves the contents of the file in a variable
    return yaml_parser

def load_hyperparams(pathConfiguratorYaml: str):
    """
    This function simplifies the access to the model hyperparameters values in the YAML file. In practice, the function loads and returns the 
    hyperparameters specified in the YAML file. The function has only one input: the string of the absolute path of the file YAML.
    """
    
    yaml_configurator = yamlParser(pathConfiguratorYaml) #takes in input the path of the YAML file. It returns a Namespace containing the key-value pairs
                                                          #corresponding to the parameters specified in the file

    #Each variable takes the value corresponding to the field specified in the Namespace:
    dataroot = yaml_configurator["dataroot"]
    reduce_dataset_path=yaml_configurator["reduce_dataset_path"]
    

    #The function returns all these variablesas a tuple, returning all the parameters as individual variables:
    return dataroot,reduce_dataset_path


################################################################################

                                #MAIN CODE START

################################################################################



if __name__ == "__main__":
    
    ### Terminal ###
    args = parse_command_line_arguments() #extracts the arguments from the command line and saves them in the "args" object
    
    ### Yaml file ###
    pathConfiguratorYaml = args.pathConfiguratorYaml #extracts the path of the YAML configuration file from the command line and saves it in a variable
    #We assign the values returned by the function, that is the values in the tuple, to the respective variables
    dataroot,reduce_dataset_path = load_hyperparams(pathConfiguratorYaml)


##########################################################################################                                    
                                    
                        ### LOAD AND PREPROCESS DATASET ###

##########################################################################################
    #Loading dataframe from .csv file
    dataframe = pd.read_csv(dataroot, sep=";") #The .csv file uses ; as a separator instead of space
    
    #Resizing dataset from original one (I take only 6 columns)
    dataset_reduced = dataset_reduction(dataframe,"NOx(GT)","PT08.S1(CO)","T","RH","PT08.S2(NMHC)","CO(GT)")
    
    #Print
    if args.print == 1:
        print('The dimensionality of the reduced and dirty datset is:',dataset_reduced.shape)
    
    #Cleaning the reduced dataset
    dataset_reduced = cleaning_dataset_function(dataset_reduced)

    #Print
    if args.print == 1:
        print('The dimensionality of the reduced datset is: ',dataset_reduced.shape)
        print('Some rows of the reduced dataset: \n',dataset_reduced.head(5))
        sleep(10)

    #Save the reduced dataset
    create_file_csv(dataset_reduced,reduce_dataset_path)
    
    #Device configuration
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################                                    
                                    
                        ### TRAINING, VALIDATION AND TEST SETS ###

##########################################################################################

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
    
    #########################

        #Eval phase

    ##########################
    #Directory per salvare le immagini
    createDirectory('D:/Results/Images')

    #Combine input and output tensors
    test_dataset = torch.Tensor(data_X_test)
    print("Dimensioni del tensore Test dataset:", len(test_dataset))
    # Dataloader
    dataloader_test = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)
    #Stampa della pdfy della mia NN
    outputs = Net_training.predict(net, dataloader_test)
    print("Dimensioni del tensore output:", outputs.shape)

    #Stampo la PDFY
    Net_training.plot_pdfy(outputs, 'D:/Results/Images/pfdy.png')

    ##### Genero samples da GMM #####

    # PARAMETERS:
    # Esempio di parametri per una GMM con 32 gaussiane
    num_gaussians = 32
    # Probabilità di mescolamento (deve sommare a 1 dopo essere normalizzata)
    mixing_parameters = np.random.rand(num_gaussians)
    mixing_parameters /= np.sum(mixing_parameters)
    # Medie delle gaussiane
    means = np.random.randint(0, 100, num_gaussians)
    # Deviazioni standard delle gaussiane
    std_deviations = np.random.randint(1, 30, num_gaussians)
    # Number of new artificial samples to generate
    n_samples = 50000

    #Salvataggio parametri:
    # Salvataggio dei parametri in un file .txt nel percorso specificato
    file_path = "D:/Results/Images/saved_GMM_parameters.txt"
    # Combiniamo i parametri in un array 2D per la scrittura
    # Il primo valore di ogni riga rappresenta la gaussiana corrispondente
    parameters = np.column_stack((np.arange(num_gaussians), mixing_parameters, means, std_deviations))
    # Scrive i parametri nel file .txt
    np.savetxt(file_path, parameters, header="Gaussian_Index Mixing_Parameter Mean Standard_Deviation", fmt='%d %.6f %.6f %.6f')                  

    #Genero:
    new_samples = generate_gaussian_mixture_samples(mixing_parameters, means, std_deviations, n_samples)
    #Print the generated samples
    print(new_samples)
    #Normalizzo:
    new_samples, _, _ = normalize_input_data(new_samples)
    print(new_samples)
    #Converto in float
    new_samples = new_samples.to(torch.float)

    #Dataloader
    dataloader_artificial = DataLoader(TensorDataset(new_samples), batch_size=8, shuffle=False, drop_last=False)

    ###### TESTO ARTIFICIAL INPUT ######

    # Assuming we already have 'net' defined and trained
    trained_mlp = net[1]
    print(trained_mlp)

    #Vado in inferenza
    outputs_artificial = Net_training.predict(trained_mlp, new_samples)

    #Stampo la PDFY artificiale
    Net_training.plot_pdfy(outputs_artificial, 'D:/Results/Images/ARTIFICIAL_pfdy.png')



    

     