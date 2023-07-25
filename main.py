'''



'''


#Import needed libraries, classes and functions
import pandas as pd
import torch
import torch.nn as nn
import argparse
import yaml
from functions import *
from autoencoder import *
from MLP import *
from data_generation import *
from time import sleep

#Let's put a seed (not necessary)
torch.manual_seed(42)

### TERMINAL ###
def parse_command_line_arguments():
    """
    Function required for using additional terminal commands necessary for selecting specific parameters needed for the correct execution of DCGAN.
    The main commands are:

        - --pathConfiguratorYaml: takes in input the yaml file path as a string (mandatory command)

        - --config: takes in input "train" or "eval" for selecting the training or the evaluation phase, if not specified the default value is "train" (optional command).
                    Choices: 
                    - train ---> training phase
                    - eval ---> evaluation evaluation

        - --print_info: takes in input an integer parameter (0 or 1) which tell if you want to print some checks
    """

    #Creates an ArgumentParser object from the argparse library used for analyzing the passed arguments from the command line:
    parser = argparse.ArgumentParser(description='Process some command line arguments.') 

    #Add the arguments to the ArgumentParser object:
    parser.add_argument('--pathConfiguratorYaml', type=str, required=True,
                        help='Insert the path of Yaml file containing all the parameters of the project')
    parser.add_argument('--config', type=str, default='train', choices=['train', 'eval'],
                        help='You have to chose the training or evaluation configuration')
    parser.add_argument('--print_info', type=str, default='0', choices=['1','0'],
                        help='You have to chose if you want to print some controls and checks')
        
   
    args = parser.parse_args() #This line of code reads the arguments passed from the command line and compares them with the options defined using the 
                                #add_argument method of the ArgumentParser object. parse_args returns a Namespace object containing the values of the
                                #arguments specified by the user on the command line.
    return args

### YAML ###
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
    #DATASET/RESULT DIRECTORY
    dataroot = yaml_configurator["dataroot"]
    results_path = yaml_configurator['results_path']
    #PATH
    reduce_dataset_path = yaml_configurator["reduce_dataset_path"]
    path_pth_autoencoder = yaml_configurator["path_pth_autoencoder"]
    path_pth_net = yaml_configurator['path_pth_net']
    path_txt_net = yaml_configurator['path_txt_net']
    result_path_net = yaml_configurator['result_path_net']
    #MLP ARCHITECTURE
    input_size_mlp = yaml_configurator["input_size_mlp"]
    output_size_mlp = yaml_configurator["output_size_mlp"]
    hidden_layers_mlp = yaml_configurator["hidden_layers_mlp"]
    #HYPERPARAMETER
    epochs_net = yaml_configurator["epochs_net"]
    learning_rate_net = yaml_configurator["learning_rate_net"]
    mini_batch_size_net = yaml_configurator["mini_batch_size_net"]
    loss_function_net = yaml_configurator["loss_function_net"]
    optimizer_net = yaml_configurator["optimizer_net"]
    
    

    #The function returns all these variablesas a tuple, returning all the parameters as individual variables:
    return dataroot,results_path,reduce_dataset_path,path_pth_autoencoder,path_pth_net,path_txt_net,result_path_net,input_size_mlp,output_size_mlp,hidden_layers_mlp,epochs_net,learning_rate_net,mini_batch_size_net,loss_function_net,optimizer_net


################################################################################

                                #MAIN CODE START

################################################################################


### START ###
if __name__ == "__main__":
    
    ### Terminal ###
    args = parse_command_line_arguments() #extracts the arguments from the command line and saves them in the "args" object
    
    ### Yaml file ###
    pathConfiguratorYaml = args.pathConfiguratorYaml #extracts the path of the YAML configuration file from the command line and saves it in a variable
    #We assign the values returned by the function, that is the values in the tuple, to the respective variables
    dataroot, results_path, reduce_dataset_path, path_pth_autoencoder, path_pth_net, path_txt_net, result_path_net, input_size_mlp, output_size_mlp, hidden_layers_mlp, epochs_net, learning_rate_net, mini_batch_size_net, loss_function_net, optimizer_net = load_hyperparams(pathConfiguratorYaml)


    ##########################################################################################                                    
                                        
                            ### LOAD AND PREPROCESS DATASET ###

    ##########################################################################################
    #Loading dataframe from csv file
    dataframe = pd.read_csv(dataroot, sep=";") #The .csv file uses ; as a separator instead of space
    
    #Resizing dataset from original one (I take only 6 columns)
    dataset_reduced = dataset_reduction(dataframe,"NOx(GT)","PT08.S1(CO)","T","RH","PT08.S2(NMHC)","CO(GT)")
    
    #Print
    if args.print_info == '1':
        print('The dimensionality of the reduced and dirty datset is:',dataset_reduced.shape)
    
    #Cleaning the reduced dataset
    dataset_reduced = cleaning_dataset_function(dataset_reduced)

    #Print
    if args.print_info == '1':
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

    #Load the data from the new CSV file and split into input and target
    data_X, data_y = load_data_from_file(reduce_dataset_path)
    
    #Splitting and normalize data in training, validation e test sets
    data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test = create_splits_unbalanced(data_X, data_y, 0.7, 0.15)

    ##########################################################################################                                    
                                        
                            ### INITIALIZATION OF THE NETWORK ###

    ##########################################################################################

                                ###############################
                                        ### ENCODER ###
                                ###############################
    #Loading the previously trained and saved autoencoder model
    autoencoder = Autoencoder() #Create the autoencoder object
    autoencoder.load_state_dict(torch.load(path_pth_autoencoder)) #Load the model onto that object

    #We define a new network with only the 5-3-1 portion, which is the encoding part used for dimensionality reduction.
    encoder = autoencoder.encoder
    encoder.to(my_device) #Put the network onto the selected device.

    #Print
    if args.print_info == '1':
        print(f'\n\n\t\t- Encoder Architecture:')
        print(encoder)
        sleep(10)


                                ###############################
                                         ### MLP ###
                                ###############################
   
    #Create MLP architecture
    mlp = MLP(input_size_mlp, output_size_mlp, hidden_layers_mlp)
    
    #Initializing weights
    mlp.initialize_weights()

    #Move the model to the selected device
    mlp.to(my_device)

    #Print
    if args.print_info == '1':
        print(f'\n\n\t\t- MLP Architecture:')
        print(mlp)
        sleep(10)

    ##################################################################################
                                #Neural Netwoek
    ##################################################################################

    #Make the composition of the two previously defined networks and obtain the starting architecture
    net = nn.Sequential(encoder,mlp)
    
    #Move the model to the selected device 
    net.to(my_device)

    #Print
    if args.print_info == '1':
        print(f'\n\n\t\t- Network Architecture:')
        print(net)

    #################################################################

                        #### TRAINING PHASE ####

    #################################################################
    if args.config == 'train':
        #Move the training data to the selected device
        data_X_train = data_X_train.to(my_device)
        data_y_train = data_y_train.to(my_device)
        data_X_val = data_X_val.to(my_device)
        data_y_val = data_y_val.to(my_device)
        data_X_test = data_X_test.to(my_device)
        data_y_test = data_y_test.to(my_device)

        #### TRAINING LOOP ####
        #Training the network
        Net_training.training(net, data_X_train, data_X_val, data_y_train, data_y_val, path_pth_net, path_txt_net, results_path, result_path_net, args.print_info, epochs_net, learning_rate_net, mini_batch_size_net, loss_function_net, optimizer_net)
        
        #########################

            #Eval phase

        ##########################
        #Directory per salvare le immagini
        createDirectory(results_path)

        #Combine input and output tensors
        test_dataset = torch.Tensor(data_X_test)
        print("Dimensioni del tensore Test dataset:", len(test_dataset))
        # Dataloader
        dataloader_test = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)
        #Stampa della pdfy della mia NN
        outputs = Net_training.predict(net, dataloader_test)
        print("Dimensioni del tensore output:", outputs.shape)

        #Stampo la PDFY
        Net_training.plot_pdfy(outputs, 'D:/Results/NET/pfdy.png')

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
        n_samples = 100000

        #Salvataggio parametri:
        # Salvataggio dei parametri in un file .txt nel percorso specificato
        file_path = "D:/Results/NET/saved_GMM_parameters.txt"
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
        new_samples = normalize_input_data(new_samples)
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
        Net_training.plot_pdfy(outputs_artificial, 'D:/Results/NET/ARTIFICIAL_pfdy.png')



    

     