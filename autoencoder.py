'''
                    ############################################################

                                            AUTOENCODER

                    ############################################################

In this file, an autoencoder is defined and trained, from which the encoder part will subsequently be extracted.

Structure of our autoecoder:
- Encoder:
    - Input layer con 5 neuroni (corrispondente alla dimensione dell'input)
    - Hidden layer con 3 neuroni (riduzione dimensionale)
    - Hidden layer con 1 neurone (rappresentazione compatta)
- Decoder:
    - Hidden layer con 3 neuroni (espansione dimensionale)
    - Output layer con 5 neuroni (corrispondente alla dimensione dell'output)

The goal of the autoencoder is to learn a compact representation of the input data in the 1-neuron hidden layer
and then reconstruct the original input in the 5-neuron output layer.

This will be done because, before passing the data into our MLP (Multi-Layer Perceptron), a coherent dimensional
reduction of our normalized data will be necessary to obtain a compact 1D representation of the latter.
This will help reduce the complexity of the problem while preserving the most important information of our data
(create a compact representation of the data). So after that, in the main.py file the encoder part of this 
pre-trained autoencoder will be extracted.

'''
#Import needed libraries, classes and functions
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from dataset_function import *
import argparse
import yaml
from tqdm import trange
from main import createDirectory
from time import sleep

#Let's put a seed (not necessary)
torch.manual_seed(0)

### TERMINAL ###
def parse_command_line_arguments():
    """ 
    Function required for using additional terminal commands necessary for selecting specific parameters needed for the correct execution of DCGAN.
    The main commands are:

        - --pathConfiguratorYaml: takes in input the yaml file path as a string (mandatory command)

        - --print: takes in input an integer parameter (0 or 1) which tell if you want to print some checks
    
    """
    #Creates an ArgumentParser object from the argparse library used for analyzing the passed arguments from the command line:
    parser = argparse.ArgumentParser(description='Process some command line arguments.') 

    #Add the arguments to the ArgumentParser object:
    parser.add_argument('--pathConfiguratorYaml', type=str, required=True,
                        help='Insert the path of Yaml file containing all the parameters of the project')
    parser.add_argument('--print_info', type=str, default='0', choices=['1','0'],
                        help='You have to chose if you want to print some controls and checks')
        
   
    args = parser.parse_args() #This line of code reads the arguments passed from the command line and compares them with the options defined using the 
                                #add_argument method of the ArgumentParser object. parse_args returns a Namespace object containing the values of the
                                #arguments specified by the user on the command line.
    return args


### YAML CONFIGURATION FIILE ###
def yamlParser(path_yaml_file: str):  #takes in input the yaml file path
    """
    Function required for reading a YAML file and saving its content into a variable. The input will be passed via the terminal and will be the path
    of the .yaml file.
    """
    
    with open(path_yaml_file, 'r') as stream: #.yaml file is opened in read mode and called 'stream'
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
    dataroot = yaml_configurator['dataroot']
    reduce_dataset_autoencoder_path = yaml_configurator['reduce_dataset_autoencoder_path']
    path_pth_autoencoder = yaml_configurator['path_pth_autoencoder']
    path_txt_autoencoder = yaml_configurator['path_txt_autoencoder']
    result_path_autoencoder = yaml_configurator['result_path_autoencoder']
    loss_function = yaml_configurator['loss_function']
    optimizer = yaml_configurator['optimizer']
    n_epochs = yaml_configurator['n_epochs']
    lr = yaml_configurator['lr']
    batch_size = yaml_configurator['batch_size']

    #The function returns all these variablesas a tuple, returning all the parameters as individual variables:
    return dataroot,reduce_dataset_autoencoder_path,path_pth_autoencoder,path_txt_autoencoder,result_path_autoencoder,loss_function,optimizer,n_epochs,lr,batch_size

            
            
            ######################################################################

                        ### AUTOENCODER ARCHITECTURE AND TRAINING CLASS ###

            ######################################################################


class Autoencoder(nn.Module):
    """ 
    Creates the main autoencoder for regression.

    Args:
        in_shape [int] : input shape
        enc_shape [int] : desired encoded shape
    Returns:
        The class to define the structure and useful methods for training and validating the autoencoder.
    """

    def __init__(self, in_shape = 5, enc_shape = 1): #Autoencoder structure
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_shape, 3),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(3, enc_shape)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 3),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(3, in_shape)
        )

    def initialize_weights(self): #Weights initialization
        '''
        Intelligently initializing the weights of my network, randomly but normalized.
        '''
        #Cycling on layers
        for module in self.modules():

            if isinstance(module, nn.Linear):
                #Make sure to initialize all linear layers.
                nn.init.normal_(module.weight.data, 0.0, 0.02) #weights
                nn.init.constant_(module.bias.data, 0.0) #0 bias
    
    def forward(self, x): #Calculate the forward outputs
        # Encoding
        encoded = self.encoder(x)        
        # Decoding
        reconstructed = self.decoder(encoded)
        
        return reconstructed
        
    def validate(model, dataloader_val, criterion): #Validation of the autoencoder
        '''
        Validate the autoencoder.

        Args:
            model: the autoencoder object of that class.
            dataloader_val: dataloader object that provides batches of validation data.
            criterion: the loss function selected.

        Returns:
            List of the average validation losses.
        '''
        #Recover device through model parameters.
        device = next(model.parameters()).device
        
        #Initializing epoch loss with floats.
        val_loss = 0.

        #In validation/evaluation don't have to keep track of gradients.
        with torch.no_grad():
            #Let the machine in evaluation mode
            model.eval()

            #Starting the loop ogn the validation set
            for X_val_minibatch in dataloader_val:
                
                #Loading in mini-batch input.
                X_val_minibatch = X_val_minibatch.to(device)
                
                #Predictions
                output = model(X_val_minibatch)

                #Calculating the minibatch losses
                loss = criterion(output, X_val_minibatch)

                #Accumulate the loss of each minibatch
                val_loss += loss.item()

        #Calculate the average validation loss
        avg_val_loss = val_loss / len(dataloader_val)

        return avg_val_loss
      
    def training(model, input_train_tensor, input_val_tensor, result_path, path_pth, path_txt, print_info, loss_function_choice, optimizer_choice, n_epochs, lr, batch_size): #Training of the autoencoder
        '''
        Training the autoencoder.

        Args:
            model: the autoencoder object of that class.
            input_train_tensor: input tensor for training.
            input_val_tensor: input tensor for validation.
            result_path: path where to save the loss trend image on training and validation.
            path_pth: path where to save the characteristics and parameters of the network.
            path_txt: path to a text file to save some network characteristics.
            print_info: identifier to make further prints.
            loss_function_choice: identifier of the loss we want to select.
            optimizer_choice: identifier of the optimizer we want to select.
            n_epochs: number of cycles over the entire training set.
            lr: learning rate.
            batch_size: dimensionality of a single mini-batch.
            

        Returns:
            net_train_losses and net_val_losses which are the lists containing the average losses of each epoch.
        '''
        #Print
        if print_info == '1':
            print('Validation and training dataset shape')
            print('Train set shape:', input_train_tensor.shape)
            print('Validation set shape:', input_val_tensor.shape)
            
            print('\n\n\n')

        ### STARTING ###
        print('Training the network...')
        sleep(2)
        print('\n\n\tLoading the dataset and creating the dataloader...\n\n\n')
        sleep(2)

        #### DATALOADER ####
        #Convert input_tensor to a PyTorch TensorDataset
        dataset_train = torch.Tensor(input_train_tensor)
        dataset_val = torch.Tensor(input_val_tensor)

        #Create a DataLoader to handle minibatching
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)
        
        #### DEVICE ####
        #We assume that all the network parameters are on the same device
        device = next(model.parameters()).device
        
        #Print
        if print_info == '1':
            # Print to verify that everything has been executed correctly, that the objects have been instantiated and the device is defined
            print(f'\t\tThe device selected for the training is: {device}')
            print(f'\t\tTraining dataset object = {dataset_train}')
            print(f'\t\tTraining dataloader object = {dataloader_train}')
            print(f'\t\tValidation dataset object = {dataset_val}')
            print(f'\t\tValidation dataloader object = {dataloader_val}')
            print(f'\t\tNumber of datasets training and validation samples = {len(dataset_train),len(dataset_val)}')
            sleep(4)

        #### LOSS FUNCTION SELECTION ####
        print('\n\n\tLoading the selected loss function...')

        # Function for multi-selection loss function
        if loss_function_choice == 'mse':  #Mean Square Error
            loss_function = nn.MSELoss()
        elif loss_function_choice == 'mae':  #Mean Absolute Error
            loss_function = nn.L1Loss()
        else:
            raise ValueError('Invalid choice of loss function')
        sleep(2)

    
        #### OPTIMIZER SELECTION AND APPLICATION ####
        print('\n\n\tLoading the selected optimizer...')

        # Setup Adam or SGD optimizers for both the generator and the discriminator
        if optimizer_choice == 'adam': #Adam
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_choice == 'sgd': #Stochastic Gradient Descent 
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_choice == 'rmsprop': #Root Mean Square Propagation
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError('Invalid choice of optimizer')
        sleep(2)

        #Print
        if print_info == '1':
            #PRINT OF THE TRAINING FEATURES:
            print('\n\n\tSome hyperparameters of the network:\n')
            print(f'\t\t- Learning rate: {lr}')
            print(f'\t\t- Optimizer: {optimizer_choice}')
            print(f'\t\t- Epochs: {n_epochs}')
            print(f'\t\t- Batch size: {batch_size}')   
            print(f'\t\t- Loss function selected: {loss_function_choice}')
            sleep(12)

        #############################################################################

                                #STARTING TRAINING LOOP

        #############################################################################
        
        print('\n\n\tStarting the training loop...')

        #Telling the network we are going to train it (and not to simply evaluate it)
        model.train()

        #Create the directory for theAutoencoder results
        createDirectory('D:/Results')
        createDirectory('D:/Results/Autoencoder')

        #Initialization of the list of training and validation losses of the all epochs
        net_train_losses = [] 
        net_val_losses = []


        ######### LOOP ON EPOCHS ##########
        for epoch in trange(n_epochs):
            #Initializes the loss of the training phase for the current epoch
            epoch_loss_train = 0.0

            ### LOOP ON MINI-BATCHES ###
            for idx_batch, X_minibatch in enumerate(dataloader_train):
                #Clearing the previously computed gradients (are saved in memory, each iteration we need to reset the gradients)
                optimizer.zero_grad()

                #Input and target data on the device
                X_minibatch = X_minibatch.to(device)
                
                #Forward outputs
                output = model(X_minibatch)
             
                #Calculate the loss
                loss_train = loss_function(output, X_minibatch)
                
                #Backward pass
                loss_train.backward()
                
                #Update model parameters
                optimizer.step()

                #Print of the losses and updating the globale loss value of a single epoch
                with torch.no_grad():
                    #Print of the minibatch
                    print('\tepoch:{}, minibatch: {}, loss_train: {:.4f}'.format(epoch+1, idx_batch, loss_train))
                    
                    # Accumulate the train loss for the epoch
                    epoch_loss_train += loss_train.item()

            #loss della training epoch
            epoch_loss_train /= len(dataloader_train)

            #Save the net losses of each batch within the lists defined earlier
            net_train_losses.append(epoch_loss_train)


            #Validation of the epoch
            loss_valid = Autoencoder.validate(model, dataloader_val, loss_function)

            #Hang up all the losses:
            net_val_losses.append(loss_valid)
        
        #### SAVING TRAINED AUTOENCODER ####
        save_net(model, epoch, epoch_loss_train, loss_valid, path_pth, path_txt)

        #### PLOT OF TRAINING AND VALIDATION LOSSES ####
        plot_loss(n_epochs, net_train_losses, net_val_losses, result_path)

        return net_train_losses,net_val_losses


if __name__ == '__main__':

    ### Terminal ###
    args = parse_command_line_arguments() #extracts the arguments from the command line and saves them in the 'args' object
    
    ### Yaml file ###
    pathConfiguratorYaml = args.pathConfiguratorYaml #extracts the path of the YAML configuration file from the command line and saves it in a variable
    #We assign the values returned by the function, that is the values in the tuple, to the respective variables
    dataroot, reduce_dataset_autoencoder_path, path_pth_autoencoder, path_txt_autoencoder, result_path_autoencoder, loss_function, optimizer, n_epochs, lr, batch_size = load_hyperparams(pathConfiguratorYaml)

    
    ##########################################################################################                                    
                                        
                            ### LOAD AND PREPROCESS DATASET ###

    ##########################################################################################
    
    #Loading dataframe from csv file
    dataframe = pd.read_csv(dataroot, sep=';') #The .csv file uses ; as a separator instead of space

    #Resizing dataset from original one (I take only 6 columns)
    dataset_reduced = dataset_reduction(dataframe,'NOx(GT)','PT08.S1(CO)','T','RH','PT08.S2(NMHC)','CO(GT)')
    
    #Print
    if args.print_info == '1':
        print('The dimensionality of the reduced and dirty datset is:',dataset_reduced.shape)
    
    #Cleaning the reduced dataset
    dataset_reduced = cleaning_dataset_function(dataset_reduced).iloc[:, :5]

    #Print
    if args.print_info == '1':
        print('The dimensionality of the reduced datset is: ',dataset_reduced.shape)
        print('Some rows of the reduced dataset: \n',dataset_reduced.head(5))
        sleep(10)
    
    #Save the reduced dataset
    create_file_csv(dataset_reduced,reduce_dataset_autoencoder_path)

    ### DEVICE ###
    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##########################################################################################                                    
                                        
                            ### TRAINING AND VALIDATION SETS ###

    ##########################################################################################

    #Load the data from the new csv file
    data = load_csv_file(reduce_dataset_autoencoder_path)

    #Splitting data in training, validation e test sets(which has 0 samples)
    data_X_train, data_X_val, _ = create_encoder_splits_unbalanced(data, 0.85, 0.15)
       
    ##########################################################################################                                    
                                        
                            ### INITIALIZATION OF THE NETWORK ###

    ##########################################################################################

    #Create Autoencoder architecture 
    autoencoder = Autoencoder().double()

    #Initializing weights
    autoencoder.initialize_weights()

    #Move the model to the selected device
    autoencoder.to(my_device)

    #Print
    if args.print_info == '1':
        print(f'\n\n\t\t- Autoencoder Architecture:')
        print(autoencoder)
        sleep(10)

    #################################################################

                        #### TRAINING PHASE ####

    #################################################################

    #Move the training data to the selected device
    data_X_train = data_X_train.to(my_device)
    data_X_val = data_X_val.to(my_device)

    #### TRAINING LOOP ####
    #Training the network
    Autoencoder.training(autoencoder, data_X_train, data_X_val, result_path_autoencoder, path_pth_autoencoder, path_txt_autoencoder, args.print_info, loss_function, optimizer, n_epochs, lr, batch_size)