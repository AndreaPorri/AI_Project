'''
                    ############################################################

                                            AUTOENCODER

                    ############################################################

In this file, an autoencoder is defined and trained, from which the encoder part will subsequently be extracted.

Structure of our auto encoder:
- Encoders:
    - Input layer with 5 neurons (corresponding to input size)
    - Hidden layer with 3 neurons (size reduction)
    - Hidden layer with 1 neuron (compact representation)
- Decoders:
    - Hidden layer with 3 neurons (dimensional expansion)
    - Output layer with 5 neurons (corresponding to output size)

The goal of the autoencoder is to learn a compact representation of the input data in the 1-neuron hidden layer
and then reconstruct the original input in the 5-neuron output layer.

This will be done because, before passing the data into our MLP (Multi-Layer Perceptron), a coherent dimensional
reduction of our normalized data will be necessary to obtain a compact 1D representation of the latter.
This will help reduce the complexity of the problem while preserving the most important information of our data
(create a compact representation of the data). So after that, in the main.py file the encoder part of this 
pre-trained autoencoder will be extracted.




EXECUTION EXAMPLE:
    - Standard training:   python autoencoder.py --print_info='1'
    

MUST: First of all, you have to execute the this file and then proceed with the execution of the training part of the main.py !!! 

'''
#Import needed libraries, classes and functions
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from functions import *
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
    Function required for using additional terminal commands necessary for selecting specific choices needed for the correct execution of the autoencoder.
    The main commands are:

        - --pathConfiguratorYaml: takes in input the yaml file path as a string.

        - --print: takes in input an integer parameter (0 or 1) which tell if you want to print some checks (optional command).
    
    """
    #Creates an ArgumentParser object from the argparse library used for analyzing the passed arguments from the command line:
    parser = argparse.ArgumentParser(description='Process some command line arguments.') 

    #Add the arguments to the ArgumentParser object:
    parser.add_argument('--pathConfiguratorYaml', type=str, default='config_file.yaml',
                        help='Insert the path of Yaml file containing all the parameters of the project')
    parser.add_argument('--print_info', type=str, default='0', choices=['1','0'],
                        help='You have to chose if you want to print some controls and checks, "1" if you want.')
        
   
    args = parser.parse_args() #This line of code reads the arguments passed from the command line and compares them with the options defined using the 
                                #add_argument method of the ArgumentParser object. parse_args returns a Namespace object containing the values of the
                                #arguments specified by the user on the command line.
    return args


### YAML CONFIGURATION FILE ###
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
    #DATASET/RESULT DIRECTORY
    dataroot = yaml_configurator['dataroot']
    results_path = yaml_configurator['results_path']
    #PATH
    reduce_dataset_autoencoder_path = yaml_configurator['reduce_dataset_autoencoder_path']
    path_pth_autoencoder = yaml_configurator['path_pth_autoencoder']
    path_txt_autoencoder = yaml_configurator['path_txt_autoencoder']
    image_loss_path = yaml_configurator['image_loss_path']
    #HYPERPARAMETER
    loss_function = yaml_configurator['loss_function']
    optimizer = yaml_configurator['optimizer']
    n_epochs = yaml_configurator['n_epochs']
    lr = yaml_configurator['lr']
    batch_size = yaml_configurator['batch_size']

    #The function returns all these variablesas a tuple, returning all the parameters as individual variables:
    return dataroot,results_path,reduce_dataset_autoencoder_path,path_pth_autoencoder,path_txt_autoencoder,image_loss_path,loss_function,optimizer,n_epochs,lr,batch_size

            
            
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

        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_shape, 3),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(3, enc_shape),
            nn.Tanh()
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 3),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(3, in_shape)
        )

    def initialize_weights(self): #Weights initialization
        '''
        Intelligently initializing the weights of the network, randomly but normalized.
        '''
        #Cycling on layers
        for module in self.modules():

            if isinstance(module, nn.Linear):
                #Make sure to initialize all linear layers.
                nn.init.normal_(module.weight.data, 0.0, 0.02) #weights
                nn.init.constant_(module.bias.data, 0.0) #0 bias
    
    def forward(self, x): #Calculate the forward outputs
        '''
        Function for calculating the step forward, to obtain the output of the network.
        '''
        
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
            List of the average validation losses and outputs.
        '''
        #Initialize device, avarage loss, outputs and targets lists
        device = next(model.parameters()).device
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
      
    def training(model, input_train_tensor, input_val_tensor, path_pth, path_txt, image_loss_path, print_info, loss_function_choice, optimizer_choice, n_epochs, lr, batch_size): #Training of the autoencoder
        '''
        Training the autoencoder.

        Args:
            model: the autoencoder object of that class.
            input_train_tensor: input tensor for training.
            input_val_tensor: input tensor for validation.
            result_path: path where to save results.
            path_pth: path where to save the characteristics and parameters of the network.
            path_txt: path to a text file to save some network characteristics.
            image_loss_path: path to save the loss trend image.
            print_info: identifier to make further prints.
            loss_function_choice: identifier of the loss we want to select.
            optimizer_choice: identifier of the optimizer we want to select.
            n_epochs: number of cycles over the entire training set.
            lr: learning rate.
            batch_size: dimensionality of a single mini-batch.
            

        Returns:
            net_train_losses and net_val_losses which are the lists containing the average losses of each epoch.
        '''
        ### STARTING ###
        print('\n\nTraining the network...')
        sleep(2)
        print('\n\nLoading the dataset and creating the dataloader...\n\n\n')
        sleep(2)
        
        ### DATA ###
        #Normalization data
        input_train_tensor, mean_train, std_train = real_norm_input(input_train_tensor)
        input_val_tensor, _, _ = real_norm_input(input_val_tensor, mean_train, std_train)

        #### DATALOADER ####
        #Convert input_tensor to a PyTorch TensorDataset
        dataset_train = torch.Tensor(input_train_tensor)
        dataset_val = torch.Tensor(input_val_tensor)

        #Create a DataLoader to handle minibatching
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)
        
        #### DEVICE ####
        #Assume that all the network parameters are on the same device
        device = next(model.parameters()).device
        
        #Print
        if print_info == '1':
            # Print to verify that everything has been executed correctly, that the objects have been instantiated and the device is defined
            print('\n\n\nSome useful info about our objects and dimension:\n')
            print(f'\t- The device selected for the training is: {device}')
            print(f'\t- Training dataset object = {dataset_train}')
            print(f'\t- Training dataloader object = {dataloader_train}')
            print(f'\t- Validation dataset object = {dataset_val}')
            print(f'\t- Validation dataloader object = {dataloader_val}')
            sleep(8)

        #### LOSS FUNCTION SELECTION ####
        print('\n\nLoading the selected loss function...')

        #Function for multi-selection loss function
        if loss_function_choice == 'mse':  #Mean Square Error
            loss_function = nn.MSELoss()
        elif loss_function_choice == 'mae':  #Mean Absolute Error
            loss_function = nn.L1Loss()
        else:
            raise ValueError('Invalid choice of loss function')
        sleep(2)

    
        #### OPTIMIZER SELECTION AND APPLICATION ####
        print('\n\nLoading the selected optimizer...')

        #Seelect the optimizers for the training
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
            print('\n\n\nSome autoencoder hyperparameters of the network:\n')
            print(f'\t- Learning rate: {lr}')
            print(f'\t- Epochs: {n_epochs}')
            print(f'\t- Batch size: {batch_size}')   
            print(f'\t- Loss function selected: {loss_function_choice}')
            print(f'\t- Optimizer: {optimizer_choice}')
            sleep(8)

        
        
        #############################################################################

                                #STARTING TRAINING LOOP

        #############################################################################
        
        print('\n\nStarting the training loop...')

        #Telling the network we are going to train it (and not to simply evaluate it)
        model.train()
     
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

                #Print of the mini-batch characteristics and cumulate the losses value of the mini-batches for each epoch
                with torch.no_grad():
                    #Accumulate the train loss of the mini-batch for the epochs
                    epoch_loss_train += loss_train.item()

                    #Print of the minibatch
                    print('\tepoch:{}, minibatch: {}, loss_train: {:.4f}'.format(epoch+1, idx_batch, loss_train))
                    

            #Loss of the training epoch
            epoch_loss_train /= len(dataloader_train)

            #Save in list all the training losses
            net_train_losses.append(epoch_loss_train)

            ### VALIDATION ###
            #Validation of the epoch
            loss_valid= Autoencoder.validate(model, dataloader_val, loss_function)

            #Save in list all the validation losses
            net_val_losses.append(loss_valid)
        

        #### SAVING TRAINED AUTOENCODER ####
        save_net(model, epoch, epoch_loss_train, loss_valid, path_pth, path_txt,net='Autoencoder')
        #### PLOT OF TRAINING AND VALIDATION LOSSES ####
        plot_loss(n_epochs, net_train_losses, net_val_losses, image_loss_path)

        return net_train_losses, net_val_losses  

    def predict_encoder(self, x): #Prediction of the encoder
        """
        Predict the encoder output using the trained autoencoder.

        Args:
            x: Input tensor for prediction.

        Returns:
            The encoded representation of the input tensor.
        """
        #Ensure the model is in evaluation mode
        self.eval()

        #Move the input tensor to the same device as the model
        device = next(self.parameters()).device

        #Put the input on the device
        x = x.to(device)

        #Encoding the 5D input into 1D output
        encoded = self.encoder(x)

        return encoded

if __name__ == '__main__':

    ### TERMINAL ###
    args = parse_command_line_arguments() #extracts the arguments from the command line and saves them in the 'args' object
    
    ### YAML ###
    pathConfiguratorYaml = args.pathConfiguratorYaml #extracts the path of the YAML configuration file from the command line and saves it in a variable
    #We assign the values returned by the function, that is the values in the tuple, to the respective variables
    dataroot,results_path,reduce_dataset_autoencoder_path,path_pth_autoencoder,path_txt_autoencoder,image_loss_path,loss_function,optimizer,n_epochs,lr,batch_size = load_hyperparams(pathConfiguratorYaml)

    ### DIRECTORY AUTOENCODER RESULTS ###
    createDirectory(results_path)
    createDirectory(f'{results_path}/Autoencoder')

    ##########################################################################################                                    
                                        
                            ### LOAD AND PREPROCESS DATASET ###

    ##########################################################################################
    
    #Loading dataframe from csv file
    dataframe = pd.read_csv(dataroot, sep=';') #The .csv file uses ; as a separator instead of space

    #Resizing dataset from original one (take only 6 columns)
    dataset_reduced_dirty = dataset_reduction(dataframe,'NOx(GT)','PT08.S1(CO)','T','RH','PT08.S2(NMHC)','CO(GT)')
    
    #Cleaning the reduced dataset
    dataset_reduced = cleaning_dataset_function(dataset_reduced_dirty).iloc[:, :5]

    #Print
    if args.print_info == '1':
        #Check dimensionality of the new dataset
        print('The dimensionality of the reduced and dirty datset is:',dataset_reduced_dirty.shape)
        print('The dimensionality of the reduced datset is:',dataset_reduced.shape)
        sleep(6)
        #Show the dataset first rows
        print('\n\nSome rows of the reduced dataset: \n',dataset_reduced.head(5))
        sleep(10)

        
        ### PLOT 3D GRAPH DATA ###        
        #Create a PCA object with 3 components, instead of 5
        pca = PCA(n_components=3)
        
        #Fit and transform the data on a reduced dimensional representation
        reduced_data_PLOT = pca.fit_transform(dataset_reduced)
        
        #Create a dataframe with this new data
        reduced_df_PLOT = pd.DataFrame(data=reduced_data_PLOT, columns=['Dimension 1', 'Dimension 2','Dimension 3'])

        # Create a 3D scatter plot.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_df_PLOT['Dimension 1'], reduced_df_PLOT['Dimension 2'], reduced_df_PLOT['Dimension 3'])
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('3D Visualization using PCA')
        plt.savefig(f'{results_path}/Autoencoder/input_data_3D_plot.png')
        plt.show()
      
    ### SAVE THE NEW DATASET ###
    #Save the reduced dataset in a csv file
    create_file_csv(dataset_reduced,reduce_dataset_autoencoder_path)

    ### DEVICE ###
    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
    ##########################################################################################                                    
                                        
                            ### TRAINING AND VALIDATION SETS ###

    ##########################################################################################

    #Load the data from the new csv file
    data = load_csv_file(reduce_dataset_autoencoder_path)

    ### CREATE THE SETS ###
    #Splitting data in training, validation e test sets(which has 0 samples)
    data_X_train, data_X_val, _ = create_autoencoder_splits_unbalanced(data, 0.85, 0.15)
        
    #Print
    if args.print_info == '1':
        #Check dimensionality of the new sets
        print('\n\nThe dimensionality training set:',data_X_train.shape)
        print('The dimensionality validation set:',data_X_val.shape)
        sleep(6)
        
       
    ##########################################################################################                                    
                                        
                          ### INITIALIZATION OF THE AUTOENCODER ###

    ##########################################################################################

    #Create Autoencoder architecture and transform the elements in double (avoid an error)
    autoencoder = Autoencoder().double()

    #Initializing weights
    autoencoder.initialize_weights()

    #Move the model to the selected device
    autoencoder.to(my_device)

    #Print
    if args.print_info == '1':
        #Autoencoder architecture
        print(f'\n\n\t\t- Autoencoder Architecture:')
        print(autoencoder)
        sleep(10)

    #################################################################

                        #### TRAINING PHASE ####

    #################################################################

    #Move the training data to the selected device
    data_X_train = data_X_train.to(my_device)
    data_X_val = data_X_val.to(my_device)

    #Training the network
    Autoencoder.training(autoencoder, data_X_train, data_X_val, path_pth_autoencoder, path_txt_autoencoder, image_loss_path, args.print_info, loss_function, optimizer, n_epochs, lr, batch_size)
