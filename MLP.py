import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from functions import *
from autoencoder import *
from tqdm import trange
from time import sleep

                                ##############################################

                                        ### MLP ARCHITECTURE CLASS ###

                                ##############################################
class MLP(nn.Module):
    """
    Creates the main MLP for regression.

    Args:
        input_size [int] : input layer shape
        output_size [int] : desired output layer shape
        hidden_sizes [list]: desired hidden layers shapes
    Returns:
        The class to define structure of the MLP.
    """
    def __init__(self, input_size, output_size, hidden_sizes): #Architecture definition
        super(MLP, self).__init__()

        #Hidden layer list
        hidden_layers = []
        #Input size next layer, initialize it with the input_size
        prev_size = input_size

        #Iterative creation of hidden layers with the following structure and specification given to the class.
        for hidden_size in hidden_sizes:
            hidden_layers.append(nn.Linear(prev_size, hidden_size))
            hidden_layers.append(nn.Tanh())
            hidden_layers.append(nn.Dropout(0.1,inplace=False)),
            
            #Add the new input size for the next layer.
            prev_size = hidden_size

        #Defining architecture with Sequential using hidden layer list + output layer defined inside
        self.layers = nn.Sequential(
            *hidden_layers,
            nn.Linear(prev_size, output_size),
            nn.Sigmoid(),       
        )

    def initialize_weights(self):#Weights initialization
        '''
        Intelligently initializing the weights of my network, randomly but normalized.
        '''
        #Cycling on layers
        for layer in self.layers:
            #Make sure to initialize all linear layers.
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0.0, 0.02) #weights
                nn.init.constant_(layer.bias.data, 0.0) #0 bias

    def forward(self, inputs): #Calculate the forward outputs
        return self.layers(inputs)



                                ##############################################

                                             ### NET TRAINING ###

                                ##############################################

class Net_training(torch.nn.Module):
    def __init__(self):
        super(Net_training, self).__init__()
        
    def predict(net, dataloader):
        '''
        Make predictions using the neural network.

        Args:
            net: the neural network of the class Net.
            dataloader: dataloader object that provides batches of data.

        Returns:
            Network output.
        '''
        #Retrieve the device where the parameters are
        device = next(net.parameters()).device  # we assume that all the network parameters are on the same device
        #lista output
        outputs = []

        with torch.no_grad():
            net.eval()

            for X_minibatch in dataloader:
                # Move the mini-batch to the device
                X_minibatch = X_minibatch.to(device)
                # Compute the output of the mini-batch
                output = net(X_minibatch)
                outputs.append(output)

        return torch.cat(outputs, dim=0)

    def validate(net, dataloader_val, criterion, MIN, MAX):
        '''
        Validate the neural network.

        Args:
            net: the neural network of the class Net.
            dataloader_val: dataloader object that provides batches of validation data.
            criterion: the loss function.

        Returns:
            Average validation loss.
        '''
        #Initialize device and avarage loss
        device = next(net.parameters()).device
        val_loss = 0.0
        outputs = []

        #Evaluation mode
        with torch.no_grad():

            #Let network in evaluation mode
            net.eval()

            #Loop on mini-batch
            for X_val_minibatch, y_val_minibatch in dataloader_val:

                #Conversion double to float
                X_val_minibatch = X_val_minibatch.float()

                #Mini-batch loading of inputs and outputs onto the selected device
                X_val_minibatch = X_val_minibatch.to(device)
                y_val_minibatch = y_val_minibatch.to(device)

                #Predictions
                output = net(X_val_minibatch)

                '''#Convert the [0,1] output in real output
                output = output * (MAX - MIN) + MIN'''

                #Calculate the losses of the minibatch
                loss = criterion(output, y_val_minibatch)

                #Accumulate the loss
                val_loss += loss.item()

                #Save outputs
                outputs.append(output)

        #Calculate the average validation loss
        avg_val_loss = val_loss / len(dataloader_val)

        return avg_val_loss, outputs
    
    def training(net, X_train, X_val, y_train, y_val, path_pth, path_txt, results_path, result_path_net, print_info, n_epochs, lr, minibatch_size, loss_function_choice, optimizer_choice, MIN, MAX):
        """
        Train a neural network for multiple epochs.

        Args:
            net: the neural network.
            X_train: training data (one-example-per-row).
            y_train: training targets.
            X_val: validation data (one-example-per-row).
            y_val: validation targets.
            path_pth_net: path where to save the characteristics and parameters of the network.
            path_txt_net: path to a text file to save some network characteristics.
            result_path: path of the resuls directory.
            result_path_net: path where to save results image.
            print_info: identifier to make further prints.
            n_epochs: number of epochs.
            lr: learning rate.
            minibatch_size: size of the training mini-batches.
            loss_function_choice: choice of loss function.
            optimizer_choice: choice of optimizer.

        Returns:
            The two arrays with the R2s on training and validation data computed during the training epochs.
        """

        print("Training the network...")
        sleep(2)
        print('\n\nLoading the dataset and creating the dataloader...\n\n\n')

        #Print
        if print_info == '1':
            #Some information about our data
            print("Validation and training dataset shape")
            print("\tX_train shape:", X_train.shape)
            print("\ty_train shape:", y_train.shape)
            print("\tX_val shape:", X_val.shape)
            print("\ty_val shape:", y_val.shape)
            print("\n\n\n")
            sleep(6)

        #Combine input and output tensors
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

        #Dataloader
        dataloader_train = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=False, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False, drop_last=True)

        #We assume that all the network parameters are on the same device
        device = next(net.parameters()).device

        #Print to verify that everything has been executed correctly, that the objects have been instantiated and the device is defined:
        if print_info == '1':
            print("Device, datasets and dataloader:\n")
            print(f'\tThe device selected for the training is: {device}')
            print(f'\tTraining dataset object = {train_dataset}')
            print(f'\tTraining dataloader object = {dataloader_train}')
            print(f'\tValidation dataset object = {val_dataset}')
            print(f'\tValidation dataloader object = {dataloader_val}')
            print(f'\tNumber of datasets training and validation samples = {len(train_dataset),len(val_dataset)}')
            sleep(4)

        
        #LOSS FUNCTION SELECTION
        print('\n\nLoading the selected loss function...')

        #Function for multi-selection loss function
        if loss_function_choice == 'mse':  # Mean Square Error
            loss_function = nn.MSELoss()
        elif loss_function_choice == 'mae':  # Mean Absolute Error
            loss_function = nn.L1Loss()
        else:
            raise ValueError("Invalid choice of loss function")

        sleep(2)

        
        #OPTIMIZER SELECTION AND APPLICATION
        print('\n\nLoading the selected optimizer...')

        #Setup Adam or SGD optimizers for both the generator and the discriminator
        if optimizer_choice == "adam":
            optimizer = optim.Adam(net.parameters(), lr=lr)
        elif optimizer_choice == "sgd":
            optimizer = optim.SGD(net.parameters(), lr=lr)
        elif optimizer_choice == "rmsprop":
            optimizer = optim.RMSprop(net.parameters(), lr=lr)
        else:
            raise ValueError("Invalid choice of optimizer")

        sleep(2)

        #Print
        if print_info == '1':
            #PRINT OF THE TRAINING FEATURES:
            print('\n\n\tSome hyperparameters of the network:\n')
            print(f'\t\t- Learning rate: {lr}')
            print(f'\t\t- Epochs: {n_epochs}')
            print(f'\t\t- Batch size: {minibatch_size}')   
            print(f'\t\t- Loss function selected: {loss_function_choice}')
            print(f'\t\t- Optimizer: {optimizer_choice}')

            sleep(6)

        
        #############################################################################

                                #STARTING TRAINING LOOP

        #############################################################################
        
        print("\n\n\tStarting the training loop...")

        #Telling the network we are going to train it (and not to simply evaluate it)
        net.train()

        #Create the directory for theAutoencoder results
        createDirectory(results_path)
        createDirectory(f'{results_path}/NET')

        #Define some useful quantities
        net_train_losses = [] #list of each loss of the epoch
        net_val_losses = [] #list of the loss of the epochs

        ######### LOOP ON EPOCHS ##########
        for e in trange(n_epochs):  #loop on epochs
            #Initializes the training phase loss for the current epoch
            loss_value_train = 0.
            
            ### LOOP ON MINI-BATCHES ###
            for nb, (X_minibatch, y_minibatch) in enumerate(dataloader_train):  #loop on mini-batches
                # Clearing the previously computed gradients (are saved in memory, each iteration we need to reset the gradients)
                optimizer.zero_grad()

                #Conversion double to float
                X_minibatch = X_minibatch.float()
                #Separate input and output
                X_minibatch = X_minibatch.to(device)
                y_minibatch = y_minibatch.to(device)

                #Calculate the outputs
                output = net(X_minibatch) #going forward, "net" is a callable object
                
                '''#Convert the [0,1] output in real output
                output = output * (MAX - MIN) + MIN'''

                #Calculate the loss on batches and save it:
                loss_value_on_minibatch = loss_function(output, y_minibatch)
            
                #Calculate gradients for all mini-batch on the net with the backward step. The backward() operation is used 
                #for calculating the gradient of the error with respect to its parameters:
                loss_value_on_minibatch.backward()
                
                #PARAMETERS UPDATE
                #This function is used to update the parameters based on the calculated gradients. This process of
                #updating parameters is called "optimization" and is performed using an optimization algorithm such as Adam,SGD or other:
                optimizer.step()

                #Print of the loss and updating the globale loss value of a single epoch
                with torch.no_grad():
                    #Print of the minibatch
                    print("\tepoch:{}, minibatch: {}, loss_train: {:.4f}".format(e + 1, nb, loss_value_on_minibatch))
                    
                    #Accumulate the loss in the training phase loss
                    loss_value_train += loss_value_on_minibatch.item()

            #loss della training epoch
            loss_value_train /= len(dataloader_train)

            #Save the net losses of each batch within the lists defined earlier
            net_train_losses.append(loss_value_train)
            
            ### VALIDATION ###
            # Validation of the epoch
            loss_valid, output_val = Net_training.validate(net, dataloader_val, loss_function, MIN, MAX)

            #Append all losses
            net_val_losses.append(loss_valid)

        #Combine the tensors in the list output_val into a single tensor along the samples dimension
        output_val_tensor = torch.cat(output_val, dim=0)
        
        ### MEAN AND STD OF THE OUTPUT ###
        #Calculate the VALIDATION mean of the output data
        mean_value = torch.mean(output_val_tensor)

        # Calculate the VALIDATION standard deviation of the output data
        std_value = torch.std(output_val_tensor)

        #### SAVING TRAINED NET ####
        save_net(net,n_epochs, loss_value_train, loss_valid, mean_value, std_value, path_pth, path_txt)

        #### PLOT OF TRAINING AND VALIDATION LOSSES ####
        plot_loss(n_epochs, net_train_losses, net_val_losses, result_path_net)

        return net_train_losses,net_val_losses
    
    def plot_pdfy(output_data,path:str):
        # Move the output_data tensor from GPU to CPU
        output_data = output_data.cpu().numpy()
        # Increase the number of bins for a smoother histogram
        #num_bins = 200
        # Calcola l'istogramma per stimare la PDF
        hist, bin_edges = np.histogram(output_data, bins='auto', density=True)

        # Grafica l'istogramma e la PDF
        plt.hist(output_data, bins='auto', density=True, alpha=0.7, color='blue', label='Histogram')
        plt.plot(0.5*(bin_edges[1:] + bin_edges[:-1]), hist, color='red', label='PDF')
        plt.xlabel('Output')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Probability Density Function (PDF) of Output Data')
        plt.grid(True)
        

        # Salva l'immagine nel percorso specificato
        plt.savefig(path)
    
        # Mostra il grafico
        plt.show()

    def plot_pdfy_prova(output_data, path: str):
        # Converti il tensore in una lista di liste
        data_list = output_data.tolist()

        # Crea un DataFrame da data_list
        df = pd.DataFrame(data_list)

        # Print della dimensionalità
        print("Dimensionalità del DataFrame:", df.shape)

        # Print dei nomi delle colonne
        print("Nomi delle colonne:", df.columns)

        # Density Plot and Histogram of the data in df
        sns.histplot(df, kde=True, bins=int(180/5), color='darkblue', 
                    edgecolor='black', linewidth=1.5)

        plt.title('Density Plot and Histogram')
        plt.xlabel('Data')
        plt.ylabel('Density')
         
        # Salva l'immagine nel percorso specificato
        plt.savefig(path)
        
        plt.show()


        

    