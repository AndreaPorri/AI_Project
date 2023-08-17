
'''                        
                        ############################################################

                                            MLP AND TRAINING

                        ############################################################

Introduction:
The provided code offers a flexible implementation of a Multi-Layer Perceptron (MLP) along with functionalities 
for training and making predictions in regression tasks. This code allow easy modifications to the network's structure
simpy provide a differente list of hidden layers dimensions.

The MLP architecture consists of the following layers:
    - Input layer: 1 neuron (corresponding to the input size).
    - Hidden layer: 4 neurons, using the Parametric ReLU activation function and dropout regularization.
    - Hidden layer: 2 neurons, using the Parametric ReLU activation function and dropout regularization.
    - Output layer: 1 neuron, using the sigmoid activation function (corresponding to the output size).

Note: Additional hidden layers can be added or existing layers can be removed or modified by adjusting the 'hidden_sizes' parameter.
Note: Input layer e Output layer can be changed, but you have to check the dimensions of the input-target tensors.


Classes and Functionalities:

1. Class MLP()
The 'MLP' class defines the architecture of the Multi-Layer Perceptron, which is used for regression problems.
This class allows users to customize the network's architecture by specifying the input and output layer shapes,
as well as the desired sizes of hidden layers using the 'hidden_sizes' parameter. The activation function and 
dropout regularization are adjustable for each hidden layer, offering considerable flexibility in model design.
    
    Key Methods:
    a. initialize_weights(): This method intelligently initializes the network's parameters (weights and biases)
       with normalized random values, ensuring a proper starting point for training.
    b. forward(input_tensor): The 'forward' method performs the forward pass through the MLP, calculating the output
       of the network for a given input float tensor. This is the key function used during inference and prediction,
       but also have a key role during the training and validation part.

2. Class Net_training()
The 'Net_training' class is dedicated to handling the training, validation and inference of the neural network designed
for regression tasks. It takes an instance of the 'MLP' class as input and provides necessary functionality for model training.
   
   Key Methods:
    a. predict(net, input_tensor): This method makes predictions using the given neural network. It takes an input tensor
       and returns the network output. Users can use this function to obtain predictions for new data points after the model is trained.

    b. validate(net, dataloader_val, criterion): The 'validate' method evaluates the neural network's performance on a
       validation dataset. It calculates the average validation loss and provides a list of output tensors, which can be
       valuable for performance analysis. Additionally, it calculates the R2 score, offering insights into the model's 
       goodness of fit.

    c. training(net, X_train, X_val, y_train, y_val, ... ): The 'training' method trains the neural network for 
       multiple epochs. During training, it updates the network's parameters based on the provided training data 
       using an optimizer and loss function. Furthermore, it generates plots of the probability density functions 
       (PDFs) of the outputs/target, along with a plot of the losses during training.

Note:
To ensure efficient prediction and validation phases and avoid unnecessary gradient computation, it is recommended 
to use torch.no_grad() when executing these operations. Moreover, for effective configuration of the training process,
users can supply a YAML file containing relevant details such as loss function choices, optimizer options, and other 
training hyperparameters.
'''

#Import needed libraries, classes and functions
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
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
    Creates the architecture of the MLP used in a regression problem. It also defines a method for the correct 
    initialization of the network's parameters and the forward method.

    Args:
        input_size [int] : input layer shape
        output_size [int] : desired output layer shape
        hidden_sizes [list]: desired hidden layers shapes
    Returns:
        The class to define structure of the MLP and useful methods.
    """
    def __init__(self, input_size:int, output_size:int, hidden_sizes:list): #Architecture definition
        super(MLP, self).__init__()

        #Hidden layer list
        hidden_layers = []
        #Input size next layer, initialize it with the input_size
        prev_size = input_size

        #Iterative creation of hidden layers with the following structure and specification given to the class.
        for hidden_size in hidden_sizes:
            hidden_layers.append(nn.Linear(prev_size, hidden_size))
            hidden_layers.append(nn.PReLU())
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

    def predict(net, input_tensor): #Calculate the output
        '''
        Make predictions using the neural network.

        Args:
            net: the neural network of the class Net.
            X: PyTorch tensor containing the input data.

        Returns:
            Network output.
        '''
        #Retrieve the device where the parameters are
        device = next(net.parameters()).device  # we assume that all the network parameters are on the same device
        
        with torch.no_grad():
            net.eval()
            #Move the input data to the device
            X = input_tensor.to(device)
            #Compute the output of the network
            output = net(X)

        return output

    def validate(net, dataloader_val, criterion): #Validate the network
        '''
        Validate the neural network.

        Args:
            net: the neural network of the class Net.
            dataloader_val: dataloader object that provides batches of validation data.
            criterion: the loss function.

        Returns:
            Average validation loss.
        '''
        #Initialize device, avarage loss, outputs and targets lists
        device = next(net.parameters()).device
        val_loss = 0.0
        outputs = []
        targets = []

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

                #Calculate the losses of the minibatch
                loss = criterion(output, y_val_minibatch)

                #Accumulate the loss
                val_loss += loss.item()

                #Save outputs and targets
                outputs.append(output)
                targets.append(y_val_minibatch)

        #Calculate the average validation loss
        avg_val_loss = val_loss / len(dataloader_val)
        
        #Combine the tensors in the list output_val and targets into a single tensor along the samples dimension and convert tensors to numpy arrays for calculating the R2 score
        output_val_np = torch.cat(outputs, dim=0).cpu().numpy()
        targets_np = torch.cat(targets, dim=0).cpu().numpy()

        #Calculate the R2 score
        r2 = r2_score(targets_np, output_val_np)

        return avg_val_loss, outputs, r2
    
    def training(net, X_train, X_val, y_train, y_val, path_pth, path_txt, results_path, result_path_net, print_info, n_epochs, lr, minibatch_size, loss_function_choice, optimizer_choice):
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

        print("\n\nTraining the network...")
        sleep(2)
        print('\n\nLoading the dataset and creating the dataloader...\n\n\n')

        #Reshape the target in a correct way
        y_train = y_train.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        
        #Print
        if print_info == '1':
            #Some information about our data
            print("Validation and training dataset shape_")
            print("\t- X_train shape:", X_train.shape)
            print("\t- y_train shape:", y_train.shape)
            print("\t- X_val shape:", X_val.shape)
            print("\t- y_val shape:", y_val.shape)
            print("\n\n\n")
            sleep(6)
  
        ### DATASET ###
        #Combine input and output tensors
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

        ### DATALOADER ###
        dataloader_train = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=False, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False, drop_last=True)

        ### DEVICE ###
        device = next(net.parameters()).device

        #Print
        if print_info == '1':
            print("\n\nDevice, datasets and dataloader:\n")
            print(f'\t- The device selected for the training is: {device}')
            print(f'\t- Training dataset object = {train_dataset}')
            print(f'\t- Training dataloader object = {dataloader_train}')
            print(f'\t- Validation dataset object = {val_dataset}')
            print(f'\t- Validation dataloader object = {dataloader_val}')          
            sleep(8)

        ### LOSS FUNCTION SELECTION ###
        print('\n\nLoading the selected loss function...')

        #Function for multi-selection loss function
        if loss_function_choice == 'mse':  # Mean Square Error
            loss_function = nn.MSELoss()
        elif loss_function_choice == 'mae':  # Mean Absolute Error
            loss_function = nn.L1Loss()
        else:
            raise ValueError("Invalid choice of loss function")

        sleep(2)

        
        ### OPTIMIZER SELECTION ###
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

            sleep(8)

        
        #############################################################################

                                #STARTING TRAINING LOOP

        #############################################################################
        
        print("\n\n\tStarting the training loop...")

        #Telling the network we are going to train it (and not to simply evaluate it)
        net.train()

        #Define some useful quantities
        net_train_losses = [] #list of each loss of the epoch of training part
        net_val_losses = [] #list of the loss of the epochs of validation part

        
        ######### LOOP ON EPOCHS ##########
        for e in trange(n_epochs):  #loop on epochs
            #Initializes the training phase loss for the current epoch
            loss_value_train = 0.
            #Initialize the sum of the R2 scores during training epoch
            net_train_r2 = 0.
            #Outputs
            outputs_train = []
            ### LOOP ON MINI-BATCHES ###
            for nb, (X_minibatch, y_minibatch) in enumerate(dataloader_train):  #loop on mini-batches
                #Clearing the previously computed gradients (are saved in memory, each iteration we need to reset the gradients)
                optimizer.zero_grad()

                #Conversion double to float
                X_minibatch = X_minibatch.float()
                #Separate input and output
                X_minibatch = X_minibatch.to(device)
                y_minibatch = y_minibatch.to(device)
                
                #Calculate the outputs
                output = net(X_minibatch) #going forward, "net" is a callable object
               
                #Save the last epoch training outputs
                outputs_train.append(output)

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
                    #Accumulate the loss in the training phase loss
                    loss_value_train += loss_value_on_minibatch.item()
                    
                    ### R2SCORE MINIBATCH ###
                    #Put the tensor on CPU and transform it into a numpy array
                    y_train_np = y_minibatch.cpu().numpy()
                    output_train_np = output.cpu().numpy()
                    #R2score minibatch
                    r2_mb = r2_score(y_train_np, output_train_np)
                    #Save it
                    net_train_r2 += r2_mb

                    #Print of the minibatch
                    print("\tepoch:{}, minibatch: {}, loss_train: {:.4f}, r2score: {}%".format(e + 1, nb, loss_value_on_minibatch, r2_mb*100))
                    
                    
            #R2score last epoch train and validation
            net_train_r2 /= len(dataloader_train)

            #loss della training epoch
            loss_value_train /= len(dataloader_train)

            #Save the net losses of each batch within the lists defined earlier
            net_train_losses.append(loss_value_train)
            
            ### VALIDATION ###
            # Validation of the epoch
            loss_valid, output_val, net_val_r2 = Net_training.validate(net, dataloader_val, loss_function)

            #Append all losses
            net_val_losses.append(loss_valid)

        #Combine the tensors in the list output_val into a single tensor along the samples dimension
        output_val_tensor = torch.cat(output_val, dim=0).reshape(-1,1)
        outputs_train_tensor = torch.cat(outputs_train, dim=0).reshape(-1,1)


        ### MEAN AND STD OF THE OUTPUT ###
        #Calculate the VALIDATION mean and std of the output data
        mean_value = torch.mean(output_val_tensor)
        std_value = torch.std(output_val_tensor)
        
        ### R2 SCORE PRINT ###
        print(f'\n\n\n R2 SCORE OF THE TRAINING PHASE LAST EPOCH: {net_train_r2 * 100}%')
        
        ### SAVING TRAINED NET ###
        save_net(net,n_epochs, loss_value_train, loss_valid, path_pth, path_txt, mean_value=mean_value, std_value=std_value, r2 = net_val_r2)

        ### PLOT OF TRAINING AND VALIDATION LOSSES ###
        plot_loss(n_epochs, net_train_losses, net_val_losses, result_path_net)
        
        ### PREPARE THE DATA FOR PLOT BOX ###
        #Convert Pytorch tensor into list on CPU
        data_train = y_train.cpu().tolist()
        data_train_output = outputs_train_tensor.cpu().tolist()
        data_val = y_val.cpu().tolist()
        data_val_output = output_val_tensor.cpu().tolist()

        #Create the DataFrames
        df_train = pd.DataFrame(data_train, columns=['Target'])
        df_train_output = pd.DataFrame(data_train_output, columns=['Output'])
        df_val = pd.DataFrame(data_val, columns=['Target'])
        df_val_output = pd.DataFrame(data_val_output, columns=['Output'])
        
        #Take the values from the dataframes
        target_t = df_train.loc[:, 'Target'].values
        output_t = df_train_output.loc[:, 'Output'].values
        target_v = df_val.loc[:, 'Target'].values
        output_v = df_val_output.loc[:, 'Output'].values
        
        ### PLOT BOX ###
        ### BOX PLOT TO COMPARE TARGET AND OUTPUT TRAINING SET ###
        plot_box([target_t, output_t],['Target', 'Output'],'Boxplot target and output training set pdf','y data','Pdf values',f'{results_path}/NET/boxplot_training_pdy.png') 
        ### BOX PLOT TO COMPARE TARGET AND OUTPUT VALIDATION SET ###
        plot_box([target_v, output_v],['Target', 'Output'],'Boxplot target and output validation set pdf','y data','Pdf values',f'{results_path}/NET/boxplot_validation_pdy.png') 
        
        ### PLOT PDY ###
        ### PLOT PDY TRAINING SET ###
        plot_pdf(y_train, outputs_train_tensor, 'Target training set', 'Output training set', 'Comparison target and output training set', f'{results_path}/NET/training_pdy.png')
        ### PLOT PDY VALIDATION SET ###
        plot_pdf(y_val, output_val_tensor, 'Target validation set', 'Output validation set', 'Comparison target and output validation set', f'{results_path}/NET/validation_pdy.png')

        
        return net_train_losses,net_val_losses


        

    