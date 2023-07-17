import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataset_function import *
from tqdm import trange
from time import sleep
from torchsummary import summary
from sklearn.metrics import r2_score

def createDirectory(nameDirectory: str):
    """
    This function is used to create folders, taking as input the absolute path of the folder we want to create (path/foldername). In our case, the path is passed
    through the yaml file. The folders that will be created are those for saving the generator/discriminator training, and during the evaluation phase, the folder
    where the images produced by the generator will be saved.
    """
    if not os.path.exists(f'{nameDirectory}'):  #checks if the folder with that path already exists
        os.mkdir(f'{nameDirectory}')  #if it does not exist creates it

class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(Net, self).__init__()

        hidden_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            hidden_layers.append(nn.Linear(prev_size, hidden_size))
            hidden_layers.append(nn.Tanh())
            prev_size = hidden_size

        self.layers = nn.Sequential(
            *hidden_layers,
            nn.Linear(prev_size, output_size),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def forward(self, inputs):
        return self.layers(inputs)

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
                nn.init.constant_(layer.bias.data, 0.0)




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
        #Recupero il device dove sono i parametri
        device = next(net.parameters()).device  # we assume that all the network parameters are on the same device
        #lista output
        outputs = []

        with torch.no_grad():
            net.eval()

            for X_minibatch, y_minibatch in dataloader:
                # Move the mini-batch to the device
                X_minibatch = X_minibatch.to(device)
                y_minibatch = y_minibatch.to(device)
                # Compute the output of the mini-batch
                output = net(X_minibatch)
                outputs.append(output)

        return torch.cat(outputs, dim=0)


    def validate(net, dataloader_val, criterion):
        '''
        Validate the neural network.

        Args:
            net: the neural network of the class Net.
            dataloader_val: dataloader object that provides batches of validation data.
            criterion: the loss function.

        Returns:
            Average validation loss.
        '''
        device = next(net.parameters()).device
        val_loss = 0.0

        with torch.no_grad():
            net.eval()

            for X_val_minibatch, y_val_minibatch in dataloader_val:
                #Caricamento in mini-batch di input e output
                X_val_minibatch = X_val_minibatch.to(device)
                y_val_minibatch = y_val_minibatch.to(device)

                #Predictions
                output = net(X_val_minibatch)
                #Calcolo le loss del minibatch
                loss = criterion(output, y_val_minibatch)

                #Accumulate the loss
                val_loss += loss.item()

        # Calculate the average validation loss
        avg_val_loss = val_loss / len(dataloader_val)

        return avg_val_loss
    
    def prova(net, X_train, X_val, y_train, y_val, epochs=2000, lr=0.001, minibatch_size=32, loss_function_choice="rmse", optimizer_choice="adam"):
        """
        Train a neural network for multiple epochs.

        Args:
            net: the neural network.
            X_train: training data (one-example-per-row).
            y_train: training targets.
            X_val: validation data (one-example-per-row).
            y_val: validation targets.
            epochs (optional): number of epochs.
            lr (optional): learning rate.
            minibatch_size (optional): size of the training mini-batches.
            loss_function_choice (optional): choice of loss function.
            optimizer_choice (optional): choice of optimizer.

        Returns:
            The two arrays with the R2s on training and validation data computed during the training epochs.
        """

        print("Training the network...")
        sleep(2)
        print('\n\n\tLoading the dataset and creating the dataloader...\n')

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_val shape:", X_val.shape)
        print("y_val shape:", y_val.shape)


        #Combine input and output tensors
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

        # Dataloader
        dataloader_train = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=False, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False, drop_last=True)

        # We assume that all the network parameters are on the same device
        device = next(net.parameters()).device

        # Print to verify that everything has been executed correctly, that the objects have been instantiated and the device is defined:
        print(f'\t\tThe device selected for the training is: {device}')
        print(f'\t\tTraining dataset object = {train_dataset}')
        print(f'\t\tTaining dataloader object = {dataloader_train}')
        print(f'\t\tNumber of datasets training samples = {len(train_dataset)}')
        sleep(4)

        
        # LOSS FUNCTION SELECTION
        print('\n\n\tLoading the selected loss function...')

        # Function for multi-selection loss function
        if loss_function_choice == 'mse':  # Mean Square Error
            loss_function = nn.MSELoss()
        elif loss_function_choice == 'mae':  # Mean Absolute Error
            loss_function = nn.L1Loss()
        else:
            raise ValueError("Invalid choice of loss function")

        sleep(2)

        
        # OPTIMIZER SELECTION AND APPLICATION
        print('\n\n\tLoading the selected optimizer...')

        # Setup Adam or SGD optimizers for both the generator and the discriminator
        if optimizer_choice == "adam":
            optimizer = optim.Adam(net.parameters(), lr=lr)
        elif optimizer_choice == "sgd":
            optimizer = optim.SGD(net.parameters(), lr=lr)
        elif optimizer_choice == "rmsprop":
            optimizer = optim.RMSprop(net.parameters(), lr=lr)
        else:
            raise ValueError("Invalid choice of optimizer")

        sleep(2)

        #PRINT OF THE TRAINING FEATURES:
        print('\n\n\tSome hyperparameters of the network:\n')
        print(f'\t\t- Learning rate: {lr}')
        print(f'\t\t- Optimizer: {optimizer_choice}')
        print(f'\t\t- Epochs: {epochs}')
        print(f'\t\t- Batch size: {minibatch_size}')   
        print(f'\t\t- Loss function selected: {loss_function_choice}')

        sleep(4)
        
        #Print of the architecture:
        print(f'\t\t- Architecture:')    
        summary(net,(minibatch_size,1))

        
        sleep(10)

        
        #############################################################################

                                #STARTING TRAINING LOOP

        #############################################################################
        
        print("\n\n\tStarting the training loop...")

        #Telling the network we are going to train it (and not to simply evaluate it)
        net.train()

        #Create the directory for the result
        createDirectory('D:/Results')

        # Definisco alcune quantità utili
        net_train_losses = [] #list of each loss of the epoch
        net_val_losses = [] #list of the loss of the epochs

        ######### LOOP ON EPOCHS ##########
        for e in trange(epochs):  # loop on epochs
            # Azzero tutte le mie quantità
            loss_value_train= 0.
            nb = 0  # rappresenta l'indice del mini-batch corrente nel ciclo dei mini-batch
            loss_value_train = 0.0 # Inizializza la loss della fase di training per l'epoca corrente

            for nb, (X_minibatch, y_minibatch) in enumerate(dataloader_train):  #loop on mini-batches
                # Clearing the previously computed gradients (are saved in memory, each iteration we need to reset the gradients)
                optimizer.zero_grad()

                #Separate input and output
                X_minibatch = X_minibatch.to(device)
                y_minibatch = y_minibatch.to(device)

                #Calculate the outputs
                output = net(X_minibatch) #going forward, "net" is a callable object
                
                #Calculate the loss on batches and save it:
                loss_value_on_minibatch = loss_function(output, y_minibatch)
            
                #Calculate gradients for all mini-batch on the net with the backward step. The backward() operation is used 
                #for calculating the gradient of the error with respect to its parameters:
                loss_value_on_minibatch.backward()
                
                #PARAMETERS UPDATE
                #This function is used to update the parameters based on the calculated gradients. This process of
                #updating parameters is called "optimization" and is performed using an optimization algorithm such as Adam,SGD or other:
                optimizer.step()

                #Print of the loss and the r2score, and updating the globale loss value of a single epoch
                with torch.no_grad():
                    #Print of the minibatch
                    print("\tepoch:{}, minibatch: {}, loss_train: {:.4f}".format(e + 1, nb, loss_value_on_minibatch))
                    
                    # Accumula la loss nella loss della fase di training
                    loss_value_train += loss_value_on_minibatch.item()

            
            #### Saving of the best net ####
            #Save the "model state" of the net to a .pth file in the specified path
            torch.save(net.state_dict(), f"D:/Results/net_ep_{e}.pth")
            #Write a .txt file to the specified path and writes information regarding the batch number and the epoch to which
            #the best trained net belongs
            with open(f"D:/Results/net_ep_{e}.txt", "w") as f:
                print(f"Checkpoint net\n\n\tBATCH_ID:\t{nb + 1}\n\EPOCH:\t{e + 1}", file=f)

            #loss della training epoch
            loss_value_train /= len(dataloader_train)

            #Save the net losses of each batch within the lists defined earlier
            net_train_losses.append(loss_value_train)

            #Validation of the epoch
            loss_valid = Net_training.validate(net, dataloader_val, loss_function)

            #Appendo tutte le loss:
            net_val_losses.append(loss_valid)
        #Alla fine del ciclo di addestramento, crea il plot delle perdite di training e validation per ogni epoch
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), net_train_losses, label='Training Loss')
        plt.plot(range(1, epochs + 1), net_val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        return net_train_losses,net_val_losses


    def plot(r2s_train, r2s_val, output_test, test_y):
        """Plot the R2 score computed on the training and validation data during the training stage.

        Args:
            r2s_train: the array with the training R2 scores (during the training epochs).
            r2s_val: the arrays with the validation R2 scores (during the training epochs).
            output_test: the prediction on the test data.
            test_y: the target of the test data
        """

        plt.figure()
        plt.plot(r2s_train, label='Training Data')
        plt.plot(r2s_val, label='Validation Data')
        plt.ylabel('R2 Score')
        plt.xlabel('Epochs')
        plt.ylim((-1.1, 1.1))
        plt.legend(loc='lower right')
        plt.savefig('training_stage.pdf')

        plt.figure()
        plt.plot(output_test, test_y, '*')
        plt.ylabel('Target Price')
        plt.xlabel('Predicted Price')
        plt.savefig('test_stage.pdf')



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

    #Carichiamo i dati dal nuovo file .csv e dividiamo in input e target
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
    
    #Ensure that we keep track of the mean and std used to normalize the data
    torch.save([m, s, max, min], 'C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/normalizers_hyperparam.pth')

    #Normalizziamo pure gli output
    data_y_train = normalize_output_data(data_y_train)
    data_y_val = normalize_output_data(data_y_val)
    data_y_test = normalize_output_data(data_y_test)

    ######## INIZIALIZZO LA RETE ###########
    #Neuroni dei miei layer
    input_size = 1
    output_size = 1
    hidden_layers = [128, 256]

    #Creo l'architettura:
    net = Net(input_size, output_size,hidden_layers)
    
    #Inizializzo i pesi
    net.initialize_weights()

    # Sposto il modello sulla GPU
    net.to(my_device)

    # Sposto i dati di addestramento sulla GPU
    data_X_train = data_X_train.to(my_device)
    data_y_train = data_y_train.to(my_device)
    data_X_val = data_X_val.to(my_device)
    data_y_val = data_y_val.to(my_device)

    #### TRAINING PHASE ####
    # training the network
    bah = Net_training.prova(net, data_X_train, data_X_val, data_y_train, data_y_val, epochs=50, lr=0.01, minibatch_size=32, loss_function_choice="mse", optimizer_choice="adam")
    
    

      
   