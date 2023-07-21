import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from matplotlib import pyplot as plt
from dataset_function import *
from tqdm import trange
from main import createDirectory
from time import sleep
from torchsummary import summary


torch.manual_seed(0)

class Autoencoder(nn.Module):
    """Makes the main denoising autoencoder for regression

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape = 5, enc_shape = 1):
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

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)        
        # Decoding
        reconstructed = self.decoder(encoded)
        
        return reconstructed
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
    
    def validate(model, dataloader_val, criterion):
        '''
        Validate the neural network.

        Args:
            net: the neural network of the class Net.
            dataloader_val: dataloader object that provides batches of validation data.
            criterion: the loss function.

        Returns:
            Average validation loss.
        '''
        device = next(model.parameters()).device
        
        #Azzero la loss del epoch
        val_loss = 0.

        with torch.no_grad():
            #Let the machine in evaluation mode
            model.eval()

            #Starting the loop ogn the validation set
            for X_val_minibatch in dataloader_val:
                
                #Caricamento in mini-batch di input
                X_val_minibatch = X_val_minibatch.to(device)
                
                #Predictions
                output = model(X_val_minibatch)

                #Calcolo le loss del minibatch
                loss = criterion(output, X_val_minibatch)

                #Accumulate the loss
                val_loss += loss.item()

        # Calculate the average validation loss
        avg_val_loss = val_loss / len(dataloader_val)

        return avg_val_loss
      
    def training(model, input_train_tensor, input_val_tensor, loss_function_choice = "mse", optimizer_choice = "adam", n_epochs=300, lr =0.005, batch_size=128):
        
        print("Validation and training dataset shape")
        print("Train set shape:", input_train_tensor.shape)
        print("Validation set shape:", input_val_tensor.shape)
        
        print("\n\n\n")


        print("Training the network...")
        sleep(2)
        print('\n\n\tLoading the dataset and creating the dataloader...\n\n\n')
        sleep(2)

        
        # Convert input_tensor to a PyTorch TensorDataset
        dataset_train = torch.Tensor(input_train_tensor)
        dataset_val = torch.Tensor(input_val_tensor)

        # Create a DataLoader to handle minibatching
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # We assume that all the network parameters are on the same device
        device = next(model.parameters()).device
        # Print to verify that everything has been executed correctly, that the objects have been instantiated and the device is defined
        print(f'\t\tThe device selected for the training is: {device}')
        print(f'\t\tTraining dataset object = {dataset_train}')
        print(f'\t\tTraining dataloader object = {dataloader_train}')
        print(f'\t\tValidation dataset object = {dataset_val}')
        print(f'\t\tValidation dataloader object = {dataloader_val}')
        print(f'\t\tNumber of datasets training and validation samples = {len(dataset_train),len(dataset_val)}')
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
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_choice == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_choice == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError("Invalid choice of optimizer")

        sleep(2)

        #PRINT OF THE TRAINING FEATURES:
        print('\n\n\tSome hyperparameters of the network:\n')
        print(f'\t\t- Learning rate: {lr}')
        print(f'\t\t- Optimizer: {optimizer_choice}')
        print(f'\t\t- Epochs: {n_epochs}')
        print(f'\t\t- Batch size: {batch_size}')   
        print(f'\t\t- Loss function selected: {loss_function_choice}')

        sleep(4)

        #############################################################################

                                #STARTING TRAINING LOOP

        #############################################################################
        
        print("\n\n\tStarting the training loop...")

        #Telling the network we are going to train it (and not to simply evaluate it)
        model.train()

        #Create the directory for the result
        createDirectory('D:/Results')
        createDirectory('D:/Results/Autoencoder')

        # Definisco alcune quantità utili
        net_train_losses = [] #list of each loss of the epoch
        net_val_losses = [] #list of the loss of the epochs


        ######### LOOP ON EPOCHS ##########
        for epoch in trange(n_epochs):
            #Inizializza la loss della fase di training per l'epoca corrente
            epoch_loss = 0.0

            for idx_batch, X_minibatch in enumerate(dataloader_train):
                # Clearing the previously computed gradients (are saved in memory, each iteration we need to reset the gradients)
                optimizer.zero_grad()

                #Input and target data on the device
                X_minibatch = X_minibatch.to(device)
                
                # Forward pass
                output = model(X_minibatch)
             
                # Calculate the loss for this batch
                loss = loss_function(output, X_minibatch)
                
                # Backward pass
                loss.backward()
                
                # Update model parameters
                optimizer.step()

                #Print of the loss and the r2score, and updating the globale loss value of a single epoch
                with torch.no_grad():
                    #Print of the minibatch
                    print("\tepoch:{}, minibatch: {}, loss_train: {:.4f}".format(epoch+1, idx_batch, loss))
                    
                    # Accumulate the loss for the epoch
                    epoch_loss += loss.item()
            #loss della training epoch
            epoch_loss /= len(dataloader_train)

            #Save the net losses of each batch within the lists defined earlier
            net_train_losses.append(epoch_loss)


            #Validation of the epoch
            loss_valid = Autoencoder.validate(model, dataloader_val, loss_function)

            #Appendo tutte le loss:
            net_val_losses.append(loss_valid)
        
        #### Saving of the best net ####
        #Save the "model state" of the net to a .pth file in the specified path
        torch.save(model.state_dict(), f"D:/Results/Autoencoder/net_ep_{epoch+1}.pth")
        #Write a .txt file to the specified path and writes information regarding the batch number and the epoch to which
        #the best trained net belongs
        with open(f"D:/Results/Autoencoder/net_ep_{epoch+1}.txt", "w") as f:
            print(f"Checkpoint net:\n\n\tEPOCH:\t{epoch + 1}\n\n\tLOSS:\t{epoch_loss}", file=f)

        #Alla fine del ciclo di addestramento, crea il plot delle perdite di training e validation per ogni epoch
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_epochs + 1), net_train_losses, label='Training Loss')
        plt.plot(range(1, n_epochs + 1), net_val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        return net_train_losses,net_val_losses


if __name__ == "__main__":

    #Caricamento dataframe da file .csv
    filename = "C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/AirQualityUCI.csv"
    dataframe = pd.read_csv(filename, sep=";")

    #Ridimensionamento dataset
    dataset_reduced = dataset_reduction(dataframe,"NOx(GT)","PT08.S1(CO)","T","RH","PT08.S2(NMHC)","CO(GT)")
    print('la dimensionalità è: ',dataset_reduced.shape)
    
    #Pulitura dataset ridotto
    dataset_reduced = cleaning_dataset_function(dataset_reduced).iloc[:, :5]
    print('la dimensionalità ridotta è: ',dataset_reduced.shape)
    print('le colonnne sono: ',dataset_reduced.columns)
    sleep(6)

    #Salvo il dataset ridotto
    create_file_csv(dataset_reduced,"C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/dataset_reduced_Autoencoder.csv")

    #Device configuration
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Carico il dataset
    data = load_csv_file("C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/dataset_reduced_Autoencoder.csv")

    #Splitting data in training, validation e test sets
    data_X_train, data_X_val, _ = create_encoder_splits_unbalanced(data, 0.85, 0.15)
    
    #Normalizing data input
    data_X_train, _, _ = normalize_input_data(data_X_train)
    data_X_val, _, _ = normalize_input_data(data_X_val)
    
    
    ######## INIZIALIZZO LA RETE ###########
    #Creo l'architettura:
    encoder = Autoencoder().double().to(my_device)

    #Inizializzo i pesi
    encoder.initialize_weights()

    #Sposto il modello sulla GPU
    encoder.to(my_device)

    # Sposto i dati di addestramento sulla GPU
    data_X_train = data_X_train.to(my_device)
    data_X_val = data_X_val.to(my_device)

    #### TRAINING PHASE ####
    # training the network
    Autoencoder.training(encoder, data_X_train, data_X_val)
    