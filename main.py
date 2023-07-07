import pandas as pd
import numpy as np
import os
import matplotlib as plt
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
    
    def r2_score(outputs, targets, update_from=None):
        """
        Compute the R2 score of the provided network outputs with respect to given targets.

        Args:
            outputs: the outputs of the network on a batch of data.
            targets: the ground truth targets of the data.
            update_from (optional): a 4-element list/tuple, composed of those quantities that are needed to model
                an initial status of the R2 computation; the computation of the R2 will start from this status.

        Returns:
            R2 score computed on the given data, computed also considering the initial status, and the status (4-element
                tuple) needed for future updates of this score.
        Usage:
            The R2 score is a common measure of reliability or goodness of fit of a model that we are training.
            It indicates how well the model is able to explain the variation in the data compared to the expected 
            output values (targets).
        """
        #Calcolo le seguenti quantità utili per ottenere R2
        sum_errors = torch.sum(torch.pow(outputs - targets, 2)).item()  #.item() utilizzato per ottenere lo scalare e non vettore
        sum_squared_targets = torch.sum(torch.pow(targets, 2)).item()
        sum_targets = torch.sum(targets).item()
        n_sample = outputs.shape[0]

        #Calcolo di r2
        r2 = 1. - sum_errors / (sum_squared_targets - (sum_targets ** 2) / n_sample)

        
        """
        Se il parametro update_from viene fornito, significa che la funzione viene chiamata per aggiornare 
        un punteggio R2 esistente con nuovi dati(caso di online mode - mini-batch). Se il parametro update_from non viene 
        fornito, significa che la funzione viene chiamata per calcolare il punteggio R2 da zero su un singolo 
        gruppo di dati(batch mode).
        """        
        if update_from is not None:
            #Aggiornamento delle quantità precedenti
            sum_errors += update_from[0]
            sum_squared_targets += update_from[1]
            sum_targets += update_from[2]
            n_sample += update_from[3]
            #Aggiornamento score R2
            r2_updated = 1. - sum_errors / (sum_squared_targets - (sum_targets ** 2) / n_sample)
        else:
            #Calcolo da zero dello score r2
            r2_updated = r2

        #Aggiornamento della 4-tuple con valori aggiornati
        status = (sum_errors, sum_squared_targets, sum_targets, n_sample)

        #restituisce r2 di questo singolo e cumulato, ed inoltre restituisce la 4-tuple aggiornata
        return r2, r2_updated, status
    
    def r2_score_prova(outputs, targets, update_from=None):
        """
        Compute the R2 score of the provided network outputs with respect to given targets.

        Args:
            outputs: the outputs of the network on a batch of data.
            targets: the ground truth targets of the data.
            update_from (optional): a 4-element list/tuple, composed of those quantities that are needed to model
                an initial status of the R2 computation; the computation of the R2 will start from this status.

        Returns:
            R2 score computed on the given data, computed also considering the initial status, and the status (4-element
                tuple) needed for future updates of this score.
        Usage:
            The R2 score is a common measure of reliability or goodness of fit of a model that we are training.
            It indicates how well the model is able to explain the variation in the data compared to the expected 
            output values (targets).
        """
        #Calcolo le seguenti quantità utili per ottenere R2
        #numeratore
        sum_errors = torch.sum(torch.pow(outputs - targets, 2)).item()  #.item() utilizzato per ottenere lo scalare e non vettore
        
        #denominatore
        mean_output = torch.mean(outputs, dim=0)
        sum_errors_den = torch.sum(torch.pow(outputs - mean_output, 2)).item()

        #Calcolo di r2
        r2 = 1. - sum_errors / sum_errors_den

        
        """
        Se il parametro update_from viene fornito, significa che la funzione viene chiamata per aggiornare 
        un punteggio R2 esistente con nuovi dati(caso di online mode - mini-batch). Se il parametro update_from non viene 
        fornito, significa che la funzione viene chiamata per calcolare il punteggio R2 da zero su un singolo 
        gruppo di dati(batch mode).
        """        
        if update_from is not None:
            #Aggiornamento delle quantità precedenti
            sum_errors += update_from[0]
            sum_errors_den += update_from[1]
            #Aggiornamento score R2
            r2_updated = 1. - sum_errors / sum_errors_den
        else:
            #Calcolo da zero dello score r2
            r2_updated = r2

        #Aggiornamento della 4-tuple con valori aggiornati
        status = (sum_errors,sum_errors_den)

        #restituisce r2 di questo singolo e cumulato, ed inoltre restituisce la 4-tuple aggiornata
        return r2, r2_updated, status


    
    def predict_and_r2_score(net, X, y, minibatch_size=None):
        '''
        Make prediction and compute the R2 score (it supports mini-batches).

        Args:
            net: the neural network of the class Net.
            X: dataset on which predictions are performed.
            y: ground truth targets.
            minibatch_size (optional): size of the mini-batches.

        Returns:
            R2 score and the network output.
        '''

        device = next(net.parameters()).device  # we assume that all the network parameters are on the same device
        n = X.shape[0]
        r2_status = None
        t = 0
        outputs = []

        if minibatch_size is None:
            minibatch_size = n

        with torch.no_grad():
            training_mode = net.training
            net.eval()

            while True:  # loop on mini-batches
                f = t
                t = min(f + minibatch_size, n)
                X_minibatch = X[f:t, :].to(device)
                y_minibatch = y[f:t].to(device)

                output = net(X_minibatch)
                outputs.append(output)

                #_, r2, r2_status = Net_training.r2_score(output, y_minibatch, update_from=r2_status)
                r2 = r2_score(output, y_minibatch)

                if t == n:
                    break

            if training_mode:
                net.train()

            return r2, torch.cat(outputs, dim=0)
        
    def predict_and_r2_score_prova(net, dataloader):
        '''
        Make prediction and compute the R2 score (it supports mini-batches).

        Args:
            net: the neural network of the class Net.
            dataloader: dataloader object that provides batches of data.

        Returns:
            R2 score and the network output.
        '''

        device = next(net.parameters()).device  # we assume that all the network parameters are on the same device
        outputs = []
        r2_status = None

        with torch.no_grad():
            training_mode = net.training
            net.eval()

            for X_minibatch, y_minibatch in dataloader:
                X_minibatch = X_minibatch.to(device)
                y_minibatch = y_minibatch.to(device)

                output = net(X_minibatch)
                outputs.append(output)

                # _, r2, r2_status = Net_training.r2_score(output, y_minibatch, update_from=r2_status)
                r2 = r2_score(output, y_minibatch)

        if training_mode:
            net.train()

        return r2, torch.cat(outputs, dim=0)
    
    def prova(net, X_train, y_train, X_val, y_val, epochs=2000, lr=0.001, minibatch_size=32, loss_function_choice="rmse", optimizer_choice="adam"):
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

        # Combine input and output tensors
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
        print(f'\t\tValidation dataset object = {val_dataset}')
        print(f'\t\tTaining dataloader object = {dataloader_train}')
        print(f'\t\tValidation dataloader object = {dataloader_val}')
        print(f'\t\tNumber of datasets training and validation samples = {len(train_dataset), len(val_dataset)}')
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
        #print(f'\t\t- Model: {args.selectionModel}')
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

        print("\n\n\tStarting the training loop...")

        #Telling the network we are going to train it (and not to simply evaluate it)
        net.train()

        #Create the directory for the result
        createDirectory('D:/Results')

        # Definisco alcune quantità utili
        n_sample = X_train.shape[0]  # numero training sample
        best_r2_val = None
        r2s_train = np.zeros(epochs)  # Vettore di zeri dove salveremo r2s per ogni training epoch
        r2s_val = np.zeros(epochs)  # Vettore di zeri dove salveremo r2s per ogni validation epoch
        net_losses = [] #list of each loss

        ######### LOOP ON EPOCHS ##########
        for e in trange(epochs):  # loop on epochs
            # Azzero tutte le mie quantità
            loss_value = 0.
            r2_status = None
            nb = 0  # rappresenta l'indice del mini-batch corrente nel ciclo dei mini-batch

            for X_minibatch, y_minibatch in dataloader_train:  #loop on mini-batches
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
                
                #### Saving of the best net ####

                if len(net_losses) == 0: #check
                    #Save the "model state" of the net to a .pth file in the specified path
                    torch.save(net.state_dict(), f"D:/Results/net_Best.pth")
                    #Write a .txt file to the specified path and writes information regarding the batch number and the epoch to which
                    #the best trained net belongs
                    with open(f"D:/Results/net_Best.txt", "w") as f:
                        print(f"Checkpoint BEST\n\n\tBATCH_ID:\t{nb + 1}\n\EPOCH:\t{e + 1}", file=f)
                elif loss_value_on_minibatch.item() < np.min(net_losses):
                     #check if the loss in this batch is the smallest compared to previous batches
                    #Save the "model state" of the generator to a .pth file in the specified path:
                    torch.save(net.state_dict(), f"D:/Results/net_Best.pth")
                    #Writes a .txt file to the specified path and writes information regarding the batch number and the era to which 
                    #the best trained generator belongs:
                    with open(f"D:/Results/net_Best.txt", "w") as f:
                        print(f"Checkpoint BEST\n\tBATCH_ID:\t{nb + 1}\n\tEPOCH:\t{e + 1}", file=f)

                #Save the generator and discriminator losses of each batch within the lists defined earlier:
                net_losses.append(loss_value_on_minibatch.item())
                


                #Print of the loss and the r2score, and updating the globale loss value of a single epoch
                with torch.no_grad():
                    #r2_train_on_minibatch, r2_train, r2_status = Net_training.r2_score(output, y_minibatch, update_from=r2_status)
                    #r2_train_on_minibatch, r2_train, r2_status = Net_training.r2_score_prova(output, y_minibatch)


                    print("\tMinibatch: {}, loss_train: {:.4f}".format(nb + 1, loss_value_on_minibatch))

                    loss_value += (loss_value_on_minibatch.item() ** 2) * X_minibatch.size(0)  # needed to estimate the train loss

                #This line of code save the "model state" of the net for each mini-batch.
                #So the .state_dict() method returns a "model state" that is like a dictionary of parameters that can be loaded
                #into a model of identical architecture. In practice, the model state includes network parameters, e.g. the 
                #weights of connections, information about the activation functions used, etc.   
                torch.save(net.state_dict(), f"D:/Results/net_Epoch{e + 1}_Batch{nb + 1}.pth")
                
                #Counter of the mini-batch index
                nb += 1
                if nb == len(dataloader_train):
                    break
            
            #EVALUATION ON VALIDATION SET(CONTROLLA)
            for X_val_minibatch, y_val_minibatch in dataloader_val:

                X_val_minibatch = X_val_minibatch.to(device)
                y_val_minibatch = y_val_minibatch.to(device)

                r2_val, _ = Net_training.predict_and_r2_score(net, X_val_minibatch, y_val_minibatch, minibatch_size=minibatch_size)
                found_best = False
                if best_r2_val is None or r2_val > best_r2_val:
                    best_r2_val = r2_val
                    found_best = True
                    torch.save(net.state_dict(), 'net_best.pth')

            loss_value = np.sqrt(loss_value / n_sample)
            print("epoch: {}, loss_train: {:.4f}, r2_train: {:.2f}, r2_val: {:.2f}".format(e + 1, loss_value, r2_train, r2_val) + (" (best)" if found_best else ""))

            r2s_train[e] = r2_train
            r2s_val[e] = r2_val

        return r2s_train

       
    
    def train(net, X_train, y_train, X_val, y_val, epochs=2000, lr=0.001, minibatch_size=32, loss_function_choice="rmse", optimizer_choice="adam"):
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

        Returns:
            The two arrays with the R2s on training and validation data computed during the training epochs.
        """
        # Function for multi-selection loss function
        def loss_function(loss_function_choice, o, y):
            if loss_function_choice == 'mse':  # Mean Square Error
                return nn.MSELoss()(o, y)
            elif loss_function_choice == 'rmse':  # Root Mean Squared Error
                return torch.sqrt(nn.MSELoss()(o, y))
            elif loss_function_choice == 'mae':  # Mean Absolute Error
                return nn.L1Loss()(o, y)
            elif loss_function_choice == 'msle':
                return nn.MSELoss()(torch.log(o + 1), torch.log(y + 1))
            else:
                raise ValueError("Invalid choice of loss function")

        # Function for multi-selection optimizer
        def select_optimizer(optimizer_choice, parameters, lr):
            if optimizer_choice == 'adam':
                return optim.Adam(parameters, lr=lr)
            elif optimizer_choice == 'sgd':
                return optim.SGD(parameters, lr=lr)
            elif optimizer_choice == 'rmsprop':
                return optim.RMSprop(parameters, lr=lr)
            else:
                raise ValueError("Invalid choice of optimizer")

        print("Training the network...")
        sleep(2)
        print('\n\n\tLoading the dataset and creating the dataloader...\n')

        # Unisco i tensori del input e output
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

        # Dataloader
        dataloader_train = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=False, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False, drop_last=True)

        # We assume that all the network parameters are on the same device
        device = next(net.parameters()).device

        #Print to verify that everything has been executed correctly, that the objects have been instantiated and the device is defined:
        print(f'\t\tThe device selected for the training is: {device}')
        print(f'\t\tDatasets objects = {train_dataset, val_dataset}')
        print(f'\t\tDataloaders objects = {dataloader_train, dataloader_val}')
        print(f'\t\tNumber of datasets training and validation samples = {len(train_dataset, val_dataset)}')
        sleep(4)

        # Telling the network we are going to train it (and not to simply evaluate it)
        net.train()

        # Defining the way we are going to update the net parameters
        optimizer = select_optimizer(optimizer_choice, net.parameters(), lr)

        

        # Definisco alcune quantità utili
        n_sample = X_train.shape[0]  # numero training sample
        best_r2_val = None
        r2s_train = np.zeros(epochs)  # Vettore di zeri dove salveremo r2s per ogni training epoch
        r2s_val = np.zeros(epochs)  # Vettore di zeri dove salveremo r2s per ogni validation epoch

        for e in trange(epochs):  # loop on epochs
            # Azzero tutte le mie quantità
            loss_value = 0.
            r2_status = None
            nb = 0  # rappresenta l'indice del mini-batch corrente nel ciclo dei mini-batch

            for X_minibatch, y_minibatch in dataloader_train:  #loop on mini-batches
                # Clearing the previously computed gradients (are saved in memory, each iteration we need to reset the gradients)
                optimizer.zero_grad()

                #Separate input and output
                X_minibatch = X_minibatch.to(device)
                y_minibatch = y_minibatch.to(device)

                outputs = net.forward(X_minibatch)  # going forward, "net" is a callable object
                loss_value_on_minibatch = loss_function(loss_function_choice, outputs, y_minibatch)  # RMSE

                with torch.no_grad():
                    r2_train_on_minibatch, r2_train, r2_status = Net_training.r2_score(outputs, y_minibatch, update_from=r2_status)

                    print("\tminibatch: {}, loss_train: {:.4f}, r2_train: {:.2f}".format(nb + 1, loss_value_on_minibatch, r2_train_on_minibatch))

                    loss_value += (loss_value_on_minibatch.item() ** 2) * X_minibatch.size(0)  # needed to estimate the train loss

                loss_value_on_minibatch.backward()  # going backward
                optimizer.step()  # updating model parameters

                nb += 1
                if nb == len(dataloader_train):
                    break

            torch.save(net.state_dict(), 'net.pth')

            for X_val_minibatch, y_val_minibatch in dataloader_val:
                X_val_minibatch = X_val_minibatch.to(device)
                y_val_minibatch = y_val_minibatch.to(device)

                r2_val, _ = Net_training.predict_and_r2_score(net, X_val_minibatch, y_val_minibatch, minibatch_size=minibatch_size)
                found_best = False
                if best_r2_val is None or r2_val > best_r2_val:
                    best_r2_val = r2_val
                    found_best = True
                    torch.save(net.state_dict(), 'net_best.pth')

            loss_value = np.sqrt(loss_value / n_sample)
            print("epoch: {}, loss_train: {:.4f}, r2_train: {:.2f}, r2_val: {:.2f}".format(e + 1, loss_value, r2_train, r2_val) + (" (best)" if found_best else ""))

            r2s_train[e] = r2_train
            r2s_val[e] = r2_val

        return r2s_train, r2s_val


        ############## TRAINING LOOP ##############

        for e in trange(0, epochs):  # loop on epochs
            # Azzero tutte le mie quantità
            loss_value = 0.
            r2_status = None
            nb = 0  # rappresenta l'indice del mini-batch corrente nel ciclo dei mini-batch

            for X_minibatch, y_minibatch in dataloader_train:  # loop on mini-batches
                # Clearing the previously computed gradients (are saved in memory, each iteration we need to reset the gradients)
                optimizer.zero_grad()

                X_minibatch = X_minibatch.to(device)
                y_minibatch = y_minibatch.to(device)

                outputs = net.forward(X_minibatch)  # going forward, "net" is a callable object
                loss_value_on_minibatch = loss_function(loss_function_choice, outputs, y_minibatch)  # RMSE

                with torch.no_grad():
                    r2_train_on_minibatch, r2_train, r2_status = Net_training.r2_score(outputs, y_minibatch, update_from=r2_status)

                    print("\tminibatch: {}, loss_train: {:.4f}, r2_train: {:.2f}".format(nb + 1, loss_value_on_minibatch, r2_train_on_minibatch))

                    loss_value += (loss_value_on_minibatch.item() ** 2) * X_minibatch.size(0)  # needed to estimate the train loss

                loss_value_on_minibatch.backward()  # going backward
                optimizer.step()  # updating model parameters

                nb += 1
                if nb == len(dataloader_train):
                    break

            torch.save(net.state_dict(), 'net.pth')

            for X_val_minibatch, y_val_minibatch in dataloader_val:
                X_val_minibatch = X_val_minibatch.to(device)
                y_val_minibatch = y_val_minibatch.to(device)

                r2_val, _ = Net_training.predict_and_r2_score(net, X_val_minibatch, y_val_minibatch, minibatch_size=minibatch_size)
                found_best = False
                if best_r2_val is None or r2_val > best_r2_val:
                    best_r2_val = r2_val
                    found_best = True
                    torch.save(net.state_dict(), 'net_best.pth')

            loss_value = np.sqrt(loss_value / n_sample)
            print("epoch: {}, loss_train: {:.4f}, r2_train: {:.2f}, r2_val: {:.2f}".format(e + 1, loss_value, r2_train, r2_val) + (" (best)" if found_best else ""))

            r2s_train[e] = r2_train
            r2s_val[e] = r2_val

        return r2s_train, r2s_val



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
    bah = Net_training.prova(net, data_X_train, data_y_train, data_X_val, data_y_val, epochs=5, lr=0.01, minibatch_size=32, loss_function_choice="mse", optimizer_choice="adam")
    
    

      
   