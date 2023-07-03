import pandas as pd
from dataset_function import *
import torch
import torch.nn as nn
import matplotlib as plt


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])

    def add_hidden_layer(self, hidden_size):
        self.layers.insert(-1, nn.Linear(self.layers[-1].out_features, hidden_size))
        self.layers.insert(-1, nn.Tanh())

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and not isinstance(layer, nn.Tanh):
                x = torch.sigmoid(x)  # Applica sigmoid all'ultimo layer prima dell'output
        return x


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

        
        '''
        Se il parametro update_from viene fornito, significa che la funzione viene chiamata per aggiornare 
        un punteggio R2 esistente con nuovi dati(caso di online mode - mini-batch). Se il parametro update_from non viene 
        fornito, significa che la funzione viene chiamata per calcolare il punteggio R2 da zero su un singolo 
        gruppo di dati(batch mode).
        '''        
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

                _, r2, r2_status = Net_training.r2_score(output, y_minibatch, update_from=r2_status)

                if t == n:
                    break

            if training_mode:
                net.train()

            return r2, torch.cat(outputs, dim=0)

    def train(net, X_train, y_train, X_val, y_val, epochs=2000, lr=0.001, minibatch_size=32):
        """Train a neural network for multiple epochs.

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

        def rmse_loss(o, y):
            return torch.sqrt(torch.nn.functional.mse_loss(o, y, reduction='mean'))

        net.train()  # telling the network we are going to train it (and not to simply evaluate it)
        loss = rmse_loss  # defining the loss function
        optimizer = torch.optim.Adam(net.parameters(), lr)  # defining the way we are going to update the net parameters
        device = next(net.parameters()).device  # we assume that all the network parameters are on the same device

        n = X_train.shape[0]
        best_r2_val = None
        r2s_train = np.zeros(epochs)
        r2s_val = np.zeros(epochs)

        for e in range(0, epochs):  # loop on epochs
            loss_value = 0.
            r2_status = None
            t = 0
            nb = 0

            while True:  # loop on mini-batches
                optimizer.zero_grad()  # clearing the previously computed gradients

                f = t
                t = min(f + minibatch_size, n)
                X_minibatch = X_train[f:t, :].to(device)
                y_minibatch = y_train[f:t].to(device)

                outputs = net(X_minibatch)  # going forward, "net" is a callable object
                loss_value_on_minibatch = loss(outputs, y_minibatch)  # RMSE

                with torch.no_grad():
                    r2_train_on_minibatch, r2_train, r2_status = Net_training.r2_score(outputs, y_minibatch, update_from=r2_status)

                    print("\tminibatch: {}, loss_train: {:.4f}, "
                        "r2_train: {:.2f}".format(nb + 1, loss_value_on_minibatch, r2_train_on_minibatch))

                    loss_value += (loss_value_on_minibatch.item() ** 2) * (t - f)  # needed to estimate the train loss

                loss_value_on_minibatch.backward()  # going backward
                optimizer.step()  # updating model parameters

                nb += 1
                if t == n:
                    break

            torch.save(net.state_dict(), 'net.pth')

            r2_val, _ = Net_training.predict_and_r2_score(net, X_val, y_val, minibatch_size=minibatch_size)
            found_best = False
            if best_r2_val is None or r2_val > best_r2_val:
                best_r2_val = r2_val
                found_best = True
                torch.save(net.state_dict(), 'net_best.pth')

            loss_value = np.sqrt(loss_value / n)
            print("epoch: {}, loss_train: {:.4f}, "
                "r2_train: {:.2f}, r2_val: {:.2f}".format(e + 1, loss_value, r2_train, r2_val)
                + (" (best)" if found_best else ""))

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

    #Carichiamo i dati dal nuovo file .csv e dividiamo in input e output
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

    
    
    #Stampe forma dati
    #print("DATI:\n",data_X_train)
    #print(f"MEDIA:{m}, STD:{s}, MAX:{max}, MIN:{min}")
    
    '''
    ### CONTEGGIO NEGATIVI POST NORMALIZZAZIONE + DOMANDA ###
    # Converti il tensore Torch in un tensore booleano con valori True per i numeri negativi
    negative_mask = (data_X_train < 0)

    # Conta il numero di valori True nel tensore booleano
    num_negatives = torch.sum(negative_mask).item()

    print("Numero di numeri negativi nel tensore:", num_negatives)
    
    #Domanda: è normale che io abbia più numeri negativi post normalizzazione? In teoria questa dovrebbe mantenere la distribuzione relativa dei dati nel proprio intervallo.
    #        Oppure conviene fissare max = 100 e min = -100 e quindi viene una distribuzione sbilanciata ma che mantiene i segni? normalize_input_data_100
    '''
    
    
    #Ensure that we keep track of the mean and std used to normalize the data
    torch.save([m, s], 'C:/Users/andre/OneDrive/Desktop/MAGISTRALE/AI_Project/Dataset/normalizers_hyperparam.pth')

    
 
    
    

      
   