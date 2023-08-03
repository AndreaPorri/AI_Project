'''



'''


#Import needed libraries, classes and functions
import pandas as pd
import torch
import torch.nn as nn
import argparse
import yaml
from sklearn.metrics import r2_score
from functions import *
from autoencoder import *
from MLP import *
from data_generation import *
from time import sleep

#Let's put a seed (not necessary)
#torch.manual_seed(42)

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
    
        - --dimensional_red_choice: takes in input "autoencoder" or "PCA" to choose the dimensional reduction method (optional command).
                    Choices:
                    - autoencoder ---> Autoencoder method
                    - PCA ---> Principal Component Analysis method
    
    """

    #Creates an ArgumentParser object from the argparse library used for analyzing the passed arguments from the command line:
    parser = argparse.ArgumentParser(description='Process some command line arguments.') 

    #Add the arguments to the ArgumentParser object:
    parser.add_argument('--pathConfiguratorYaml', type=str, default='config_file.yaml',
                        help='Insert the path of Yaml file containing all the parameters of the project')
    parser.add_argument('--config', type=str, default='train', choices=['train', 'eval'],
                        help='You have to chose the training or evaluation configuration')
    parser.add_argument('--print_info', type=str, default='0', choices=['1','0'],
                        help='You have to chose if you want to print some controls and checks')
    parser.add_argument('--dimensional_red_choice', type=str, default='autoencoder', choices=['autoencoder', 'PCA'],
                        help='You have to choose the dimensional reduction method')
    parser.add_argument('--target_sel', type=str, default='skewness', choices=['skewness', 'no_skewness'],
                        help='You have to choose the dimensional autoeencoder model method')
            
   
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
    #GMM
    num_gaussians = yaml_configurator["num_gaussians"]
    n_samples = yaml_configurator["n_samples"]
    
    

    #The function returns all these variablesas a tuple, returning all the parameters as individual variables:
    return dataroot,results_path,reduce_dataset_path,path_pth_autoencoder,path_pth_net,path_txt_net,result_path_net,input_size_mlp,output_size_mlp,hidden_layers_mlp,epochs_net,learning_rate_net,mini_batch_size_net,loss_function_net,optimizer_net,num_gaussians,n_samples


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
    dataroot, results_path, reduce_dataset_path, path_pth_autoencoder, path_pth_net, path_txt_net, result_path_net, input_size_mlp, output_size_mlp, hidden_layers_mlp, epochs_net, learning_rate_net, mini_batch_size_net, loss_function_net, optimizer_net,num_gaussians,n_samples = load_hyperparams(pathConfiguratorYaml)


    ##########################################################################################                                    
                                        
                            ### LOAD AND PREPROCESS DATASET ###

    ##########################################################################################
    #Loading dataframe from csv file
    dataframe = pd.read_csv(dataroot, sep=";") #The .csv file uses ; as a separator instead of space
    
    #Resizing dataset from original one (I take only 6 columns)
    dataset_reduced_dirty = dataset_reduction(dataframe,"NOx(GT)","PT08.S1(CO)","T","RH","PT08.S2(NMHC)","CO(GT)")
          
    #Cleaning the reduced dataset
    dataset_reduced = cleaning_dataset_function(dataset_reduced_dirty)
    
    #Print
    if args.print_info == '1':
        #Print of the shape of the dataset and his data
        print('The dimensionality of the reduced and dirty datset is:',dataset_reduced_dirty.shape)
        print('The dimensionality of the reduced datset is: ',dataset_reduced.shape)
        sleep(4)
        print('\n\nSome rows of the reduced dataset: \n',dataset_reduced.head(5))
        sleep(8)

    ### SAVE REDUCED DATASET ###
    create_file_csv(dataset_reduced,reduce_dataset_path)
    
    ### DEVICE ###
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##########################################################################################                                    
                                        
                         ### TRAINING, VALIDATION AND TEST SETS ###

    ##########################################################################################

    #Load the data from the new CSV file and split into input and target
    data_X, data_y = load_data_from_file(reduce_dataset_path)

    #Find Max and Min of the dataset
    global_min_target = torch.min(data_y)
    global_max_target = torch.max(data_y)
    
    ### CREATE THE SETS ###
    data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test, max_val, min_val = create_splits_unbalanced(data_X, data_y, 0.7, 0.15)

    ##########################################################################################                                    
                                        
                            ### INITIALIZATION OF THE NETWORK ###

    ##########################################################################################

                                ###############################
                                         ### MLP ###
                                ###############################
   
    #Create MLP architecture
    mlp = MLP(input_size_mlp, output_size_mlp, hidden_layers_mlp)
    
    #Initializing weights
    mlp.initialize_weights()

    #Move the model to the selected device
    mlp.to(my_device)

    ##################################################################################
                                        #Data
    ##################################################################################
    '''             ################################################
                        ### PREPROCESS INPUT AND TARGET ###
                    ################################################   
    '''
    ### INPUT ###
    #Normalization data
    data_X_train, mean_train, std_train = real_norm_input(data_X_train)
    data_X_val, _, _ = real_norm_input(data_X_val, mean_train, std_train)
    data_X_test, _, _ = real_norm_input(data_X_test, mean_train, std_train)
    

    ### OUTPUT ###
    #Reduction to [0,1] the output
    #Normal target affected by positive skewness
    if args.target_sel == 'skewness':
        data_y_train = restrict_output_data(data_y_train, global_min_target, global_max_target)
        data_y_val = restrict_output_data(data_y_val, global_min_target, global_max_target)
        data_y_test = restrict_output_data(data_y_test, global_min_target, global_max_target)
    #Target not affected by positive skewness
    if args.target_sel == 'no_skewness':
        data_y_train = torch.sqrt(restrict_output_data(data_y_train, global_min_target, global_max_target))
        data_y_val = torch.sqrt(restrict_output_data(data_y_val, global_min_target, global_max_target))
        data_y_test = torch.sqrt(restrict_output_data(data_y_test, global_min_target, global_max_target))

    #Print
    if args.print_info == '1':
        #Print of mean and STD of the various set:
        print(f'\n\n\nMEAN-MEDIAN-STD TRAIN: {torch.mean(data_y_train)}-{torch.median(data_y_train)}-{torch.std(data_y_train)}')
        print(f'MEAN-MEDIAN-STD VAL: {torch.mean(data_y_val)}-{torch.median(data_y_val)}-{torch.std(data_y_val)}')
        print(f'MEAN-MEDIAN-STD TEST: {torch.mean(data_y_test)}-{torch.median(data_y_test)}-{torch.std(data_y_test)}')
        print('\n\n\n')
        sleep(10)

    ### PUT DATA ON THE DEVICE ###
    #Move the training data to the selected device
    data_X_train = data_X_train.to(my_device)
    data_y_train = data_y_train.to(my_device)
    data_X_val = data_X_val.to(my_device)
    data_y_val = data_y_val.to(my_device)
    data_X_test = data_X_test.to(my_device)
    data_y_test = data_y_test.to(my_device)
    
                    ################################################
                        ### DIMESIONALITY REDUCTION ENCODER ###
                    ################################################
    
    ### ENCODER PREDICTION + MLP ###
    if args.dimensional_red_choice == 'autoencoder':
    
        #Loading the previously trained and saved autoencoder model
        autoencoder = Autoencoder() #Create the autoencoder object
        autoencoder.load_state_dict(torch.load(path_pth_autoencoder)) #Load the model onto that object

        
        #Apply the prediction of the encoder on the input data
        data_X_train = autoencoder.predict_encoder(data_X_train).detach().to(my_device)
        data_X_val = autoencoder.predict_encoder(data_X_val).detach().to(my_device)
        data_X_test = autoencoder.predict_encoder(data_X_test).detach().to(my_device)
        
        #Print
        if args.print_info == '1':
            

            #Architecture
            print(f'\n\n\t\t- Encoder Architecture:')
            print(autoencoder.encoder)
            print(f'\n\n\t\t- MLP Architecture:')
            print(mlp)
            sleep(10)

            #MAX E MIN SETS
            print('MAX: ',torch.max(data_X_train),torch.max(data_X_val),torch.max(data_X_test))
            print('MIN: ',torch.min(data_X_train),torch.min(data_X_val),torch.min(data_X_test))
            sleep(5)
            
            ### PLOT OF THE NEW DATA ###
            # Convert x_values and y_values to numpy arrays
            x_values = data_X_train.clone().detach().cpu().numpy()
            y_values = data_y_train.clone().detach().cpu().numpy()
            # For plot: consider every 100th sample
            '''sample_interval = 100
            x_plot = x_values[::sample_interval]
            y_plot = y_values[::sample_interval]'''

            #Print of mean and STD of the various set:
            print(f'\n\n\nMEAN-MEDIAN-STD INPUT: {np.mean(x_values)}-{np.median(x_values)}-{np.std(x_values)}')
            print(f'MEAN-MEDIAN-STD OUTPUT: {np.mean(y_values)}-{np.median(y_values)}-{np.std(y_values)}')
            print('\n\n\n')
            sleep(10)
            #plot
            plt.figure(figsize=(8, 6))
            plt.scatter(x_values, y_values, s=50, marker='o', label='Points')
            plt.xlabel('X-input')
            plt.ylabel('Y-target')
            plt.title('Plot of our data')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'{results_path}/NET/new_input_data_encoded_MLP.png')
            plt.show()
        #Define the architecture for training
        net = mlp
        

    #DIMESIONALITY REDUCTION PCA
    if args.dimensional_red_choice == 'PCA':
            
            #Take the MLP part as the whole Network
            net = mlp

            #Put the data in CPU
            data_X_train_cpu = data_X_train.cpu()
            data_X_val_val = data_X_val.cpu()
            data_X_test_cpu = data_X_test.cpu()
            #Make the dimensional reduction with PCA
            data_X_train = PCA_fun(data_X_train_cpu).to(my_device)
            data_X_val = PCA_fun(data_X_val_val).to(my_device)
            data_X_test = PCA_fun(data_X_test_cpu).to(my_device)

            
            #Print
            if args.print_info == '1':
                #Architecture
                print(f'\n\n\t\t- MLP Architecture:')
                print(mlp)
                sleep(10)

                #MAX E MIN SETS
                print('MAX: ',torch.max(data_X_train),torch.max(data_X_val),torch.max(data_X_test))
                print('MIN: ',torch.min(data_X_train),torch.min(data_X_val),torch.min(data_X_test))
                sleep(5)

                ### PLOT OF THE NEW DATA ###
                # Convert x_values and y_values to numpy arrays
                x_values = data_X_train.clone().detach().cpu().numpy()
                y_values = data_y_train.clone().detach().cpu().numpy()
                
                #plot
                plt.figure(figsize=(8, 6))
                plt.scatter(x_values, y_values, s=50, marker='o', label='Points')
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                plt.title('Plot of our PCA data')
                plt.grid(True)
                plt.legend()
                plt.savefig(f'{results_path}/NET/new_input_data_encoded_MLP.png')
                plt.show()

    ### TAKE MAX AND MIN OF TRAINING SET (TO RESTRICT THE GMM GENERATION) ###
    min_train = torch.min(data_X_train)
    max_train = torch.max(data_X_train)

    #Print
    if args.print_info == '1':
        print(f'MEAN dei 3 set (tr,v,te): {torch.mean(data_X_train)} e {torch.mean(data_X_val)} e {torch.mean(data_X_test)}')
        print(f'STD dei 3 set (tr,v,te): {torch.std(data_X_train)} e {torch.std(data_X_val)} e {torch.std(data_X_test)}')
        sleep(5)
    #################################################################

                        #### TRAINING PHASE ####

    #################################################################
    if args.config == 'train':
    
        #### TRAINING LOOP ####
        #Training the network
        Net_training.training(net, data_X_train, data_X_val, data_y_train, data_y_val, path_pth_net, path_txt_net, results_path, result_path_net, args.print_info, epochs_net, learning_rate_net, mini_batch_size_net, loss_function_net, optimizer_net)
        
    #################################################################

                    #### EVALUATION PHASE ####

    #################################################################
    if args.config == 'eval':

        #################################################################

                        #### INFERENCE ON TEST SET ####

        #################################################################

        #Directory to save the evaluation part
        createDirectory(results_path)
        createDirectory(f'{results_path}/Evaluation')
        
        #Test target reshape
        data_y_test = data_y_test.reshape(-1,1)
        
        #Load net
        net.load_state_dict(torch.load(path_pth_net))
        
        #Print
        if args.print_info == '1':
            #Input Max and Min
            print('Min e Max of the input after the encoding: ',torch.min(data_X_test),torch.max(data_X_test))
            #Input shape
            print('Input shape: ',data_X_test.shape)

        ### PREDICTION ###
        outputs_test = Net_training.predict(net,data_X_test.float())
        outputs_train = Net_training.predict(net,data_X_train.float())
        outputs_val = Net_training.predict(net,data_X_val.float())
        #Concatenation for the box plot
        concatenated_tensor_output_boxp = torch.cat((outputs_test, outputs_train, outputs_val), dim=0)
        concatenated_tensor_input_boxp = torch.cat((data_X_train, data_X_val, data_X_test), dim=0)

        
        #Print
        if args.print_info == '1':
            #Calculation of mean and std target and output
            print(f'MEAN E STD TARGET:{torch.mean(data_y_test.float())},{torch.std(data_y_test.float())}')
            print(f'MEAN E STD OUTPUT:{torch.mean(outputs_test)},{torch.std(outputs_test)}')

        ### R2SCORE TEST ###
        #Pute the tensor in CPU and transform it into a numpy array
        test_y_np = data_y_test.cpu().numpy()
        outputs_test_np = outputs_test.cpu().numpy()
        #R2 score of the test set
        r2_test = r2_score(test_y_np, outputs_test_np)
        #Print r2 score
        print(f'\n\nR2 SCORE OF THE TEST PHASE LAST EPOCH: {r2_test * 100}%')
        
        ### PLOT PDY TEST SET ###
        Net_training.plot_pdfy(data_y_test,outputs_test,f'{results_path}/Evaluation/test_pdy.png')



        #################################################################

                    #### GENERATE NEW SAMPLES BY GMM ####

        #################################################################

                                ###############################
                                    ### GMM PARAMETERS ###
                                ###############################
        #These are parameters necessary to define the probability that a specific Gaussiam could be selected for 
        #generating. the means and std  lists characterizes every single Gaussian of the mixture

        #MIXING PARAMETERs
        #Definition of Gaussian mixing parameter
        mixing_parameters = np.random.rand(num_gaussians)
        #The sum of the mixing parameter must be 1, so we normalize them
        mixing_parameters /= np.sum(mixing_parameters)

        #MEANs
        means = np.random.uniform(-0.39, -0.36, num_gaussians)
        
        #STDs
        std_deviations = np.random.uniform(0.12, 0.18, num_gaussians)
        
        #SAVE PARAMETERS ON TEXT FILE
        #Combine the parameters into an array 2D, a row for each gaussian of the GMM
        parameters = np.column_stack((np.arange(num_gaussians), mixing_parameters, means, std_deviations))
        # Scrive i parametri nel file .txt
        np.savetxt(f'{results_path}/Evaluation/saved_GMM_parameters.txt', parameters, header="Gaussian_Index Mixing_Parameter Mean Standard_Deviation", fmt='%d %.6f %.6f %.6f')                  

        
                            ######################################
                                ### GMM GENERATION SAMPLES ###
                            ######################################
        
        #Generate artificial input:
        new_samples = generate_gaussian_mixture_samples(mixing_parameters, means, std_deviations, n_samples)
        
        #Print
        if args.print_info == '1':
            print(f'Artificial Sample Max e Min: {torch.min(new_samples)} e {torch.max(new_samples)}')
            print(f'Sample Max e Min: {min_train} e {max_train}')
            print('Shape of the new sample tensor: ',new_samples.shape)
            sleep(10)
        
        #Converto in float
        new_samples = new_samples.to(torch.float)
        

                        ################################################
                            ### CHECK ARTIFICIAL SAMPLES OUTPUTS ###
                        ################################################
        #Calculate Ouputs of the artificial sample
        outputs_artificial = Net_training.predict(net, new_samples)
        

        
                        ################################################
                                    ### CREATE BOXPLOT ###
                        ################################################
        if args.print_info == '1':
            print('Mediana Test data: ', torch.median(concatenated_tensor_output_boxp))
            print('Mediana Artificial data: ', torch.median(outputs_artificial))
        
        # Converti i tensori PyTorch in liste
        data_real = concatenated_tensor_output_boxp.cpu().tolist()
        data_real_input = concatenated_tensor_input_boxp.cpu().tolist()
        data_artificial = outputs_artificial.cpu().tolist()
        data_artificial_input = new_samples.cpu().tolist()

        # Crea un DataFrame di pandas con colonne nominate
        df_real = pd.DataFrame(data_real, columns=['Real'])
        df_real_input = pd.DataFrame(data_real_input, columns=['Real'])
        df_artificial = pd.DataFrame(data_artificial, columns=['Artificial'])
        df_artificial_input = pd.DataFrame(data_artificial_input, columns=['Artificial'])
        
        #selection
        real = df_real.loc[:, 'Real'].values
        real_input = df_real_input.loc[:, 'Real'].values
        artific = df_artificial.loc[:, 'Artificial'].values
        artific_input = df_artificial_input.loc[:, 'Artificial'].values

        # Creazione del plot box
        plt.figure(figsize=(8, 6))
        plt.boxplot([real_input, artific_input, real, artific], labels=['Real inp', 'Artificial inp','Real out', 'Artificial out'],meanline=True)
        plt.title('Box Plot di due set di dati')
        plt.xlabel('Set di dati')
        plt.ylabel('Valore')
        plt.grid(True)
        plt.savefig(f'{results_path}/Evaluation/box_plot_output_inp.png')
        plt.show()


                        ################################################
                                    ### CREATE PDF PLOTS ###
                        ################################################
        Net_training.plot_pdfy(concatenated_tensor_input_boxp, new_samples, f'{results_path}/Evaluation/pdx_artificial_real.png')
        Net_training.plot_pdfy(concatenated_tensor_output_boxp, outputs_artificial, f'{results_path}/Evaluation/pdy_artificial_real.png')


                        #######################################################
                                    ### CREATE BOXPLOT WITHOUT OUTLIERS ###
                        #######################################################

        #Calcolo mean and std
        mean_value_art = torch.mean(outputs_artificial)
        std_value_art = torch.std(outputs_artificial)
        
        ################## RIMOZIONE OULIERS IN INPUT #############
        # Calcola il primo quartile (Q1) e il terzo quartile (Q3)
        Q1 = np.percentile(df_artificial_input.loc[:, 'Artificial'].values, 25)
        Q3 = np.percentile(df_artificial_input.loc[:, 'Artificial'].values, 75)

        # Calcola l'IQR
        IQR = Q3 - Q1

        # Definisci gli intervalli per gli outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #Print
        if args.print_info == '1':
            print("Lower Bound:", lower_bound)
            print("Upper Bound:", upper_bound)
            print("Min value:", df_artificial_input.loc[:, 'Artificial'].values.min())
            print("Max value:", df_artificial_input.loc[:, 'Artificial'].values.max())
            
        #Condition
        condition = (df_artificial_input['Artificial'] >= lower_bound) & (df_artificial_input['Artificial'] <= upper_bound)

        # Filtra gli outlier
        filtered_ARTIFICIAL_input_data = df_artificial_input[condition]
        
        #Rifaccio
        artific_input_fil = filtered_ARTIFICIAL_input_data.loc[:, 'Artificial'].values

        #Put it into torch tensor
        artific_input_fil_tens = torch.tensor(artific_input_fil).reshape(-1,1)

        #Obtain the prediction
        artific_output_fil = Net_training.predict(net, artific_input_fil_tens.float())

        # Converti i tensori PyTorch in liste
        data_art_fil = artific_output_fil.cpu().tolist()
        # Crea un DataFrame di pandas con colonne nominate
        data_art_fil = pd.DataFrame(data_art_fil, columns=['Artificial'])
        #selection
        data_art_fil_out = data_art_fil.loc[:, 'Artificial'].values

        # Creazione del plot box
        plt.figure(figsize=(8, 6))
        plt.boxplot([real, data_art_fil_out, artific], labels=['Real out', 'Artificial\n filtered out', 'Artificial\n not filtered out'],meanline=True)
        plt.title('Box Plot di due set di dati')
        plt.xlabel('Set di dati')
        plt.ylabel('Valore')
        plt.grid(True)
        plt.savefig(f'{results_path}/Evaluation/box_plot_output_inp_fil1.png')
        plt.show()

        # Creazione del plot box
        plt.figure(figsize=(8, 6))
        plt.boxplot([real_input ,artific_input_fil, artific_input], labels=['Real inp','Artificial\n filtered inp', 'Artificial\n not filtered inp'],meanline=True)
        plt.title('Box Plot di due set di dati')
        plt.xlabel('Set di dati')
        plt.ylabel('Valore')
        plt.grid(True)
        plt.savefig(f'{results_path}/Evaluation/box_plot_output_inp_fil2.png')
        plt.show()

        #Saves
        #Write a .txt file to the specified path and writes information regarding the epoch and the loss to which
        #the best trained net belongs
        with open('D:/Results/NET/artificial_info.txt', "w") as f:
            print(f"Info net:\n\n\tMEAN VALIDATION:\t{mean_value_art}\n\n\tSTD VALIDATION:\t{std_value_art}", file=f)
            
        #Stampo la PDFY artificiale
        #Net_training.plot_pdfy_prova(outputs_artificial, 'D:/Results/NET/ARTIFICIAL_pfdy.png')
