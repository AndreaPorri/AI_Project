'''
                        ############################################################

                                        MLP AND GMM GENERATORATING

                        ############################################################

The provided code train/validate/evaluate a Multi-Layer Perceptron (MLP) and after using a GMM generate new artificial samples
which will be provide to this MLP in order to get results, after the pdf will be compared.
The code performs the following steps, which are briefly explained below:

1. It parses command-line arguments, such as specifying the YAML configuration file,
   choosing the training or evaluation phase, selecting dimensional reduction method (autoencoder or PCA),
   and other parameters. The two special functions loads hyperparameters from the specified YAML configuration file.

3. Loads the dataset from a CSV file, reduces its dimensionality by selecting specific columns, and then cleans it.

4. Splits the dataset into training, validation, and test sets.

5. Initializes the MLP architecture, applies normalization to the input data, and performs a data preprocessing for the target data.

6. If the chosen dimensional reduction method is "autoencoder," it loads a pre-trained autoencoder model and 
   applies the encoder to the input data. After that the input is a 1D tensor.

7. If the chosen dimensional reduction method is "PCA," it applies Principal Component Analysis (PCA) to the input data.
   After that the input is a 1D tensor.

8. If the configuration is set to "train," the code proceeds with the training phase of the MLP using the specified hyperparameters,
   the 1D input and the preprocessed 1D outputs.

9. If the configuration is set to "eval," the code performs inference on the test set using the trained MLP model. It calculates the R2 score for the test set.

10. Then generates new artificial samples using the parameters of a Gaussian Mixture Model (GMM) specified in the
    YAML configuration file. Then we pass this artificial input throught the pre-trained MLP obtaining the outputs.
    Then visualizes the results using box plots and probability density plots for the real and artificial X and y pdf.


EXECUTION EXAMPLE:
    - Standard training:   python main.py --print_info='1'
    - Standard evaluation:   python main.py --print_info='1' --config='eval'
    - Standard limitate GMM:   python main.py --print_info='1' --config='generating'

    
MUST: First of all, you have to execute the 'autoencoder.py' and then proceed with the training !!! 
'''

#Import needed libraries, classes and functions
import pandas as pd
import torch
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
    Function required for using additional terminal commands necessary for selecting specific choices needed for the correct execution of MLP.
    The main commands are:

        - --pathConfiguratorYaml: takes in input the yaml file path as a string

        - --config: takes in input "train" or "eval" for selecting the training or the evaluation phase, if not specified the default value is "train" (optional command).
                    Choices: 
                    - train ---> training phase
                    - eval ---> evaluation phase
                    - generating ---> Generating GMM and evaluation phase

        - --print_info: takes in input a string parameter ('0' or '1') which tell if you want to print some checks (optional command).
    
        - --dimensional_red_choice: takes in input "autoencoder" or "PCA" to choose the dimensional reduction method (optional command).
                    Choices:
                    - autoencoder ---> Autoencoder method
                    - PCA ---> Principal Component Analysis method
        - --target_sel: takes in input "skewness" or "no_skewness" to choose if you want remove the skewness problem from the target or not (optional command).
                    Choices:
                    - skewness ---> Target
                    - no_skewness ---> sqrt(Target)
        - --GMM_limit: takes in input "limitate" or "not_limitate" to choose if you want limitate the GMM generation (optional command).
                    Choices:
                    - limitate ---> GMM generation to the range [min_X_training,max_X_training], so on the input training domain
                    - not_limitate ---> GMM generation without any external limitation.
    
    """

    #Creates an ArgumentParser object from the argparse library used for analyzing the passed arguments from the command line:
    parser = argparse.ArgumentParser(description='Process some command line arguments.') 

    #Add the arguments to the ArgumentParser object:
    parser.add_argument('--pathConfiguratorYaml', type=str, default='config_file.yaml',
                        help='Insert the path of Yaml file containing all the parameters of the project')
    parser.add_argument('--config', type=str, default='train', choices=['train', 'eval', 'generating'],
                        help='You have to chose the training, evaluation or generating configuration')
    parser.add_argument('--print_info', type=str, default='0', choices=['1','0'],
                        help='You have to chose if you want to print some controls and checks')
    parser.add_argument('--dimensional_red_choice', type=str, default='autoencoder', choices=['autoencoder', 'PCA'],
                        help='You have to choose the dimensional reduction method')
    parser.add_argument('--target_sel', type=str, default='skewness', choices=['skewness', 'no_skewness'],
                        help='You have to choose if you want, or not, fix the problem of positive skewness on the input data')
    parser.add_argument('--GMM_limit', type=str, default='limitate', choices=['limitate', 'not_limitate'],
                        help='You have to choose if you want, or not, put a limitation on the samples generated by the GMM')
            
           
   
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
    split_path = yaml_configurator['split_path']
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
    return dataroot,results_path,reduce_dataset_path,path_pth_autoencoder,path_pth_net,path_txt_net,result_path_net,split_path,input_size_mlp,output_size_mlp,hidden_layers_mlp,epochs_net,learning_rate_net,mini_batch_size_net,loss_function_net,optimizer_net,num_gaussians,n_samples


################################################################################

                                #MAIN CODE START

################################################################################


### START ###
if __name__ == "__main__":
    
    ### TERMINAL ###
    args = parse_command_line_arguments() #extracts the arguments from the command line and saves them in the "args" object
    
    ### YAML ###
    pathConfiguratorYaml = args.pathConfiguratorYaml #extracts the path of the YAML configuration file from the command line and saves it in a variable
    #We assign the values returned by the function, that is the values in the tuple, to the respective variables
    dataroot, results_path, reduce_dataset_path, path_pth_autoencoder, path_pth_net, path_txt_net, result_path_net, split_path, input_size_mlp, output_size_mlp, hidden_layers_mlp, epochs_net, learning_rate_net, mini_batch_size_net, loss_function_net, optimizer_net,num_gaussians,n_samples = load_hyperparams(pathConfiguratorYaml)

    ### RESULT DIRECTORY ###
    createDirectory(results_path)
    createDirectory(f'{results_path}/NET')

    ### DEVICE ###
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''             #################################################################

                                      #### TRAINING PHASE ####

                    ################################################################# 
    '''
    
    if args.config == 'train':
        ##########################################################################################                                    
                                            
                                ### LOAD AND REDUCE DATASET ###

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
            print('Some dataset information:')
            print('\t- The dimensionality of the reduced and dirty datset is:',dataset_reduced_dirty.shape)
            print('\t- The dimensionality of the reduced datset is: ',dataset_reduced.shape)
            print('\nSome rows of the reduced dataset: \n',dataset_reduced.head(5))
            sleep(8)

        ### SAVE REDUCED DATASET ###
        create_file_csv(dataset_reduced,reduce_dataset_path)

        ##########################################################################################                                    
                                            
                            ### TRAINING, VALIDATION AND TEST SETS ###

        ##########################################################################################

        #Load the data from the new CSV file and split into input and target
        data_X, data_y = load_data_from_file(reduce_dataset_path)

        ### MAX AND MIN TARGET FOR PREPROCESS TARGET ###
        global_min_target = torch.min(data_y)
        global_max_target = torch.max(data_y)
        
        ### CREATE THE SETS ###
        data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test, max_val, min_val = create_splits_unbalanced(data_X, data_y, 0.7, 0.15)

        #Print
        if args.print_info == '1':
            #Shapes of the three sets
            print('\n\nShapes of the input and target of the three sets:')
            print(f'\t- Shapes input training sets: {data_X_train.shape}')
            print(f'\t- Shapes output training sets: {data_y_train.shape}')
            print(f'\t- Shapes input validation sets: {data_X_val.shape}')
            print(f'\t- Shapes output validation sets: {data_y_val.shape}')
            print(f'\t- Shapes input test sets: {data_X_test.shape}')
            print(f'\t- Shapes output test sets: {data_y_test.shape}\n\n')


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
                        ################################################
                            ### PREPROCESS INPUT AND TARGET ###
                        ################################################   
        ### INPUT ###
        #Normalization data
        data_X_train, mean_train, std_train = real_norm_input(data_X_train)
        data_X_val, _, _ = real_norm_input(data_X_val, mean_train, std_train)
        data_X_test, _, _ = real_norm_input(data_X_test, mean_train, std_train)

        ### TARGET ###
        #Preprocess the target to [0,1]
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
            print(f'\nCharacteristics of the target 1D data preprocessed with {args.target_sel} problem: ')
            print(f'\t- MEAN-MEDIAN-STD-MIN-MAX TARGET TRAIN: {torch.mean(data_y_train)} / {torch.median(data_y_train)} / {torch.std(data_y_train)} / {torch.min(data_y_train)} / {torch.max(data_y_train)}')
            print(f'\t- MEAN-MEDIAN-STD-MIN-MAX TARGET VAL: {torch.mean(data_y_val)} / {torch.median(data_y_val)} / {torch.std(data_y_val)} / {torch.min(data_y_train)} / {torch.max(data_y_train)}')
            print(f'\t- MEAN-MEDIAN-STD-MIN-MAX TARGET TEST: {torch.mean(data_y_test)} / {torch.median(data_y_test)} / {torch.std(data_y_test)} / {torch.min(data_y_train)} / {torch.max(data_y_train)}')
            sleep(10)
        
        #Put the target data onto the correct device
        data_y_train = data_y_train.to(my_device)
        data_y_val = data_y_val.to(my_device)
        data_y_test = data_y_test.to(my_device)
        
                        ################################################
                            ### DIMESIONALITY REDUCTION ENCODER ###
                        ################################################
        
        ### ENCODER PREDICTION + MLP ###
        if args.dimensional_red_choice == 'autoencoder':
            
            print('\n\n\nEncoder method for DR is selected...')
            sleep(6)
            
            ### NET FOR TRAINING AND EVALUATION ###
            net = mlp

            ### AUTOENCODER ###
            #Loading the previously trained and saved autoencoder model
            autoencoder = Autoencoder() #Create the autoencoder object
            autoencoder.load_state_dict(torch.load(path_pth_autoencoder)) #Load the model onto that object

            ### DIMENSIONAL REDUCTION ENCODER ###
            #Apply the prediction of the encoder on the input data, so we obtain the dimensional reduction
            data_X_train = autoencoder.predict_encoder(data_X_train).detach().to(my_device)
            data_X_val = autoencoder.predict_encoder(data_X_val).detach().to(my_device)
            data_X_test = autoencoder.predict_encoder(data_X_test).detach().to(my_device)
            
            #Print
            if args.print_info == '1':
                
                ### Architecture of the Encoder and MLP ###
                print(f'\n- Encoder Architecture:')
                print(autoencoder.encoder)
                print(f'\n- MLP Architecture:')
                print(mlp)
                sleep(10)

                ### INPUT ###
                #Characteristic of the input 1D dimensional reduced
                print('\nCharacteristics of the normalized preprocessed input 1D: ')
                print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT TRAIN: {torch.mean(data_X_train)} / {torch.median(data_X_train)} / {torch.std(data_X_train)} / {torch.min(data_X_train)} / {torch.max(data_X_train)}')
                print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT VAL: {torch.mean(data_X_val)} / {torch.median(data_X_val)} / {torch.std(data_X_val)} / {torch.min(data_X_val)} / {torch.max(data_X_val)}')
                print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT TEST: {torch.mean(data_X_test)} / {torch.median(data_X_test)} / {torch.std(data_X_test)} / {torch.min(data_X_test)} / {torch.max(data_X_test)}')
                sleep(8)
                
                ### PLOT OF THE NEW DATA ###
                plot_initial_data_afterDR(data_X_train, data_y_train, 'Preprocessed data points of the training set - Encoder', f'{results_path}/NET/training_new_data_ENCODER.png')
                plot_initial_data_afterDR(data_X_val, data_y_val, 'Preprocessed data points of the validation set - Encoder', f'{results_path}/NET/validation_new_data_ENCODER.png')
                plot_initial_data_afterDR(data_X_test, data_y_test, 'Preprocessed data points of the test set - Encoder', f'{results_path}/NET/test_new_data_ENCODER.png')


        #DIMESIONALITY REDUCTION PCA
        if args.dimensional_red_choice == 'PCA':
                
                print('PCA method for DR is selected ...')
                sleep(6)
                
                ### NET FOR TRAINING AND EVALUATION ###
                net = mlp

                ### DIMENSIONAL REDUCTION PCA ###
                data_X_train = PCA_fun(data_X_train.cpu()).to(my_device)
                data_X_val = PCA_fun(data_X_val.cpu()).to(my_device)
                data_X_test = PCA_fun(data_X_test.cpu()).to(my_device)

                
                #Print
                if args.print_info == '1':
                    ### Architecture of the PCA and MLP ###
                    print(f'\n\n- PCA Class:')
                    print(PCA)
                    print(f'\n- MLP Architecture:')
                    print(mlp)
                    sleep(10)

                    ### INPUT ###
                    #Characteristic of the input 1D dimensional reduced
                    print('\nCharacteristics of the normalized preprocessed input 1D: ')
                    print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT TRAIN: {torch.mean(data_X_train)} / {torch.median(data_X_train)} / {torch.std(data_X_train)} / {torch.min(data_X_train)} / {torch.max(data_X_train)}')
                    print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT VAL: {torch.mean(data_X_val)} / {torch.median(data_X_val)} / {torch.std(data_X_val)} / {torch.min(data_X_val)} / {torch.max(data_X_val)}')
                    print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT TEST: {torch.mean(data_X_test)} / {torch.median(data_X_test)} / {torch.std(data_X_test)} / {torch.min(data_X_test)} / {torch.max(data_X_test)}')
                    sleep(8)

                    
                    ### PLOT OF THE NEW DATA ###    
                    plot_initial_data_afterDR(data_X_train, data_y_train, 'Preprocessed data points of the training set - PCA', f'{results_path}/NET/training_new_data_PCA.png')
                    plot_initial_data_afterDR(data_X_val, data_y_val, 'Preprocessed data points of the validation set - PCA', f'{results_path}/NET/validation_new_data_PCA.png')
                    plot_initial_data_afterDR(data_X_test, data_y_test, 'Preprocessed data points of the test set - PCA', f'{results_path}/NET/test_new_data_PCA.png')


        #### TRAINING THE NETWORK ####
        Net_training.training(net, data_X_train, data_X_val, data_y_train, data_y_val, path_pth_net, path_txt_net, results_path, result_path_net, args.print_info, epochs_net, learning_rate_net, mini_batch_size_net, loss_function_net, optimizer_net)
        
        ### CREATE THE CSV FILE OF THIS SPLIT ###
        create_file_csv(pd.DataFrame(data_X_train.cpu().numpy()),f'{split_path}/X_train.csv')
        create_file_csv(pd.DataFrame(data_X_val.cpu().numpy()),f'{split_path}/X_val.csv')
        create_file_csv(pd.DataFrame(data_X_test.cpu().numpy()),f'{split_path}/X_test.csv')
        create_file_csv(pd.DataFrame(data_y_train.cpu().numpy()),f'{split_path}/y_train.csv')
        create_file_csv(pd.DataFrame(data_y_val.cpu().numpy()),f'{split_path}/y_val.csv')
        create_file_csv(pd.DataFrame(data_y_test.cpu().numpy()),f'{split_path}/y_test.csv')          
    '''
                    #################################################################

                                    #### EVALUATION PHASE ####

                    #################################################################
    
    '''
    if args.config == 'eval':
        
        ### DIRECTORY FOR EVALUATION PART ###
        createDirectory(f'{results_path}/Evaluation')

        ### LOAD THE TEST SET ###
        data_X_test = torch.tensor(pd.read_csv(f'{split_path}/X_test.csv').to_numpy()).to(my_device)
        data_y_test = torch.tensor(pd.read_csv(f'{split_path}/y_test.csv').to_numpy()).to(my_device)
   
        ### LOAD THE PRETRAINED NET ### 
        net = MLP(input_size_mlp, output_size_mlp, hidden_layers_mlp).to(my_device)
        net.load_state_dict(torch.load(path_pth_net))
                
        #################################################################

                        #### INFERENCE ON TEST SET ####

        #################################################################

        ### PREDICTION ON TEST SET ###
        outputs_test = Net_training.predict(net,data_X_test.float())
                
        #Print
        if args.print_info == '1':
            #Calculation of mean and std target and output
            print('\n\nMean e Std of the test set part:')
            print(f'\t- MEAN E STD TARGET:{torch.mean(data_y_test.float())},{torch.std(data_y_test.float())}')
            print(f'\t- MEAN E STD OUTPUT:{torch.mean(outputs_test)},{torch.std(outputs_test)}')
            print('Shape of the tensors:')
            print('\t- Input sape: ',data_X_test.shape)
            print('\t- Target sape: ',data_y_test.shape)
            print('\t- Output sape: ',outputs_test.shape)
            sleep(10)

        ### R2SCORE TEST ###
        #Put the tensor in CPU and transform it into a numpy array
        test_y_np = data_y_test.cpu().numpy()
        outputs_test_np = outputs_test.cpu().numpy()
        #R2 score of the test set
        r2_test = r2_score(test_y_np, outputs_test_np)
        #Print r2 score
        print(f'\nR2 SCORE OF THE TEST PHASE LAST EPOCH: {r2_test * 100}%')

        
        ### PREPARE THE DATA FOR THE PLOT ###  
            
        #Convert Pytorch tensor into list on CPU
        data_test_target = data_y_test.cpu().tolist()
        data_test_output = outputs_test.cpu().tolist()
        
        #Create the DataFrames
        df_target = pd.DataFrame(data_test_target, columns=['Target'])
        df_output = pd.DataFrame(data_test_output, columns=['Output'])
                
        #Take the values from the dataframes
        target_test_bp = df_target.loc[:, 'Target'].values
        output_test_bp = df_output.loc[:, 'Output'].values
            
        
        ### BOX PLOT TO COMPARE TARGET AND OUTPUT TEST SET ###
        plot_box([target_test_bp, output_test_bp],['Target', 'Output'],'Boxplot target and output test set pdf','y data','Pdf values',f'{results_path}/Evaluation/boxplot_test_pdy.png') 
        ### PLOT PDY TEST SET ###
        plot_pdf(data_y_test, outputs_test, 'Target test set', 'Output test set', 'Comparison target and output test set', f'{results_path}/Evaluation/test_pdy.png')

    '''
                    #################################################################

                                #### GENERATE NEW SAMPLES BY GMM ####

                    #################################################################
    '''
    
    if args.config == 'generating':

        ### DIRECTORY FOR EvALUATION PART ###
        createDirectory(f'{results_path}/GMM')

        ### LOAD THE TEST SET ###
        data_X_train = torch.tensor(pd.read_csv(f'{split_path}/X_train.csv').to_numpy()).to(my_device)
        data_X_val = torch.tensor(pd.read_csv(f'{split_path}/X_val.csv').to_numpy()).to(my_device)
        data_X_test = torch.tensor(pd.read_csv(f'{split_path}/X_test.csv').to_numpy()).to(my_device)
        data_y_train = torch.tensor(pd.read_csv(f'{split_path}/y_train.csv').to_numpy()).to(my_device)
        data_y_val = torch.tensor(pd.read_csv(f'{split_path}/y_val.csv').to_numpy()).to(my_device)
        data_y_test = torch.tensor(pd.read_csv(f'{split_path}/y_test.csv').to_numpy()).to(my_device)
   
        ### LOAD THE PRETRAINED NET ### 
        net = MLP(input_size_mlp, output_size_mlp, hidden_layers_mlp).to(my_device)
        net.load_state_dict(torch.load(path_pth_net))
        

                                        ###############################
                                            ### GMM PARAMETERS ###
                                        ###############################
        #These are the parameters necessary to define the probability that a specific Gaussian could be selected from the mixture
        #for generating a sample. The means and standard deviation (std) lists characterize every single Gaussian of the mixture.
        

        ### MIXING PARAMETERs ###
        # Definition of Gaussian mixing parameters that sum up to 1
        mixing_parameters = np.random.rand(num_gaussians)
        mixing_parameters /= np.sum(mixing_parameters)
        
        ### MEANs and STDs ###
        #Randomly generates parameters (means and standard deviations) for the Gaussian mixture model with a specified number of gaussians.
        #These range are defined observing the mean and std of the real input data.
        means = np.random.uniform(-0.39, -0.36, num_gaussians)
        std_deviations = np.random.uniform(0.12, 0.18, num_gaussians)
        
        #SAVE PARAMETERS ON TEXT FILE
        #Combine the parameters into an array 2D, a row for each gaussian of the GMM
        parameters = np.column_stack((np.arange(num_gaussians), mixing_parameters, means, std_deviations))
        #Write the parameters in txt file
        np.savetxt(f'{results_path}/GMM/saved_GMM_parameters.txt', parameters, header="Gaussian_Index Mixing_Parameter Mean Standard_Deviation", fmt='%d %.6f %.6f %.6f')                  

        
                                    ######################################
                                        ### GMM GENERATION SAMPLES ###
                                    ######################################
        if args.GMM_limit == 'limitate':
            ### GENERATE LIMITATE ARTIFICIAL INPUT ###
            #The new samples are limited to the range of the training input, between the minimum and maximum values of the training set input after the dimensional reduction.
            new_samples = generate_gaussian_mixture_samples(mixing_parameters, means, std_deviations, n_samples, torch.min(data_X_train), torch.max(data_X_train))
        
        if args.GMM_limit == 'not_limitate':
            ### GENERATE ARTIFICIAL INPUT ###
            new_samples = generate_gaussian_mixture_samples(mixing_parameters, means, std_deviations, n_samples)
        
        
        #Print
        if args.print_info == '1':
            #Artificial samples
            print('\n\nBelow there are some characteristics of the new artificial input distributed as the real 1D input:')
            print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT ARTIFICIAL SAMPLES: {torch.mean(new_samples)} / {torch.median(new_samples)} / {torch.std(new_samples)} / {torch.min(new_samples)} / {torch.max(new_samples)}')
            print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT TRAIN: {torch.mean(data_X_train)} / {torch.median(data_X_train)} / {torch.std(data_X_train)} / {torch.min(data_X_train)} / {torch.max(data_X_train)}')
            print('\t- Shape of the new sample tensor: ',new_samples.shape)
            sleep(8)
                

                                    
                                    ################################################
                                        ### CHECK ARTIFICIAL SAMPLES OUTPUTS ###
                                    ################################################
        
        ### PREDICTION OF ARTIFICIAL SAMPLES ### 
        outputs_artificial = Net_training.predict(net, new_samples.to(torch.float))
        
        #Print
        if args.print_info == '1':
            #Artificial samples output
            print('\n\nBelow there are some characteristics of the artificial output:')
            print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT ARTIFICIAL SAMPLES: {torch.mean(outputs_artificial)} / {torch.median(outputs_artificial)} / {torch.std(outputs_artificial)} / {torch.min(outputs_artificial)} / {torch.max(outputs_artificial)}')
            print(f'\t- MEAN-MEDIAN-STD-MIN-MAX INPUT TRAIN: {torch.mean(data_y_train)} / {torch.median(data_y_train)} / {torch.std(data_y_train)} / {torch.min(data_y_train)} / {torch.max(data_y_train)}')
            print('\t- Shape of the new sample tensor: ',new_samples.shape)
            print('\t- Shape of the output of new sample tensor: ',outputs_artificial.shape)
            sleep(8)
        
        
                              
                                    ################################################
                                                ### CREATE BOXPLOT ###
                                    ################################################
        
        ### PREDICTIONS ###
        outputs_test = Net_training.predict(net,data_X_test.float())
        outputs_train = Net_training.predict(net,data_X_train.float())
        outputs_val = Net_training.predict(net,data_X_val.float())

        ### PREPARE THE DATA FOR THE PLOT ###
        #Concatenation for the sets of the real data
        concatenated_tensor_output_boxp = torch.cat((outputs_test, outputs_train, outputs_val), dim=0)
        concatenated_tensor_input_boxp = torch.cat((data_X_train, data_X_val, data_X_test), dim=0)
        
        #Convert Pytorch tensor into list on CPU
        data_real = concatenated_tensor_output_boxp.cpu().tolist()
        data_real_input = concatenated_tensor_input_boxp.cpu().tolist()
        data_artificial = outputs_artificial.cpu().tolist()
        data_artificial_input = new_samples.cpu().tolist()

        #Create the DataFrames
        df_real = pd.DataFrame(data_real, columns=['Real'])
        df_real_input = pd.DataFrame(data_real_input, columns=['Real'])
        df_artificial = pd.DataFrame(data_artificial, columns=['Artificial'])
        df_artificial_input = pd.DataFrame(data_artificial_input, columns=['Artificial'])
        
        #Take the values from the dataframes
        real = df_real.loc[:, 'Real'].values
        real_input = df_real_input.loc[:, 'Real'].values
        artific = df_artificial.loc[:, 'Artificial'].values
        artific_input = df_artificial_input.loc[:, 'Artificial'].values

        ### BOX PLOT ###
        ### BOX PLOT TO COMPARE REAL AND ARTIFICIAL INPUT ###
        plot_box([real_input, artific_input],['Real input', 'Artificial input'],'Boxplot real and artificial input pdf','X data','Pdf values',f'{results_path}/GMM/real_artific_input.png') 
        ### BOX PLOT TO COMPARE REAL AND ARTIFICIAL OUTPUT ###
        plot_box([real, artific],['Real output', 'Artificial output'],'Boxplot real and artificial output pdf','y data','Pdf values',f'{results_path}/GMM/real_artific_output.png') 
        
                        
                                    ################################################
                                                ### CREATE PDF PLOTS ###
                                    ################################################
        ### PDF PLOT TO COMPARE REAL AND ARTIFICIAL INPUT ###
        plot_pdf(concatenated_tensor_input_boxp, new_samples, 'Real input', 'Artificial input', 'Comparison real and artificial input', f'{results_path}/GMM/pdx_artificial_real.png')
        ### PDF PLOT TO COMPARE REAL AND ARTIFICIAL OUTPUT ###
        plot_pdf(concatenated_tensor_output_boxp, outputs_artificial, 'Real output', 'Artificial output', 'Comparison real and artificial output', f'{results_path}/GMM/pdy_artificial_real.png')

                                    
                                    
                                    #######################################################
                                                ### CREATE BOXPLOT WITHOUT OUTLIERS ###
                                    #######################################################
        
        ### RIMOZIONE OULIERS IN INPUT ###
        #Calculate the first quartile (Q1) and the third quartile (Q3)
        Q1 = np.percentile(df_artificial_input.loc[:, 'Artificial'].values, 25) #25% but we can choose 10% for example
        Q3 = np.percentile(df_artificial_input.loc[:, 'Artificial'].values, 75) #75% but we can choose 90% for example

        #Calculate IQR
        IQR = Q3 - Q1

        #Define bounds to find the outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #Print
        if args.print_info == '1':
            #Compare bounds wrt min-max values
            print("\n\nLower Bound: ", lower_bound)
            print("Upper Bound: ", upper_bound)
            print("Min value: ", df_artificial_input.loc[:, 'Artificial'].values.min())
            print("Max value: ", df_artificial_input.loc[:, 'Artificial'].values.max())
            
        #Condition to remove outliers
        condition = (df_artificial_input['Artificial'] >= lower_bound) & (df_artificial_input['Artificial'] <= upper_bound)

        #Remove outliers from the dataset
        filtered_ARTIFICIAL_input_data = df_artificial_input[condition]
        
        #Take the values from the dataframes
        artific_input_fil = filtered_ARTIFICIAL_input_data.loc[:, 'Artificial'].values

        #Put it into torch tensor and reshape it in a correct dimensionality
        artific_input_fil_tens = torch.tensor(artific_input_fil).reshape(-1,1)

        #Obtain the prediction
        artific_output_fil = Net_training.predict(net, artific_input_fil_tens.float())

        ### PREPARE THE DATA FOR THE PLOT ###
        #Convert Pytorch tensor into list on CPU
        data_art_fil = artific_output_fil.cpu().tolist()
        #Create the DataFrames
        data_art_fil = pd.DataFrame(data_art_fil, columns=['Artificial'])
        #Take the values from the dataframes
        data_art_fil_out = data_art_fil.loc[:, 'Artificial'].values

        ### BOX PLOT ###
        ### BOX PLOT TO COMPARE REAL AND FILTERED/NOT FILTERED ARTIFICIAL INPUT ###
        plot_box([real_input ,artific_input_fil, artific_input],['Real input','Artificial\n filtered input', 'Artificial\n not filtered input'],'Boxplot real and filtered/not filtered artificial input pdf','X data','Pdf values',f'{results_path}/GMM/real_f_nf_artific_input.png') 
        ### BOX PLOT TO COMPARE REAL AND FILTERED/NOT FILTERED ARTIFICIAL OUTPUT ###
        plot_box([real, data_art_fil_out, artific],['Real output', 'Artificial\n filtered output', 'Artificial\n not filtered output'],'Boxplot real and filtered/not filtered artificial output pdf','y data','Pdf values',f'{results_path}/GMM/real_f_nf_artific_output.png') 

        #Saves
        #Write a .txt file to the specified path and writes information
        with open(f'{results_path}/GMM/artificial_info.txt', "w") as f:
            print(f"Info artificial samples:\n\n\tMEAN:\t{torch.mean(new_samples)}\n\n\tSTD:\t{torch.std(new_samples)}\n\n\tMEDIAN:\t{torch.median(new_samples)}\n\n\tMIN:\t{torch.min(new_samples)}\n\n\tMAX:\t{torch.max(new_samples)}\
                \nInfo artificial outputs:\n\n\tMEAN:\t{torch.mean(outputs_artificial)}\n\n\tSTD:\t{torch.std(outputs_artificial)}\n\n\tMEDIAN:\t{torch.median(outputs_artificial)}\n\n\tMIN:\t{torch.min(outputs_artificial)}\n\n\tMAX:\t{torch.max(outputs_artificial)}\
                \nInfo artificial filtered samples:\n\n\tMEAN:\t{torch.mean(artific_input_fil_tens)}\n\n\tSTD:\t{torch.std(artific_input_fil_tens)}\n\n\tMEDIAN:\t{torch.median(artific_input_fil_tens)}\n\n\tMIN:\t{torch.min(artific_input_fil_tens)}\n\n\tMAX:\t{torch.max(artific_input_fil_tens)}\
                \nInfo artificial filtered outputs:\n\n\tMEAN:\t{torch.mean(artific_output_fil)}\n\n\tSTD:\t{torch.std(artific_output_fil)}\n\n\tMEDIAN:\t{torch.median(artific_output_fil)}\n\n\tMIN:\t{torch.min(artific_output_fil)}\n\n\tMAX:\t{torch.max(artific_output_fil)}", file=f)
            
       
