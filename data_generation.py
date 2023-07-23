import numpy as np
import torch

def generate_gaussian_mixture_samples(mixing_parameters, means, std_deviations, n_samples):
    #Normalize mixing parameters to ensure they sum up to 1
    mixing_parameters = np.array(mixing_parameters) / sum(mixing_parameters)
    
    #Lista samples
    samples = []
    
    #Loop for generating samples
    for _ in range(n_samples):
        #Generate a random number between 0 and 1
        random_number = np.random.random()
        
        #Select the Gaussian component based on the mixing parameters and random number
        component_index = np.argmax(random_number <= np.cumsum(mixing_parameters))
        
        #Generate a random variable from the selected Gaussian component
        sample = np.random.normal(loc=means[component_index], scale=std_deviations[component_index])
        samples.append(sample)
    
    # Convert samples list to PyTorch tensor
    samples = np.array(samples).reshape(n_samples, 1)
    samples_tensor = torch.tensor(samples)

    return samples_tensor



