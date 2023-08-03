import numpy as np
import torch


def generate_gaussian_mixture_samples(mixing_parameters, means, std_deviations, n_samples, min_val=-0.9617483019828796, max_val=3.4695065021514893):
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
        
        #Check if the sample is within the desired range
        if min_val is not None and max_val is not None:
            if min_val <= sample <= max_val:
                samples.append(sample)
        else:
            samples.append(sample)


    # Convert samples list to PyTorch tensor
    samples = np.array(samples).reshape(n_samples, 1)
    samples_tensor = torch.tensor(samples)

    return samples_tensor


