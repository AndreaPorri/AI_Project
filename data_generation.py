'''

                        ############################################################

                                           GMM GENERATOR

                        ############################################################


The provided code implements a function named 'generate_gaussian_mixture_samples' that generates a specified number of samples (YAML file)
from a mixture of Gaussian distributions.

Each Gaussian distribution has its own set of parameters: 
    - Mixing parameters (generated randomly and sum up to 1)
    - Means (generated randomly within an empirical range)
    - Standard deviations (generated randomly within an empirical range)
     
The function takes these three parameters, the number of samples to generate, and optionally, the lower and upper limits
of the range in which the samples should be generated.

The process of generating the samples is as follows:
- Initialize an empty list called samples to store the generated samples.
- Enter a loop until the desired number of samples is reached.
- Generate a random number between 0 and 1.
- Based on the provided mixing parameters, select one of the Gaussian distributions by comparing the 
  random number with the cumulative sum of the mixing parameters.
- Draw a random sample from the selected Gaussian distribution.
- If the lower and upper limits (min_val and max_val) are provided, check whether the generated sample falls
  within those limits. Only in this case, the sample is added to the samples list.

This code can be used to generate random artificial samples from a mixture of Gaussian distributions, which is useful 
in various data analysis and machine learning contexts. In that case it will be used to generate new sample, so new 
monodimensional artificial input that will be passed to the MLP and they have to behave as the real input. Also the std and mean
ranges are decided looking the reale input.

This code can be used to generate random artificial samples from a mixture of Gaussian distributions, which is 
useful in various data analysis and machine learning contexts. In this case, it will be used to generate new 
samples as monodimensional artificial inputs. These artificial inputs are then passed to the pretrained, on the real date, MLP 
(Multi-Layer Perceptron), and then they are expected to behave similarly to the real output(dame pdy more or less).
The ranges for the mean and standard deviation are determined by observing the real input data pdf.

'''


#Import needed libraries, classes and functions
import numpy as np
import torch

### GMM GENERATOR ###
def generate_gaussian_mixture_samples(mixing_parameters, means, std_deviations, n_samples, min_val=None, max_val=None):
    #Lista samples
    samples = []
    
    #Loop for generating samples
    while len(samples) < n_samples:
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
    samples = np.array(samples).reshape(-1, 1)
    samples_tensor = torch.tensor(samples)

    return samples_tensor


