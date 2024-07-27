import numpy as np


# Generate lognormal samples directly
def lognormal_samples(mu=120, sigma=0.4, upper=.4):
    """
    lognormal distribution

    :param upper:
    :param mu: Mean of the underlying normal distribution
    :param sigma: Standard deviation of the underlying normal distribution
    :return: Function to generate lognormal samples
    """

    def lognormal(size):
        """
        Generate lognormal samples

        :param size: Number of samples to generate
        :return: Array of lognormal distributed samples
        """
        data = np.random.lognormal(mu, sigma, size)
        return data / np.max(data) * upper

    return lognormal


# Generate Gaussian (normal) samples
def gaussian_samples(mu=120, sigma=0.4, upper=.4):
    """
    Gaussian distribution

    :param upper:
    :param mu: Mean of the Gaussian distribution
    :param sigma: Standard deviation of the Gaussian distribution
    :return: Function to generate Gaussian samples
    """

    def gaussian(size):
        """
        Generate Gaussian samples

        :param size: Number of samples to generate
        :return: Array of Gaussian distributed samples
        """
        data = np.random.normal(mu, sigma, size)
        return data / np.max(data) * upper

    return gaussian


# Function to get the appropriate distribution function
def get_distribution(disttype='lognormal', **kwargs):
    """
    Get the distribution function based on disttype

    :param disttype: Type of distribution ('lognormal' or 'gaussian')
    :param kwargs: Additional parameters for the distribution function.
    :return: Distribution function
    """
    distributions = {
        'lognormal': lognormal_samples,
        'gaussian': gaussian_samples,
        # Add other distributions here
    }

    if disttype not in distributions:
        raise ValueError(f"Unknown distribution type: {disttype}")

    return distributions[disttype](**kwargs)
