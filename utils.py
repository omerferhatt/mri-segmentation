import numpy as np


def squared_gaussian_kernel(r2: int, sigma: float) -> 'float':
    """Gaussian function taking the squared radius

    :param r2: Squared radius of kernel
    :param sigma: Sigma value for gaussian kernel
    :return: Gaussian function implementation
    """
    return (np.exp(-0.5 * r2 / sigma ** 2) * 3).astype(int) * 1.0 / 3.0


def gaussian_kernel(r: int, sigma: float) -> 'float':
    """Simple Gaussian function implementation

    :param r: Radius of kernel
    :param sigma: Sigma value for gaussian kernel
    :return: Gaussian function implementation
    """
    euclidean_distance = np.sqrt((r ** 2).sum(axis=1))
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (euclidean_distance / sigma) ** 2)


def multivariate_gaussian_kernel(distances, bandwidths: 'list'):
    """Kernel density estimation is a nonparametric technique for density estimation
     i.e., estimation of probability density functions, which is one of the fundamental questions in statistics.

    :param distances:
    :param bandwidths: Dimension list of gaussian kernel
    :return:
    """
    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * np.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return val


def euclidean_dist(point_p: 'tuple', point_q: 'tuple') -> 'float':
    """ Calculating "ordinary" straight-line distance between two points in Euclidean space

    The Euclidean distance between points p and q is the length of the line segment connecting them

    :param point_p: Reference point p of Euclidean line
    :param point_q: Reference point q of Euclidean line

    :return: Euclidean distance between two point
    """
    try:
        if len(point_p) != len(point_q):
            raise TypeError

        total = float(0)
        for dimension in range(0, len(point_p)):
            total += (point_p[dimension] - point_q[dimension])**2
        return np.sqrt(total)

    except TypeError:
        print("Expected point dimensionality to match")


def calc_snr(img, denoised):
    m = np.mean(img)
    sd = np.std(denoised)

    snr = m / sd
    return snr