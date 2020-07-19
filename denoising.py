import numpy as np
import matplotlib.pyplot as plt

from utils import squared_gaussian_kernel


def bilateral_filter(img_in: 'np.ndarray', sigma_s: 'float', sigma_v: 'float', reg_constant=1e-8) -> 'np.ndarray':
    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image.
    If padding is desired, `img_in` should be padded prior to calling

    :param img_in: Monochrome input image
    :param sigma_s: Spatial gaussian std. dev.
    :param sigma_v: Value gaussian std. dev.
    :param reg_constant: Optional regularization constant for pathalogical cases

    :raises ValueError: Whenever `img_in` is not a 2D float32 valued `np.ndarray`

    :returns: Output bilateral-filtered image

    """

    # check the input
    if not isinstance(img_in, np.ndarray) or img_in.dtype != 'float32' or img_in.ndim != 2:
        raise ValueError('Expected a 2D numpy.ndarray with float32 elements')

    # define the window width to be the 3 time the spatial std. dev. to
    # be sure that most of the spatial kernel is actually captured
    win_width = int(3 * sigma_s + 1)

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    weighted_sum = np.ones(img_in.shape) * reg_constant
    result = img_in * reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and
    # the unnormalized result image
    for shift_x in range(-win_width, win_width + 1):
        for shift_y in range(-win_width, win_width + 1):
            # compute the spatial weight
            w = squared_gaussian_kernel(shift_x ** 2 + shift_y ** 2, sigma_s)

            # shift by the offsets
            off = np.roll(img_in, [shift_y, shift_x], axis=[0, 1])

            # compute the value weight
            tw = w * squared_gaussian_kernel((off - img_in) ** 2, sigma_v)

            # accumulate the results
            result += off * tw
            weighted_sum += tw

    # normalize the result and return
    return np.asarray((result / weighted_sum) * 255, dtype=np.uint8)


def plot_filtering(img: 'np.ndarray', denoised: 'np.ndarray', title: 'str', fname: 'str'):
    """ Specific comparison plot function

    :param img: Noisy image, raw
    :param denoised: Denoised image after filtering
    :param title: Title of plot
    :param fname: Filename to saving in disk

    """
    # Subtracting image from denoised image to get noise profile
    noise = img - (denoised / 255)

    fig, axs = plt.subplots(nrows=1, ncols=3)

    axs[0].imshow(img)
    axs[0].set_axis_off()
    axs[0].set_title('Normal')

    axs[1].imshow(noise)
    axs[1].set_axis_off()
    axs[1].set_title('Noise')

    axs[2].imshow(denoised)
    axs[2].set_axis_off()
    axs[2].set_title('Denoised')

    fig.suptitle(title)
    plt.savefig(f'{fname}.png')
    plt.show()
