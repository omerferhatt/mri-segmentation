import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


def otsu(gray: 'np.ndarray') -> 'np.ndarray':
    """ Performs automatic image thresholding

    :param gray: Grayscale image
    :return: Threshold applied image
    """
    shape = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0 / shape
    hist, bins = np.histogram(gray, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity = np.arange(256)
    # Goes from 5 to 245 uint8 range
    for t in bins[5:-10]:
        # The class probability is computed from the bins of the histogram
        pcb = np.sum(hist[:t])
        pcf = np.sum(hist[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight
        # For 2 classes, minimizing the intra-class variance is equivalent to maximizing inter-class variance:
        mub = np.sum(intensity[:t]*hist[:t]) / float(pcb)
        muf = np.sum(intensity[t:]*hist[t:]) / float(pcf)
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value

    final_img = gray.copy()
    final_img[gray < final_thresh+20] = 255
    final_img[gray > final_thresh+50] = 0

    return final_img


def plot_segmentation(binary, denoised, title, fname):
    # Masking both upper and under segmentations
    binary[binary == 255] = 0
    # Creating RGB images in order to show segmentation in red
    gray_img = np.stack([denoised, denoised, denoised], axis=-1)
    mask = np.stack([binary, np.zeros_like(binary), np.zeros_like(binary)], axis=-1)
    blend = np.array((gray_img * 0.5) + (mask * 0.5), dtype=np.uint8)
    # Plotting and saving figures
    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(denoised)
    axs[0].set_axis_off()
    axs[0].set_title('Denoised')

    axs[1].imshow(binary)
    axs[1].set_axis_off()
    axs[1].set_title('Mask')

    axs[2].imshow(blend)
    axs[2].set_axis_off()
    axs[2].set_title('Mask Blend')

    fig.suptitle(title)
    plt.savefig(f'{fname}.png')
    plt.show()
