import argparse
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from skimage.io import imread
import denoising
import segmentation
import utils

if __name__ == "__main__":
    # Command line setup with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action="store", required=True)
    args = parser.parse_args()

    # Loading image and min-max scale between 0-1
    img = imread(args.path)
    img_filt = np.asarray(img, dtype=np.float32) / 255

    # Part 1 - Bilateral Filtering
    denoised = denoising.bilateral_filter(img_filt, 15.0, 0.1)
    denoising.plot_filtering(img_filt, denoised,
                             title=f'Bilateral Filtering, SNR={utils.calc_snr(img, denoised):0.4f}',
                             fname='bilateral_filtering')
    # Signal to Noise ratio
    print(f"SNR Ratio: {utils.calc_snr(img, denoised)}")

    # Part 2 - Otsu's Method Segmentation
    img_seg = np.asarray(denoised, dtype=int)
    mask = segmentation.otsu(np.asarray(img_seg, dtype=int))
    segmentation.plot_segmentation(mask, img_seg,
                                   title="Otsu's Method Segmentation",
                                   fname='otsu_segment')
