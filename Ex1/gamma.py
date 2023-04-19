"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import cv2
import numpy as np
from ex1_utils import LOAD_GRAY_SCALE
from ex1_utils import LOAD_RGB

title_window = 'Gamma Correction'


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img

    # If the representation mode is set to 1, convert the image to grayscale
    # Convert to grayscale if requested
    if rep == 1:
        img = img = cv2.imread(img_path, 2)

    # Convert to RGB if requested
    elif rep == 2:
        img = img = cv2.imread(img_path, 1)

    else:
        raise Exception("rep must be 0 or 1")

    # Create a named window for displaying the corrected image
    cv2.namedWindow('Gamma Correction')

    # Create a trackbar for adjusting the gamma value
    # The trackbar ranges from 0 to 200 with a resolution of 1
    # This corresponds to a gamma range of 0.01 to 2.0 with a resolution of 0.01
    cv2.createTrackbar('Gamma', 'Gamma Correction', 100, 200, apply_gamma_correction)
    apply_gamma_correction(100)
    cv2.waitKey()


# Define a function that applies gamma correction to an image
def apply_gamma_correction(gamma):

    # Read the current position of the trackbar and convert it to a gamma value
    gamma = gamma / 100.0
    # Make sure gamma is not zero
    gamma = max(gamma, 0.01)

    # Compute the inverse of gamma
    inv_gamma = 1.0 / gamma

    # Create a lookup table that maps input pixel values to output pixel values
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

    # Apply the lookup table to the input image
    lut = cv2.LUT(img, table)

    cv2.imshow('Gamma Correction', lut)


def main():
    gammaDisplay('beach.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
