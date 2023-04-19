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


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    # Load the input image
    img = cv2.imread(img_path)

    # If the representation mode is set to 1, convert the image to grayscale
    # Convert to grayscale if requested
    if rep == 1:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to RGB if requested
    elif rep == 2:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        raise Exception("rep must be 0 or 1")

    # Create a named window for displaying the corrected image
    cv2.namedWindow('Gamma Correction')
    # Create a trackbar for adjusting the gamma value
    # The trackbar ranges from 0 to 200 with a resolution of 1
    # This corresponds to a gamma range of 0.01 to 2.0 with a resolution of 0.01
    cv2.createTrackbar('Gamma', 'Gamma Correction', 1, 200, lambda x: None)

    # Enter a loop that reads the current position of the trackbar,
    # applies gamma correction to the input image, and displays the corrected image
    while True:

        # Read the current position of the trackbar and convert it to a gamma value
        gamma = cv2.getTrackbarPos('Gamma', 'Gamma Correction') / 100.0

        # Apply gamma correction to the input image using the apply_gamma_correction function
        corrected_img = apply_gamma_correction(img, gamma)

        # Display the corrected image in the Gamma Correction window
        cv2.imshow('Gamma Correction', corrected_img)

        # Wait for a key press and check if it is the Escape key
        key = cv2.waitKey(10)
        if key == 27:  # Escape key
            break

    # Destroy the Gamma Correction window
    cv2.destroyAllWindows()


# Define a function that applies gamma correction to an image
def apply_gamma_correction(img, gamma):

    # Make sure gamma is not zero
    gamma = max(gamma, 0.01)

    # Compute the inverse of gamma
    inv_gamma = 1.0 / gamma

    # Create a lookup table that maps input pixel values to output pixel values
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

    # Apply the lookup table to the input image
    return cv2.LUT(img, table)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
