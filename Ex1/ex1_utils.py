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
import sys
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import mean_squared_error

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211485461


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Read the image using OpenCV
    image = cv2.imread(filename)

    # Convert to grayscale if requested
    if representation == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to RGB if requested
    elif representation == 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    else:
        raise Exception("rep must be 0 or 1")

    image = np.float64(image) / 255.0  # Normalize intensities to [0, 1]

    return image


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # Load image using imReadAndConvert function
    image = imReadAndConvert(filename, representation)

    # Display the image using plt.imshow
    plt.imshow(image, cmap='gray')  # if img is gray, plot it gray

    # Set the title of the figure window
    if representation == 1:
        plt.title('Grayscale Image')
    elif representation == 2:
        plt.title('RGB Image')

    # Show the image
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # Conversion matrix from RGB to YIQ
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.311]])

    # Reshape the input image to a 3D array of pixels (height, width, channels)
    imgRGB = imgRGB.astype(np.float64)  # Convert to float for accurate calculations
    imYIQ = np.dot(imgRGB, conversion_matrix.T)  # Perform matrix multiplication to convert RGB to YIQ

    return imYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # Conversion matrix from YIQ to RGB
    conversion_matrix = np.array([[1.0, 0.9557, 0.6199],
                           [1.0, -0.2716, -0.6469],
                           [1.0, -1.1082, 1.7051]])

    # Reshape the input image to a 3D array of pixels (height, width, channels)
    imgYIQ = imgYIQ.astype(np.float64)  # Convert to float for accurate calculations
    # Perform matrix multiplication to convert YIQ to RGB
    imRGB = np.dot(imgYIQ, conversion_matrix.T)

    # # Clip RGB values to be within [0, 1] range (values smaller than 0 become 0, and values larger than 1 become 1.)
    # imRGB = np.clip(imRGB, 0, 1)

    return imRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """

    # Convert RGB to YIQ
    if imgOrig.ndim == 3:
        imgYIQ = transformRGB2YIQ(imgOrig)
        y = imgYIQ[:, :, 0]
    else:
        y = imgOrig

    # Convert the input image to 8-bit grayscale
    max = np.amax(y)
    min = np.amin(y)
    y = (255 * ((y - min) / (max - min))).astype(np.uint8)

    # Compute the histogram of the grayscale image
    histOrg, bins = np.histogram(y, 256, [0, 255])

    # Compute the cumulative sum
    cumSum = histOrg.cumsum()

    # Compute the normalized cumulative sum (divide by #pixeles)
    cdf = cumSum / cumSum.max()

    # Create the lookup table (multiply by the maximum gray level (255) and round down)
    lut = (np.floor(255 * cdf)).astype(np.uint8)

    # Apply the lookup table to the grayscale image
    yEq = lut[y]

    # Compute the histogram of the equalized image
    histEq, bins = np.histogram(yEq, 256, [0, 255])

    # Normalize intensities to [0, 1]
    yEq = yEq.astype(np.float64) / 255.0

    # Convert back to RGB if necessary
    if imgOrig.ndim == 3:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgYIQ[:, :, 0] = yEq
        imEq = transformYIQ2RGB(imgYIQ)
    else:
        imEq = yEq

    return imEq, histOrg, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param y: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    # Convert RGB to YIQ
    if imOrig.ndim == 3:
        imgYIQ = transformRGB2YIQ(imOrig)
        y = imgYIQ[:, :, 0]
    else:
        y = imOrig

    # Convert the input image to 8-bit grayscale
    y = y * 255

    # Initialize segment division with approximately equal number of pixels
    histOrg, bins = np.histogram(y, bins=256, range=[0, 255])
    cumSum = np.cumsum(histOrg)
    z = np.zeros(nQuant + 1).astype(np.uint8)
    z[nQuant] = 255
    segSize = y.size // nQuant
    k = 1
    for i in range(255):
        if k == nQuant:
            break
        segIndex = k * segSize
        if cumSum[i] <= segIndex <= cumSum[i + 1]:
            z[k] = i
            k += 1

    # Initialize quantized image and error list
    qImage = []
    error = []

    # Iterations
    for iter in range(nIter):

        # Find quantized values for each segment by weighted average
        q = [int(np.average(range(z[i], z[i + 1]), weights=histOrg[z[i]:z[i + 1]])) for i in range(nQuant)]

        # Update segment division
        for i in range(1, nQuant):
            z[i] = int((q[i - 1] + q[i]) / 2)

        # Quantize the image. Fill tmpImg by curr quantization colors
        tmpImg = np.zeros_like(y)
        for i in range(nQuant):
            tmpImg[(z[i] <= y) & (y <= z[i + 1])] = q[i]

        # Calculate total intensities error (MSE)
        MSE = ((y - tmpImg) ** 2).mean()
        error.append(MSE)

        # Normalize intensities to [0, 1]
        tmpImg = tmpImg.astype(np.float64) / 255.0

        # Convert back to RGB if necessary
        if imOrig.ndim == 3:
            imgYIQ = transformRGB2YIQ(imOrig)
            imgYIQ[:, :, 0] = tmpImg
            qImage.append(transformYIQ2RGB(imgYIQ))
        else:
            qImage.append(tmpImg)

    # Convert error to list
    error = list(error)

    return qImage, error


def isGray(imgOrig: np.ndarray) -> (np.ndarray, bool):
    """
    if RGB -> convert to YIQ -> take Y
    elif GRAY -> Y = imgOrig
    :param imgOrig:
    :return: y: graySale img, gray: True ig imgOrig was gray
    """
    gray = True
    if len(imgOrig.shape) == 3:
        imgYIQ = transformRGB2YIQ(imgOrig)
        gray = False
        y = imgYIQ[:, :, 0]
    else:
        y = imgOrig
    return y, gray

if __name__ == '__main__':

    print(imReadAndConvert('beach.jpg', 1))
    print(imDisplay('beach.jpg', 1))

    print(imReadAndConvert('beach.jpg', 2))
    print(imDisplay('beach.jpg', 2))
