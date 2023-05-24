import math
import numpy as np
import cv2
import scipy.ndimage as nd
from matplotlib import pyplot as plt


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 211485461


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    # Checking that the array is 1-D:
    if len(in_signal.shape) > 1:
        if in_signal.shape[1] > 1:
            raise ValueError("Input Signal is not a 1D array")
        else:
            in_signal = in_signal.reshape(in_signal.shape[0])

    signal_len = len(in_signal)
    kernel_len = len(k_size)

    output_len = signal_len + kernel_len - 1
    # Initialize an array of zeros of size output_len
    output = np.zeros(output_len)
    # Pads the edges of the array with zeros
    padded_signal = np.pad(in_signal, kernel_len - 1, 'constant')

    # Flip the kernel
    flipped_kernel = np.flip(k_size)

    # Perform 1D convolution
    for i in range(output_len):

        output[i] = (padded_signal[i : i + kernel_len] * flipped_kernel).sum()

    return output


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    image_height, image_width = in_image.shape
    kernel_height, kernel_width = kernel.shape
    half_kernel_height = kernel_height // 2
    half_kernel_width = kernel_width // 2

    # The output image will be the same size as the original image
    output = np.zeros((image_height, image_width))

    # Pads the edges of the array with reflection
    padded_image = np.pad(in_image, ((half_kernel_height, half_kernel_height), (half_kernel_width, half_kernel_width)), 'reflect')

    # Flip the kernel
    flipped_kernel = np.flip(np.flip(kernel, axis=0), axis=1)

    # Perform 2D convolution
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = (padded_image[i:i + kernel_height, j:j + kernel_width] * flipped_kernel).sum()

    return output


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    # Define the derivative kernels
    kernel_x = np.array([1, 0, -1]).reshape(1, 3)
    kernel_y = np.array([1, 0, -1]).reshape(3, 1)

    # Compute the x and y derivatives using conv2D
    dx = conv2D(in_image, kernel_x)
    dy = conv2D(in_image, kernel_y)

    # Compute the magnitude and direction of the gradient
    direction = np.arctan2(dy, dx)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    return direction, magnitude


def createGaussianKernel(k_size: int) -> np.ndarray:

    """
    The function generates a Gaussian kernel of a specified size.
    :param k_size:
    :return:
    """

    # Check if k_size is an odd number
    if k_size % 2 == 0:
        raise ValueError("Kernel size should be an odd number.")

    # Generate the 1D Gaussian kernel using binomial coefficients
    # Convolve [1,1]*[1,1] until get k_size
    kernel_1D = np.array([1, 1])
    kernel = np.array([1, 1])
    for _ in range(k_size - 2):
        kernel_1D = conv1D(kernel_1D, kernel)

    # Make the kernel 2D
    kernel_1D = kernel_1D.reshape(k_size, 1)
    kernel_2D = kernel_1D.dot(kernel_1D.T)

    # Normalize the 2D kernel
    kernel_2D = kernel_2D / np.sum(kernel_2D)

    return kernel_2D


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    # Create 2D kernel
    kernel_2D = createGaussianKernel(k_size)

    # Perform 1D convolution twice in both row and column directions
    blurred_image = conv2D(in_image, kernel_2D)

    return blurred_image


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    # Check if k_size is an odd number
    if k_size % 2 == 0:
        raise ValueError("Kernel size should be an odd number.")

    # Get the 2D Gaussian kernel using OpenCV's getGaussianKernel
    kernel_1D = cv2.getGaussianKernel(k_size, -1)

    # Make the kernel 2D
    kernel_2D = kernel_1D.dot(kernel_1D.T)

    # Perform 2D convolution using OpenCV's filter2D
    blurred_image = cv2.filter2D(in_image, -1, kernel_2D)

    return blurred_image


def findZeroCrossing(lap_img: np.ndarray) -> np.ndarray:
    """
    The function takes a Laplacian image lap_img as input
    and returns a matrix indicating the zero-crossing points.
    :param lap_img:
    :return:
    """

    # The resulting image minLoG represents the minimum value in the local neighborhood
    # of each pixel in the Laplacian image.
    minLoG = cv2.morphologyEx(lap_img, cv2.MORPH_ERODE, np.ones((3, 3)))
    # The resulting image maxLoG represents the maximum value in the local neighborhood
    # of each pixel in the Laplacian image
    maxLoG = cv2.morphologyEx(lap_img, cv2.MORPH_DILATE, np.ones((3, 3)))

    # logical operations to identify the zero-crossing points. It checks for three conditions.
    zeroCross = np.logical_or(np.logical_and(lap_img > 0, minLoG < 0),
                              np.logical_and(lap_img < 0, maxLoG > 0),
                              np.logical_and(lap_img == 0, maxLoG > 0, minLoG < 0))

    return zeroCross


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:

    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    # Creat Laplacian matrix
    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    # Apply Laplacian filter to the image
    my_laplacian = cv2.filter2D(img, -1, laplacian)

    # Find the zero-crossing points
    return findZeroCrossing(my_laplacian)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:

    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    # Creat Laplacian matrix
    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    # Create Gaussian kernel
    my_gauss = createGaussianKernel(101)
    # Convolve the input image with the Gaussian kernel using cv2.filter2D
    my_lap = cv2.filter2D(img, -1, my_gauss)
    # Convolve my_lap with the Laplacian matrix using cv2.filter2D
    my_lap = cv2.filter2D(my_lap, -1, laplacian)

    # Find the zero-crossing points
    return findZeroCrossing(my_lap)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return

if __name__ == '__main__':

    # Creat Laplacian matrix
    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])


    img = cv2.imread('VDWIW.png', cv2.IMREAD_GRAYSCALE) / 255
    img = cv2.resize(img, (0, 0), fx=.25, fy=.25)
    lap_mat = cv2.filter2D(img, -1, laplacian)

    sharp_mat = img - lap_mat

    print(img)
    print(lap_mat)
    print(sharp_mat)


    sharp_mat[sharp_mat < 0.0] = 0.0
    sharp_mat[sharp_mat > 1.0] = 1.0
    #
    # res = (res * 255).astype(np.uint8)

    # cv2.imshow('img', img)
    # cv2.imshow('edge_pad', lap_mat )
    # cv2.imshow('out_img', sharp_mat)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    f, ax = plt.subplots(1, 3)
    f.suptitle("Sharp")
    ax[0].set_title("Ori")
    ax[1].set_title("Lap")
    ax[2].set_title("Sharp")
    ax[0].imshow(img)
    ax[1].imshow(lap_mat)
    ax[2].imshow(sharp_mat)
    plt.show()

    matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
    m = np.array([[1,2],[1,2]])
    print(matrix)
    matrix_flipped = np.flip(np.flip(matrix, 0), 1)
    print(matrix_flipped)
    # print(conv2D(matrix, m))

    mat = [1,2,3,4]
    print(np.flip(mat))

    padded_mat = np.pad(mat, 2, 'constant')
    print("padded_mat: " ,padded_mat )
    padded_matrix = np.pad(matrix, ((2, 1), (3,4)), 'reflect')
    print("padded_matrix: ", padded_matrix)