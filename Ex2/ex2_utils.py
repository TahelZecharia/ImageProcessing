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

    # flipped_kernel = kernel

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

    img = img.squeeze()

    if img.ndim > 2:
        raise ValueError("The image is not grayscale")

    image_height, image_width = img.shape

    max_radius = min((min(image_height, image_width) // 2), max_radius)

    # Get each pixels gradients direction
    i_y = cv2.Sobel(img, -1, 0, 1, ksize=3)
    i_x = cv2.Sobel(img, -1, 1, 0, ksize=3)
    ori = np.arctan2(i_y, i_x)

    # Get Edges using Canny Edge detector
    bw = cv2.Canny((img * 255).astype(np.uint8), 550, 100)
    radius_diff = max_radius - min_radius
    circle_hist = np.zeros((image_height, image_width, radius_diff))

    # Get the coordinates only for the edges
    ys, xs = np.where(bw)

    # Calculate the sin/cos for each edge pixel
    sins = np.sin(ori[ys, xs])
    coss = np.cos(ori[ys, xs])
    r_range = np.arange(min_radius, max_radius)

    for iy, ix, ss, cs in zip(ys, xs, sins, coss):

        grad_sin = (r_range * ss).astype(int)
        grad_cos = (r_range * cs).astype(int)

        xc_1 = ix + grad_cos
        yc_1 = iy + grad_sin

        xc_2 = ix - grad_cos
        yc_2 = iy - grad_sin

        # Check where are the centers that are in the image
        r_idx1 = np.logical_and(yc_1 > 0, xc_1 > 0)
        r_idx1 = np.logical_and(r_idx1, np.logical_and(yc_1 < image_height, xc_1 < image_width))

        # Check where are the centers that are in the image (Opposite direction)
        r_idx2 = np.logical_and(yc_2 > 0, xc_2 > 0)
        r_idx2 = np.logical_and(r_idx2, np.logical_and(yc_2 < image_height, xc_2 < image_width))

        # Add circles to the circle histogram
        circle_hist[yc_1[r_idx1], xc_1[r_idx1], r_idx1] += 1
        circle_hist[yc_2[r_idx2], xc_2[r_idx2], r_idx2] += 1

    # Find all the circles centers
    y, x, r = np.where(circle_hist > 11)
    circles = np.array([x, y, r + min_radius, circle_hist[y, x, r]]).T

    # Perform NMS
    circles = nms(circles, min_radius // 2)
    return circles


def nms(xyr: np.ndarray, radius: int) -> list:
    """
    Performs Non Maximum Suppression in order to remove circles that are close
    to each other to get a "clean" output.
    :param xyr:
    :param radius:
    :return:
    """

    ret_xyr = []

    while len(xyr) > 0:
        # Choose most ranked circle (MRC)
        curr_arg = xyr[:, -1].argmax()
        curr = xyr[curr_arg, :]
        ret_xyr.append(curr)
        xyr = np.delete(xyr, curr_arg, axis=0)

        # Find MRC close neighbors
        dists = np.sqrt(np.square(xyr[:, :2] - curr[:2]).sum(axis=1)) < radius
        idx_to_delete = np.where(dists)

        # Delete MRCs neighbors
        xyr = np.delete(xyr, idx_to_delete, axis=0)

    return ret_xyr


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    # OpenCV Bilateral Filter
    CV2_bilateral = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space, borderType=cv2.BORDER_REPLICATE)

    # My Bilateral Filter Implementation
    image_height, image_width = in_image.shape[:2]
    my_bilateral = np.zeros((image_height, image_width))

    # Padding size
    padding = k_size // 2

    # Pad the image using reflection
    padded_image = np.pad(in_image, padding, mode='reflect')

    # Gaussian kernel for space domain (for Wc)
    space_kernel = cv2.getGaussianKernel(k_size, sigma_space)
    space_kernel = space_kernel.dot(space_kernel.T)

    # Apply bilateral filtering
    for i in range(image_height):

        for j in range(image_width):

            # Extract the neighborhood from the padded image
            neighborhood = padded_image[i : i + k_size, j : j + k_size]

            # Gaussian kernel for color domain (for Ws)
            color_kernel = np.exp(- ( (neighborhood - in_image[i, j]) ** 2) / (2 * sigma_color))

            # Combine space and color kernels
            combined_kernel = space_kernel * color_kernel

            # Normalize the kernel
            combined_kernel /= np.sum(combined_kernel)

            # Apply the kernel to the neighborhood
            kernel = combined_kernel * neighborhood

            # Calculation of the sum of kernel values
            my_bilateral[i, j] = np.sum(kernel)

    return CV2_bilateral, my_bilateral


if __name__ == '__main__':

    def sharp1(img: np.ndarray):

        # Creat Laplacian matrix
        laplacian_matrix = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        img = cv2.imread('VDWIW.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=.25, fy=.25)
        sharp_img = cv2.filter2D(img, -1, laplacian_matrix)

        f, ax = plt.subplots(1, 3)
        f.suptitle("Sharp")
        ax[0].set_title("Ori")
        ax[1].set_title("Laplacian")
        ax[2].set_title("Sharp")
        ax[0].imshow(img)
        ax[1].imshow(laplacian_matrix)
        ax[2].imshow(sharp_img)
        plt.show()


    def sharp2(img: np.ndarray):

        # Creat Laplacian matrix
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])

        img = cv2.imread('VDWIW.png', cv2.IMREAD_GRAYSCALE) / 255
        img = cv2.resize(img, (0, 0), fx=.25, fy=.25)
        laplacian_matrix = cv2.filter2D(img, -1, laplacian)
        sharp_img = img - laplacian_matrix

        print(img)
        print(laplacian_matrix)
        print(sharp_img)

        sharp_img[sharp_img < 0.0] = 0.0
        sharp_img[sharp_img > 1.0] = 1.0

        f, ax = plt.subplots(1, 3)
        f.suptitle("Sharp")
        ax[0].set_title("Ori")
        ax[1].set_title("Lap")
        ax[2].set_title("Sharp")
        ax[0].imshow(img)
        ax[1].imshow(laplacian_matrix)
        ax[2].imshow(sharp_img)
        plt.show()

        # cv2.imshow('img', img)
        # cv2.imshow('edge_pad', laplacian_matrix )
        # cv2.imshow('out_img', sharp_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    def sharp1VSsharp2():

        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        laplacian_matrix_1 = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        laplacian_matrix_2 = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])

        s1 = conv2D(matrix, laplacian_matrix_1)
        s2 = conv2D(matrix, laplacian_matrix_2)
        print(s2)
        s2 = matrix - s2

        print(s1)
        print(s2)


    sharp1VSsharp2()
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

    kernel = np.array([1, 2, 3])
    inv_k = kernel[::-1]
    print("inv_k: ", inv_k)