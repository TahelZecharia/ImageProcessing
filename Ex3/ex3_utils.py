import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211485461

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """

    points = []
    flows = []  # uv

    height, width = im1.shape
    window = win_size // 2

    # Compute gradients Ix and Iy
    kernel = np.array([[-1, 0, 1]])
    Ix = cv2.filter2D(im2, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, kernel.T, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    for y in range(window, height - window, step_size):

        for x in range(window, width - window, step_size):

            # Compute the sums of derivatives products within the window
            Ix_window = Ix[y - window : y + window + 1, x - window : x + window + 1].flatten()
            Iy_window = Iy[y - window : y + window + 1, x - window : x + window + 1].flatten()
            It_window = It[y - window : y + window + 1, x - window : x + window + 1].flatten()

            ATA = [[(Ix_window * Ix_window).sum(), (Ix_window * Iy_window).sum()],
                    [(Ix_window * Iy_window).sum(), (Iy_window * Iy_window).sum()]]

            lambada = np.linalg.eigvals(ATA)
            lambada1 = np.max(lambada)
            lambada2 = np.min(lambada)

            if lambada1 >= lambada2 > 1 and (lambada1 / lambada2) < 100:

                ATb = [[-(Ix_window * It_window).sum()], [-(Iy_window * It_window).sum()]]
                flow = (np.dot(np.linalg.inv(ATA), ATb)).reshape(2)
                points.append([x, y])
                flows.append(flow)

            else:
                points.append([x, y])
                flows.append(np.array([0., 0.]))

    return np.array(points), np.array(flows)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    Ui = Ui + 2 ∗ Ui−1, Vi = Vi + 2 ∗ Vi−1
    """

    # Create image pyramids
    pyramid1 = gaussianPyr(img1, k)
    pyramid2 = gaussianPyr(img2, k)

    # Initialize flow at the lowest pyramid level
    height, width = pyramid1[-1].shape
    print(pyramid1[-1].shape)
    _, flow = opticalFlow(pyramid1[- 1], pyramid2[- 1], stepSize, winSize)
    flow = flow.reshape((((height -1) // stepSize) + 1, ((width -1) // stepSize) + 1, 2))

    for i in range(k - 2, -1, -1):

        current_img1 = pyramid1[i]
        current_img2 = pyramid2[i]
        height, width = current_img1.shape

        expand = np.zeros((((height -1) // stepSize) + 1, ((width -1) // stepSize) + 1, 2))
        expand[::2, ::2] = flow * 2
        flow = expand

        _, current_flow = opticalFlow(current_img1, current_img2, stepSize, winSize)
        # Compute optical flow at the current level
        flow = flow.reshape((((height -1) // stepSize) + 1, ((width -1) // stepSize) + 1, 2))
        # Add the current optical flow to the upscaled flow
        flow += current_flow.reshape((((height -1) // stepSize) + 1, ((width -1) // stepSize) + 1, 2))

    height, width = img1.shape
    expand = np.zeros((height, width, 2))
    expand[winSize // 2 :: stepSize, winSize // 2 :: stepSize ] = flow
    flow = expand
    return flow

# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    # iterative method
    for k in range(4):
        ans_u = 0
        ans_v = 0
        list_u_v = []
        # Use lk, move all over the points, and find the u,v that minimizes the diff between im2 that will be to the real im2 that should be
        points, u_v = opticalFlow(im1.astype(float), im2.astype(float),step_size=20, win_size=5)
        for i in range(len(u_v)):
            curr_u = u_v[i,0]
            curr_v = u_v[i,1]
            mat = np.array([[1, 0, curr_u],
                            [0, 1, curr_v],
                            [0, 0, 1]],dtype=float)
            img_2_test = cv2.warpPerspective(im1, mat,(im1.shape[1],im1.shape[0]))
            # add the diff
            list_u_v.append((((im2-img_2_test)**2).sum(),curr_u,curr_v))
        min = float("inf")
        place = 0
        # Take the u,v that minimize the diff to be the answer
        for i in range(len(list_u_v)):
            if list_u_v[i][0] < min:
                min = list_u_v[i][0]
                place = i
        u = list_u_v[place][1]
        v = list_u_v[place][2]
        ans_u+=u
        ans_v+=v
    ans = np.array([[1, 0, ans_u],
                    [0, 1, ans_v],
                    [0, 0, 1]])
    return ans


def findRotation(im1, im2):

    dif = float("inf")
    theta = 0

    for angle in range(360):

        rotation_mat = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]], dtype=float)

        test = cv2.warpPerspective(im1, rotation_mat, im1.shape[::-1])
        mse = np.square(np.subtract(im2, test)).mean()
        if mse < dif:
            dif = mse
            theta = angle

    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta), np.cos(theta), 0],
              [0, 0, 1]], dtype=float)

    return rotation


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    t = findRotation(im1,im2)
    new_pic = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))
    # Find the translation with the new pic
    mat = findTranslationLK(new_pic, im2)
    ans = mat @ t
    return (ans)


# def findRigidLK1(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
#     """
#     :param im1: input image 1 in grayscale format.
#     :param im2: image 1 after Rigid.
#     :return: Rigid matrix by LK.
#     """
#
#     translation_mat = findTranslationLK(im1, im2)
#
#     rotation_mat = findRotation(im1, im2)
#
#     return np.dot(translation_mat, rotation_mat)


# def findTranslationLK1(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
#     """
#     :param im1: image 1 in grayscale format.
#     :param im2: image 1 after Translation.
#     :return: Translation matrix by LK.
#     """
#     points, uv = opticalFlow(im1, im2)  # get uv by LK
#     print(points, uv)
#     uv = uv[uv != [0.0, 0.0]]
#     b = uv.flatten()
#     print(b)
#
#     A = np.zeros((len(b), 2))
#     A[0::2] = [1, 0]
#     A[1::2] = [0, 1]
#
#     x, _, _, _ = lstsq(A, -b)
#
#     # Extract the translation parameters
#     dX, dY = x[:2]
#     print(dX, dY)
#
#     # Create the translation matrix
#     translation_matrix = np.array([[1, 0, dX],
#                                    [0, 1, dY],
#                                    [0, 0, 1]])
#
#     return translation_matrix


def opticalFlowNCC(im1: np.ndarray, im2: np.ndarray, step_size, win_size):
    h = win_size // 2  # half of win
    uv = np.zeros((im1.shape[0], im1.shape[1], 2))  # for each pixel insert uv

    def Max_corr_idx(win: np.ndarray):
        """
        win1: img1 curr window (template)
        norm1: norm of win1
        (same for img2)

        NCC = (win1-mean(win1) * win2-mean(win2)) / (||win1|| * ||win2||)

        :param win:
        :return:
        """
        max_corr = -1000
        corr_idx = (0, 0)
        win1 = win.copy().flatten() - win.mean()
        norm1 = np.linalg.norm(win1, 2)  # normalize win1

        # correlate win1 with img2, and sum the corr in current window
        for i in range(h, im2.shape[0] - h - 1):
            for j in range(h, im2.shape[1] - h - 1):
                win2 = im2[i - h: i + h + 1, j - h: j + h + 1]  # get curr window from img2
                win2 = win2.copy().flatten() - win2.mean()
                norm2 = np.linalg.norm(win2, 2)  # normalize win2
                norms = norm1 * norm2  # ||win1|| * ||win2||
                corr = 0 if norms == 0 else np.sum(win1 * win2) / norms  # correlation sum

                # take the window that maximize the corr
                if corr > max_corr:
                    max_corr = corr
                    corr_idx = (i, j)  # top left pixels of curr window
        return corr_idx

    # each iteration take window from img2, and send to 'Max_corr_idx()' to find template matching
    for y in range(h, im1.shape[0] - h - 1, step_size):
        for x in range(h, im1.shape[1] - h - 1, step_size):
            template = im1[y - h: y + h + 1, x - h: x + h + 1]
            if cv2.countNonZero(template) == 0:
                continue
            index = Max_corr_idx(template)  # index of best 'template matching' in img2
            uv[y - h, x - h] = np.flip(index - np.array([y, x]))

    return uv


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    take median u,v from 'opticalFlowNCC()'
    :param im1: origin img
    :param im2: translated img
    :return: translation matrix
    """
    uvs = opticalFlowNCC(im1, im2, 32, 13)  # get uv of all pixels
    u, v = np.ma.median(np.ma.masked_where(
        uvs == np.zeros(2), uvs), axis=(0, 1)).filled(0)  # take the median u,v
    return np.array([[1, 0, u],
                    [0, 1, v],
                    [0, 0, 1]], dtype=float)


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    t = findRotation(im1, im2)
    new_pic = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))
    # Find the translation with the new pic
    mat = findTranslationCorr(new_pic, im2)
    ans = mat @ t
    return (ans)


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:

    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    if im1.ndim > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if im2.ndim > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    ans = np.zeros(im2.shape)
    transpose = np.linalg.inv(T)
    for i in range(1, im2.shape[0]):
        for j in range(1, im2.shape[1]):
            array = np.array([i, j, 1])
            mat = array @ transpose
            x2 = int(round(mat[0]))
            y2 = int(round(mat[1]))
            if 0 <= x2 < im1.shape[0] and 0 <= y2 < im1.shape[1]:
                ans[i, j] = im1[x2, y2]  # inserting the answer matrix

    return ans


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:

    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    # Create an empty list to store the pyramid levels
    pyramid = [np.zeros(1)] * levels

    # Generate a 2D Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(5, -1)
    gaussian_kernel = gaussian_kernel.dot(gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Calculate the power factor and optimal dimensions for the first level of the pyramid
    pow_fac = np.power(2, levels)
    image_height, image_width = img.shape[:2]
    opt_h, opt_w = pow_fac * (image_height // pow_fac), pow_fac * (image_width // pow_fac)
    pyramid[0] = img[:opt_h, :opt_w]

    # Build the Gaussian pyramid
    for level in range(1, levels):

        # Convolve the previous level's image with the Gaussian kernel
        blurred_image = cv2.filter2D(pyramid[level - 1], -1, gaussian_kernel)
        # Downscale the convolved image by taking every alternate pixel
        blurred_image = blurred_image[::2, ::2]
        # Store the downscaled image as the current level of the pyramid
        pyramid[level] = blurred_image

    return pyramid


def expandImage(img: np.ndarray, gaussian_kernel: np.ndarray) -> np.ndarray:

    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gaussian_kernel: The kernel to use in expanding
    :return: The expanded level
    """

    image_height, image_width = img.shape[:2]
    # Determine the number of color channels in the image
    colors = 1 if img.ndim < 3 else 3
    # Create an empty array for the expanded image
    expanded_img = np.zeros((image_height * 2, image_width * 2, colors)).squeeze()
    # Sample the image by placing the pixels at even positions
    expanded_img[::2, ::2] = img
    # Convolve the expanded image with the Gaussian kernel
    expanded_img = cv2.filter2D(expanded_img, -1, gaussian_kernel)

    return expanded_img


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:

    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    # Generate a 2D Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(5, -1)
    gaussian_kernel = gaussian_kernel.dot(gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    # Multiply the Gaussian kernel by 4 for the padded image
    gaussian_kernel_norm = gaussian_kernel * 4

    # Generate the Gaussian pyramid of the original image
    gaussian_pyramid = gaussianPyr(img, levels)
    # Create an empty list to store the Laplacian pyramid levels
    laplacian_pyramid = [np.zeros(1)] * levels
    # Set the highest level of the Laplacian pyramid as the highest level of the Gaussian pyramid
    laplacian_pyramid[-1] = gaussian_pyramid[-1]

    # Build the Laplacian pyramid
    for level in range(levels - 1):

        # Get the smaller image from the Gaussian pyramid
        small_image = gaussian_pyramid[level + 1]
        # Expand the smaller image to match the size of the next level in the Gaussian pyramid
        expanded_img = expandImage(small_image, gaussian_kernel_norm)
        # Compute the difference between the current Gaussian level and the expanded image
        laplacian_pyramid[level] = gaussian_pyramid[level] - expanded_img

    return laplacian_pyramid


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:

    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """

    # Generate a 2D Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(5, -1)
    gaussian_kernel = gaussian_kernel.dot(gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    # Multiply the Gaussian kernel by 4 for the padded image
    gaussian_kernel_norm = gaussian_kernel * 4

    # Start with the highest level of the Laplacian pyramid
    rolling_img = lap_pyr[-1]

    # Reconstruct the original image from the Laplacian pyramid
    for level in range(len(lap_pyr) - 1, 0, -1):
        # Expand the rolling image using the second Gaussian kernel
        expanded_img = expandImage(rolling_img, gaussian_kernel_norm)
        # Add the expanded image to the current level of the Laplacian pyramid
        rolling_img = expanded_img + lap_pyr[level - 1]

    return rolling_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    # Check if the input images and mask have the same shape
    assert (img_1.shape == img_2.shape)
    assert (img_1.shape[:2] == mask.shape[:2])

    # Create a 5x5 Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(5, -1)
    gaussian_kernel = gaussian_kernel.dot(gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    # Create a 5x5 Gaussian kernel multiplied by 4
    gaussian_kernel_norm = gaussian_kernel * 4

    h, w = img_1.shape[:2]
    pow_fac = np.power(2, levels)
    opt_h, opt_w = pow_fac * (h // pow_fac), pow_fac * (w // pow_fac)
    img_1 = img_1[:opt_h, :opt_w, :]
    img_2 = img_2[:opt_h, :opt_w, :]
    mask = mask[:opt_h, :opt_w, :]

    # Naive blend: element-wise multiplication of image 1 with the mask, and element-wise multiplication of
    # complement of the mask with image 2
    naive_blend = (img_1 * mask) + (1 - mask) * img_2

    # Reduce the resolution of the images using Laplacian pyramid
    laplacian_pyramid_image1 = laplaceianReduce(img_1, levels)
    laplacian_pyramid_image2 = laplaceianReduce(img_2, levels)

    # Generate a Gaussian pyramid for the mask
    gaussian_pyramid_mask = gaussianPyr(mask, levels)

    # Blend the smallest images in the Laplacian pyramid images (the gaussian image) using the mask pyramid
    pyramid_blend = laplacian_pyramid_image1[-1] * gaussian_pyramid_mask[-1] + (1 - gaussian_pyramid_mask[-1]) * laplacian_pyramid_image2[-1]

    # Blend the remaining levels of the Laplacian pyramid
    for level in range(levels - 1, 0, -1):

        # Expand the blended image using a 5x5 Gaussian kernel multiplied by 4
        expanded_image = expandImage(pyramid_blend, gaussian_kernel_norm)

        # Get the mask and Laplacian images for the current level
        mask = gaussian_pyramid_mask[level - 1]
        laplacian_image1 = laplacian_pyramid_image1[level - 1]
        laplacian_image2 = laplacian_pyramid_image2[level - 1]

        # Blend the expanded images using the mask
        blended_image = laplacian_image1 * mask + laplacian_image2 * (1 - mask)

        # Blend the expanded image with the previously blended image
        pyramid_blend = blended_image + expanded_image

    return naive_blend, pyramid_blend

if __name__ == '__main__':

    matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(matrix.flatten())

    gas_ker = cv2.getGaussianKernel(3, -1) * 4
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    ker = cv2.getGaussianKernel(5, sigma)
    gas_ker = gas_ker
    print(gas_ker)
    print(ker)
    gas_ker = gas_ker.dot(gas_ker.T)
    print(gas_ker)
    print(gas_ker.sum())
    gas_ker = gas_ker / gas_ker.sum()
    print(gas_ker)
    print(gas_ker.sum())

    for lv in range(5 - 1, 0, -1):
        print(lv)

    print("jj")
    print(matrix[0::2, 1::2])

    matrix = np.zeros((6, 2))
    matrix[1::2] = [0, 1]
    matrix[0::2] = [1, 0]
    print(matrix)


