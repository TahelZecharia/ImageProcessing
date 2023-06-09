from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv,0))
    print(np.mean(uv,0))

    displayOpticalFlow(img_2, pts, uv)

def getWarpMatrix(method, theta, tx, ty) -> np.ndarray:
    """

    :param method: rigid / trans / rigid_opp
    :param theta: angle (for trans put 0)
    :param tx: move x
    :param ty: move y
    :return: correct warping matrix
    """
    if method == "rigid":
        return np.array([[np.cos(theta), -np.sin(theta), tx],
                         [np.sin(theta), np.cos(theta), ty],
                         [0, 0, 1]], dtype=np.float64)
    elif method == "trans":
        return np.array([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0, 1]], dtype=np.float64)
    elif method == "rigid_opp":
        return np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]], dtype=np.float64)


def dispay_hierarchicalk(img: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    (h, w, z) = uvs.shape
    pts_list = []
    uv_list = []
    for i in range(h):
        for j in range(w):
            if uvs[i, j, 0] != 0 or uvs[i, j, 1] != 0:
                pts_list.append([i, j])
                uv_list.append([uvs[i, j, 0], uvs[i, j, 1]])

    pts = np.asarray(pts_list)
    uv = np.asarray(uv_list)
    plt.quiver(pts[:, 1], pts[:, 0], uv[:, 0], uv[:, 1], color='r')
    plt.show()


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    uv = opticalFlowPyrLK(img_1.astype(float), img_2.astype(float), k=4, stepSize=10, winSize=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    dispay_hierarchicalk(img_1, uv)

def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """

    print("Compare LK & Hierarchical LK")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = getWarpMatrix("trans", 0, -.2, -.1)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st_lk = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float64), img_2.astype(np.float64), step_size=20, win_size=5)
    et_lk = time.time() - st_lk
    st_hir = time.time()
    uv_hir = opticalFlowPyrLK(img_1.astype(np.float64), img_2.astype(np.float64), k=4, stepSize=20, winSize=5)
    et_hir = time.time() - st_hir
    print("time for LK: ", et_lk * 60, " sec ")
    print("time for Hierarchical LK: ", et_hir * 60, " sec ")
    mean_uv_lk = np.mean(uv, axis=0)
    mean_u_hir = uv_hir[:, :, 0].sum() / len(uv_hir[:, :, 0])
    mean_v_hir = uv_hir[:, :, 1].sum() / len(uv_hir[:, :, 1])
    tx = t[0, 2]
    ty = t[1, 2]
    print("real tx: ", tx, " real ty: ", ty)
    print("mean of tx for lk: ", np.around(mean_uv_lk[0], decimals=3), " mean of ty for lk: ",
          np.around(mean_uv_lk[1], decimals=3))
    print("mean of tx for Hierarchical lk: ", np.around(mean_u_hir, decimals=3), " mean of ty for Hierarchical lk: ",
          np.around(mean_v_hir, decimals=3))


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")
    orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    # tranLkTest(orig_img)
    # rigidLKTest(orig_img)
    # tranCorrTest(orig_img)
    # rigidCorrTest(orig_img)
    WarpImageTest(orig_img)

def tranLkTest(orig_img):
    print("----------tranLkTest Demo--------------")
    t = getWarpMatrix("trans", 0, -2, -4)
    orig_img = cv2.resize(orig_img, (0,0) , fx=.5, fy=0.5)
    tran_img = cv2.warpPerspective(orig_img, t, orig_img.shape[::-1])
    tran_lk = findTranslationLK(orig_img, tran_img)
    print(tran_lk)
    f, ax = plt.subplots(2)
    plt.gray()
    img_2 = cv2.warpPerspective(orig_img, tran_lk, orig_img.shape[::-1])
    ax[0].imshow(tran_img)
    ax[0].set_title('trans_img by origin T')
    ax[1].imshow(img_2)
    ax[1].set_title('trans_img by my T')
    plt.show()
    print("mse = ", np.square(tran_img - img_2).mean())
    print("----------end tranLkTest Demo--------------")


def rigidLKTest(img1):
    print("---------rigidLKTest Demo---------")
    theta = 45
    t = getWarpMatrix("rigid", theta, 2, 4)
    tran_img = cv2.warpPerspective(img1, t, img1.shape[::-1])
    tran_lk_ri = findRigidLK(img1, tran_img)
    print("origin transformation matrix\n", t)
    print("my transformation matrix \n", tran_lk_ri)
    f, ax = plt.subplots(2)
    plt.gray()
    img_2 = cv2.warpPerspective(img1, tran_lk_ri, img1.shape[::-1])
    ax[0].imshow(tran_img)
    ax[0].set_title('trans_img by origin T')
    ax[1].imshow(img_2)
    ax[1].set_title('trans_img by my T')
    plt.show()
    MSE = np.square(np.subtract(tran_img, img_2)).mean()
    print("mse = ", MSE)
    print("---------end rigidLKTest Demo---------")


def tranCorrTest(orig_img):
    print("-------tranCorrTest Demo------")
    orig_img = cv2.resize(orig_img, (140, 140), interpolation=cv2.INTER_LINEAR)
    t = getWarpMatrix("trans", 0, -4, -2)
    tran_img = cv2.warpPerspective(orig_img, t, orig_img.shape[::-1])
    time_start = time.time()
    tran = findTranslationCorr(orig_img, tran_img)
    print(tran)
    print("time :", time.time() - time_start)
    img_2 = cv2.warpPerspective(orig_img, tran, orig_img.shape[::-1])  # with the new translation matrix
    f, ax = plt.subplots(2)
    plt.gray()
    ax[0].imshow(tran_img)
    ax[0].set_title('trans_img by origin T')
    ax[1].imshow(img_2)
    ax[1].set_title('trans_img by my T')
    plt.show()
    print("mse = ", np.square(tran_img - img_2).mean())
    print("-------end tranCorrTest Demo------")


def rigidCorrTest(orig_img):
    print("-------rigidCorrTest Demo--------")
    orig_img = cv2.resize(orig_img, (150, 150), interpolation=cv2.INTER_LINEAR)
    theta = 45
    t = getWarpMatrix("rigid", theta, 0, 0)
    tran_img = cv2.warpPerspective(orig_img, t, orig_img.shape[::-1])
    time_start = time.time()
    tran = findRigidCorr(orig_img, tran_img)
    print(tran)
    print("time :", time.time() - time_start)
    img_2 = cv2.warpPerspective(orig_img, tran, orig_img.shape[::-1])  # with the new translation matrix
    f, ax = plt.subplots(2)
    plt.gray()
    ax[0].imshow(tran_img)
    ax[0].set_title('by origin tran matrix')
    ax[1].imshow(img_2)
    ax[1].set_title('by finding the tran matrix')
    plt.show()
    print("mse = ", np.square(tran_img - img_2).mean())
    print("--------end rigidCorrTest----------- ")


def WarpImageTest(orig_img):
    print("---------warp image test-----------")
    orig_img = cv2.resize(orig_img, (0, 0), fx=.5, fy=0.5)
    t_trans = getWarpMatrix("trans", 0, 40, 20)
    theta = 50
    t_rigid = getWarpMatrix("rigid", theta, 2, 4)

    t_rot = getWarpMatrix("rigid", theta, 0, 0)

    warp_cv = cv2.warpPerspective(orig_img, t_rot, orig_img.shape[::-1])
    img_my_warp_back = warpImages(orig_img, warp_cv, t_rot)  # warp back to im1

    f, ax = plt.subplots(4)
    plt.gray()

    ax[0].imshow(orig_img)
    ax[0].set_title('orig')
    ax[1].imshow(warp_cv)
    ax[1].set_title('warped img')
    ax[2].imshow(img_my_warp_back)
    ax[2].set_title('my warping - back to orig')

    plt.show()

    print("---------end warp image test-----------")


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    # lkDemo(img_path)
    # hierarchicalkDemo(img_path)
    # compareLK(img_path)

    imageWarpingDemo(img_path)

    # pyrGaussianDemo('input/pyr_bit.jpg')
    # pyrLaplacianDemo('input/pyr_bit.jpg')
    # blendDemo()


if __name__ == '__main__':
    main()
