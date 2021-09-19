import cv2
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

def otsu_thresh(img):

    # img = cv2.imread(file, 0)
    # img = cv2.resize(img, (img_width, img_height))

    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def niblack_thresh(img):

    # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (img_width, img_height))
    res = img.copy()
    thresh_niblack = threshold_niblack(img, window_size=25, k=0.8)
    thresh_niblack = cv2.ximgproc.niBlackThreshold(img, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=2*11+1, k=-0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)

    binary_niblack = img > thresh_niblack

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if binary_niblack[i][j]==True:
                res[i][j]=255
            else:
                res[i][j]=0
    return res

def sauvola_thresh(img):

    # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (img_width, img_height))
    res = img.copy()
    thresh_sauvola = threshold_sauvola(img, window_size=25)
    thresh_sauvola = cv2.ximgproc.niBlackThreshold(img, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=2*11+1, k=-0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)

    binary_sauvola = img > thresh_sauvola

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if binary_sauvola[i][j]==True:
                res[i][j]=255
            else:
                res[i][j]=0
    return res
