from numba import cuda
from numba import njit, prange
import imageio
import matplotlib.pyplot as plt
import numpy as np


def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    raise NotImplementedError("To be implemented")


@njit
def correlation_numba(kernel, image):
    res = np.empty_like(image)
    img_rows, img_cols = res.shape
    krnl_rows, krnl_cols = kernel.shape
    img = np.zeros((img_rows + (2*(krnl_rows//2)), img_cols + (2*(krnl_cols//2))))
    img[(krnl_rows//2):-(krnl_rows//2), (krnl_cols//2):-(krnl_cols//2)] = image[:, :]
    for y in prange((krnl_rows//2), (krnl_rows//2)+img_rows):
        for x in prange((krnl_cols//2), (krnl_cols//2)+img_cols):
            curr = img[x-(krnl_rows//2):x+(krnl_rows//2), y-(krnl_cols//2):y+(krnl_cols//2)]
            res[y][x] = np.sum(curr * kernel)

    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    raise NotImplementedError("To be implemented")


def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    raise NotImplementedError("To be implemented")


def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(path)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()

# Note use image show on your local computer to view the results 
def compare_sobel():
    '''run sobel_operator with different correlation functions (CPU, numba, GPU)
        '''
    pic = load_image
    res = pic
    # your implementation
    # show_image(res)
    raise NotImplementedError("To be implemented")
