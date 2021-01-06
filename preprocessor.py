import multiprocessing
from scipy import ndimage
import numpy as np
import random
import utils
import imageio
import matplotlib.pyplot as plt
from my_queue import *


class Worker(multiprocessing.Process):

    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__(target=self.run)
        self.jobs_queue = jobs
        self.result_queue = result
        self.data = training_data[0]
        self.labels = training_data[1]
        self.batch_size = batch_size

    @staticmethod
    def rotate(image, angle):
        res = image.reshape((28, 28))
        res = ndimage.rotate(res, angle, reshape=False)
        res.reshape((784,))
        return res

    @staticmethod
    def shift(image, dx, dy):
        res = image.reshape((28, 28))
        res = ndimage.shift(res, [dx, dy])
        res = res.reshape((784,))
        return res

    @staticmethod
    def step_func(image, steps):
        res = image.reshape((28, 28))
        if steps == 1:
            return res
        res *= steps
        np.floor(res)
        res *= (1 / (steps - 1))
        res = res.reshape((784,))
        return res

    @staticmethod
    def skew(image, tilt):
        img = image.reshape((28, 28))
        h, w = img.shape
        res = np.zeros(img.shape)
        for y in range(h):
            s = int(tilt * y)
            for x in range(w):
                if w > s+x > 0:
                    res[y][x] = img[y][s + x]
        res = res.reshape((784,))
        return res

    def process_image(self, image):
        res1 = self.rotate(image, random.randint(-10, 10))
        res2 = self.shift(res1, random.randint(-3, 3), random.randint(-3, 3))
        res3 = self.step_func(res2, random.randint(2, 5))
        res4 = self.skew(res3, 0.4*random.random()-0.2)
        return res4

    def run(self):

        while True:
            indexes = self.jobs_queue.get()

            if indexes is None:
                self.jobs_queue.task_done()
                break

            processed_images = []
            processed_labels = []
            indexes = random.sample(range(0, self.data.shape[0]), self.batch_size)
            images = self.data[indexes]
            labels = self.labels[indexes]

            for image, label in zip(images, labels):
                res = self.process_image(np.array(image))
                processed_images.append(res)
                processed_labels.append(label)

            self.result_queue.put((images, labels))
            self.jobs_queue.task_done()


def load_image():
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    # img = np.array(image, dtype=float)
    plt.imshow(image, cmap='gray')
    plt.show()


# Note use image show on your local computer to view the results
def test():
    '''run sobel_operator with different correlation functions (CPU, numba, GPU)
        '''
    pic = load_image()
    res = pic
    # print("before: ")
    show_image(pic)
    res = Worker.skew(pic, 0.3)
    # print("after: ")
    show_image(res)


if __name__ == "__main__":
    test()