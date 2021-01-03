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
        res = res.reshape((784, ))
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
        res *= (1/(steps-1))
        res = res.reshape((784,))
        return res

    @staticmethod
    def skew(image, tilt):
        img = image.reshape((28, 28))
        h, w = img.shape
        res = np.zeros(img.shape)
        for y in range(h):
            s = int(tilt * y)
            for x in range(w - s):
                res[y][x] = img[y][s + x]
        res = res.reshape((784,))
        return res

    def process_image(self, image):
        image = self.rotate(image, random.randint(0, 360))
        image = self.shift(image, random.randint(0, 28), random.randint(0, 28))
        image = self.step_func(image, random.randint(1, 10))
        image = self.skew(image, (random.randint(0, 4))/10)
        return image

    def run(self):
        # print("worker started")
        # proc_name = self.name
        while True:
            indexes = self.jobs_queue.get()
            # print("got job")
            if indexes is None:
                # Poison pill means shutdown
                # print('{}: Exiting'.format(proc_name))
                self.jobs_queue.task_done()
                # print("done")
                break
            # print('{}: {}'.format(proc_name, next_job))
            processed_images = []
            processed_labels = []
            indexes = random.sample(range(0, self.data.shape[0]), self.batch_size)
            images = self.data[indexes]
            labels = self.labels[indexes]
            # print("iterating over images")
            for image, label in zip(images, labels):
                res = self.process_image(np.array(image))
                res = image
                processed_images.append(res)
                processed_labels.append(label)
            # print("job almost put")
            self.result_queue.put((images, labels))
            # print("job put")
            self.jobs_queue.task_done()
        # print("worker done")




def load_image():
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
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
    #img = np.array(image, dtype=float)
    plt.imshow(image, cmap='gray')
    plt.show()

# Note use image show on your local computer to view the results
def test():
    '''run sobel_operator with different correlation functions (CPU, numba, GPU)
        '''
    pic = load_image()
    res = pic
    #print("before: ")
    show_image(pic)
    res = Worker.skew(pic, 0.3)
    #print("after: ")
    show_image(res)


if __name__ == "__main__":
    test()