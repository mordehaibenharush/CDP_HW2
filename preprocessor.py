import multiprocessing
from scipy import ndimage
import numpy as np
import random
import utils
import imageio
import matplotlib.pyplot as plt


class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()
        self.jobs_queue = jobs
        self.result_queue = result
        self.data = training_data[0]
        self.labels = training_data[1]
        self.batch_size = batch_size
        #self.batch = utils.create_batches(training_data[0], training_data[1], batch_size)
        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''

    @staticmethod
    def rotate(image, angle):
        return ndimage.rotate(image, angle, reshape=False)

    @staticmethod
    def shift(image, dx, dy):
        return ndimage.shift(image, [dx, dy])
    
    @staticmethod
    def step_func(image, steps):
        if steps == 1:
            return image
        image *= steps
        np.floor(image)
        image *= (1/(steps-1))
        return image

    @staticmethod
    def skew(image, tilt):
        h, w = image.shape
        skewed = np.zeros(image.shape)
        #skewed[:, :] = 0
        for y in range(h):
            s = int(tilt * y)
            for x in range(w - s):
                skewed[y][x] = image[y][s + x]
        return skewed

    def process_image(self, image):
        image = self.rotate(image, random.randint(0, 360))
        image = self.shift(image, random.randint(0, 784), random.randint(0, 748))
        image = self.step_func(image, random.randint(0, 10))
        image = self.skew(image, random.randint(0, 784))
        return image

    def run(self):
        #proc_name = self.name
        #while True:
        print("worker started")
        batch = self.jobs_queue.get()
        print("got job")
            #if next_job is None:
                # Poison pill means shutdown
             #   print('{}: Exiting'.format(proc_name))
              #  self.jobs_queue.task_done()
               # break
           # print('{}: {}'.format(proc_name, next_job))
        processed_batch = []
        processed_labels = []
        data = batch[0]
        labels = batch[1]
        for image, label in zip(data, labels):
            #res = self.process_image(np.array(image).reshape((len(image), 1)))
            processed_batch.append(image)
            processed_labels.append(label)
        self.result_queue.put((processed_batch, labels))
        print("put result")
        self.jobs_queue.task_done()
        print("worker done")
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''



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