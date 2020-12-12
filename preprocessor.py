import multiprocessing
from scipy import ndimage
import numpy as np
import random


class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()
        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size
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
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''

    @staticmethod
    def shift(image, dx, dy):
        return ndimage.shift(image, [dx, dy])
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
    
    @staticmethod
    def step_func(image, steps):
        image *= steps
        np.floor(image)
        image *= (1/(steps-1))
        return image
        '''Transform the image pixels acording to the step function

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        steps : int
            The number of steps between 0 and 1

        Return
        ------
        An numpy array of same shape
        '''

    @staticmethod
    def skew(image, tilt):
        h, w = image.shape
        skewed = np.empty_like(image)
        skewed[:, :] = 0
        for y in range(h):
            s = tilt * y
            skewed[y, :-s] = image[y, s:]
        return skewed
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''

    def process_image(self, image):
        image = self.rotate(image, random.randint(0, 360))
        image = self.shift(image, random.randint(0, 784), random.randint(748))
        image = self.step_func(image, random.randint(0, 10))
        image = self.skew(image, random.randint(0, 784))
        return image
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        raise NotImplementedError("To be implemented")
