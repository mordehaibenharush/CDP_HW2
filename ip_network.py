from network import *
from preprocessor import *
import os
from my_queue import *


class IPNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):

        num_cpu = int(os.environ['SLURM_CPUS_PER_TASK'])
        # Establish communication queues
        self.tasks = multiprocessing.JoinableQueue()
        self.results = MyQueue()  # multiprocessing.Queue()
        self.num_workers = num_cpu

        workers = []
        for _ in range(self.num_workers):
            w = Worker(self.tasks, self.results, training_data, self.mini_batch_size)
            workers.append(w)

        for w in workers:
            w.start()

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)

        for _ in range(self.num_workers):
            self.tasks.put(None)

        self.tasks.join()
        for w in workers:
            w.join()

    def create_batches(self, data, labels, batch_size):
        for n in range(self.number_of_batches):
            #indexes = random.sample(range(0, data.shape[0]), batch_size)
            self.tasks.put(n)


        processed_batches = []
        for m in range(self.number_of_batches):
            r = self.results.get()
            processed_batches.append(r)
        return processed_batches