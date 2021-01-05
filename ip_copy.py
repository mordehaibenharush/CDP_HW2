from network import *
from preprocessor import *
import os


class IPNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):

        num_cpu = int(os.environ['SLURM_CPUS_PER_TASK'])
        # Establish communication queues
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        self.num_workers = num_cpu
        # print('Creating {} workers'.format(self.num_workers))

        workers = []
        for _ in range(self.num_workers):
            w = Worker(jobs=self.tasks, result=self.results, training_data=training_data, batch_size=self.mini_batch_size)
            workers.append(w)

        for w in workers:
            # print("worker almost started")
            w.start()
            # print("worker started *****************")
        # 2. Set jobs

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)

        # 3. Stop Workers
        # print("***************************")
        for _ in range(self.num_workers):
            self.tasks.put(None)
        self.tasks.join()

    def create_batches(self, data, labels, batch_size):
        for _ in range(self.number_of_batches):
            indexes = random.sample(range(0, data.shape[0]), batch_size)
            self.tasks.put(indexes)
            # print("task_put")

        processed_batches = []
        for _ in range(self.number_of_batches):
            # if not self.results.empty():
            # print("almost got result")
            r = self.results.get()
            # for k in r[0]:
            # print("got result : ", m, len(k))
            processed_batches.append(self.results.get())
            # print("got result: ", )
        return processed_batches


