from network import *
from preprocessor import *
import os


class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        #num_cpu = os.environ['SLURM_CPUS_PER_TASK']
        # Establish communication queues
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        num_jobs = self.number_of_batches
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        num_workers = num_jobs
        print('Creating {} workers'.format(num_workers))
        workers = []
        for _ in range(num_workers):
            w = Worker(tasks, results, training_data, self.mini_batch_size)
            workers.append(w)
        for w in workers:
            w.start()

		# 2. Set jobs
        data = training_data[0]
        labels = training_data[1]
        mini_batches = self.create_batches(data, labels, self.mini_batch_size)

        num_jobs = self.number_of_batches
        for batch in mini_batches:
            tasks.put(batch)

        # Add a poison pill for each consumer
        #for _ in range(num_workers):
         #   tasks.put(None)

        tasks.join()
        print("tasks joined")
        processed_data = []
        processed_labels = []
        for r in results.get():
            processed_data += r[0]
            processed_labels += r[1]
        augmented_data = (processed_data, processed_labels)
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(augmented_data, validation_data)
        
        # 3. Stop Workers
        for w in workers:
            w.join()

    def create_batches(self, data, labels, batch_size):
        """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

        """
        batches = []
        for k in range(self.number_of_batches):
            indexes = random.sample(range(0, data.shape[0]), batch_size)
            batches.append((data[indexes], labels[indexes]))
        return batches
