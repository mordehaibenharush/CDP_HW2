from network import *
from preprocessor import *
import os


class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''

        # Establish communication queues
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()

        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        num_workers = multiprocessing.cpu_count() * 2
        print('Creating {} workers'.format(num_workers))
        workers = [Worker(tasks, results, training_data, self.mini_batch_size) for _ in range(num_workers)]
        for w in workers:
            w.start()

		# 2. Set jobs
        num_jobs = self.mini_batch_size
        for i in range(num_jobs):
            tasks.put(i)

        # Add a poison pill for each consumer
        for _ in range(num_workers):
            tasks.put(None)

        tasks.join()
        processed_data = []
        processed_labels = []
        for r in results.get():
            processed_data.append(r[0])
            processed_labels.append((r[1]))
        augmented_data = (processed_data, processed_labels)
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(augmented_data, validation_data)
        
        # 3. Stop Workers
        for w in workers:
            w.join()

    def create_batches(self, data, labels, batch_size):

        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        raise NotImplementedError("To be implemented")

    
