from multiprocessing import Process, Pipe, Lock


class MyQueue(object):

    def __init__(self):
        self.lock = Lock()
        self.write_p, self.read_p = Pipe()
        ''' Initialize MyQueue and it's members.
        '''

    def put(self, msg):
        self.lock.acquire()
        try:
            self.write_p.send(msg)
        finally:
            self.lock.release()
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''

    def get(self):
        return self.read_p.recv()


def reader_pipe(q):
    while True:
        msg = q.get()
        print(msg)  # Read from the output pipe and do nothing
        if msg=='DONE':
            break


def writer_pipe(count, q):
    if count == -1:
        q.put('DONE')
    q.put(2*count)             # Write 'count' numbers into the input pipe
    q.put(2*count + 1)
    print("*** ", 2 * count)
    print("*** ", 2 * count + 1)

if __name__ == "__main__":
    q = MyQueue()
    reader_p = Process(target=reader_pipe, args=(q,))
    writers = []
    for i in range(5):
        writers.append(Process(target=writer_pipe, args=(i, q)))
    reader_p.start()  # Launch the reader process
    for w in writers:
        w.start()

    for w in writers:
        w.join()
    writer_pipe(-1, q)
    reader_p.join()
