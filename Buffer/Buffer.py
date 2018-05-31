from BufferQueue import BufferQueue
from Instances.Record import Record
from Utils.MetricsUtils import MetricsUtils


class Buffer(object):
    """
    Class for creating buffer as form of Queue object for storing tuples
    """

    def __init__(self, max_size):
        """
        Class constructor - initiate buffer object
        """
        self.__max_size = max_size
        self.__buffer = BufferQueue(max_size)
        self.__centroid = Record([], True)

    @property
    def max_size(self):
        """
        Maximum size of buffer
        """
        return self.__max_size

    @property
    def buffer(self):
        """
        Current queue of tuples
        """
        return self.__buffer.items

    @property
    def size(self):
        """
        Current size of queue
        """
        return self.__buffer.size()

    @property
    def is_empty(self):
        """
        Check whether queue is empty
        """
        return self.__buffer.is_empty()

    @property
    def is_full(self):
        """
        Check whether queue is full, i.e. contains max_size tuples
        """
        return self.__buffer.is_full()

    @property
    def centroid(self):
        """
        Center vector of all records in the buffer
        """
        return self.__centroid

    @centroid.setter
    def centroid(self, v):
        self.__centroid.quasi_identifier = v

    def insert(self, t):
        """
        Insert tuple to queue
        :param t: Tuple to be inserted
        """
        self.__buffer.enqueue(t)
        return self.is_full

    def remove(self):
        """
        Remove tuple from queue
        """
        if not self.is_empty:
            return self.__buffer.dequeue()

    def peek(self):
        """
        Peek into last inserted item in Queue, without removing it
        """
        return self.__buffer.peek()

    def reset(self):
        """
        Reset full buffer and empty it
        """
        self.__buffer.reset()

    def update_buffer_centroid(self):
        """
        Recalculate the centroid of the buffer.
        :return: Updated centroid.
        """
        # self.centroid = []
        self.centroid = MetricsUtils.calculate_centroid(self.buffer)
        return self.centroid
