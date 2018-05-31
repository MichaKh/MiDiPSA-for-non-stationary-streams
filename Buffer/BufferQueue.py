class BufferQueue(object):
    """
    Class implementing FIFO queue, using a list object
    The rear of the queue is at position 0 in the list, and front item is the last item in the list
    """
    def __init__(self, max_size):
        self.__max_size = max_size
        self.__items = []

    @property
    def max_size(self):
        """
        Maximum size of buffer
        """
        return self.__max_size

    @property
    def items(self):
        """
        Element (records) in queue
        """
        return self.__items

    def is_empty(self):
        """
        Check whether queue is empty
        """
        return self.items == []

    def is_full(self):
        """
        Check whether queue is full, i.e. contains max_size tuples
        """
        return self.size() == self.__max_size

    def enqueue(self, item):
        """
        Insert tuple to queue
        :param item: Tuple to be inserted
        """
        self.__items.insert(0, item)

    def dequeue(self):
        """
        Remove tuple from queue
        """
        return self.__items.pop()

    def peek(self):
        """
        Peek into last inserted element of Queue without removing it
        """
        return self.__items[0]

    def size(self):
        """
        Current size of queue
        :return:
        """
        return len(self.__items)

    def reset(self):
        """
        Reset full buffer and empty it
        """
        if self.is_full:
            self.__items = []
