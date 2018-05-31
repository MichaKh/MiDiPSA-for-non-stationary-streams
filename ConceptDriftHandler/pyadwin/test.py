from ConceptDriftHandler.pyadwin import Adwin
import random
import numpy

# Delta's standard of 0.01 , but Adwin builder can receive a floating point as the delta parameter.
adwin = Adwin(0.01)
# data_stream = [1] * 30
data_stream = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.7]
# for i in range(0, 1000, 2):
    # data_stream.append(i)
# data_stream = list(numpy.random.uniform(0, 1, 200))
# data_stream.extend([2] * 15)
# data_stream.extend(numpy.random.binomial(10, 0.9, 200))
# data_stream.extend(numpy.random.binomial(10, 0.2, 100))

for data in data_stream:
    if adwin.update(data):
        print "Change has been detected in data: " + str(data)
    print str(data) +" -------- " + str(adwin.getEstimation())  # Prints the next value of the estimated form of data_stream)