import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict
import sys
caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

# Modify the paths given below
deploy_prototxt_file_path = '../../models/paviaUni_alexnet/deploy.prototxt'
caffe_model_file_path = '../../models/paviaUni_alexnet/paviaUni_alexnet_train_iter_{iter_num}.caffemodel'
#caffe_model_file_path = '../models/patches-15/patches-_iter_' + str(sys.argv[1])  +'.caffemodel'


KappaValues = []
TotalAccuracyValues = []
RandomAccuracyValues = []
Iteration = [10000,20000,30000,40000,50000]
for iter in xrange(1,6):
     
    cModelPath = caffe_model_file_path.replace('{iter_num}',str(iter*10000))
    # CNN reconstruction and loading the trained weights
    net = caffe.Net(deploy_prototxt_file_path, cModelPath, caffe.TEST)

    print cModelPath + ' train starting...'


    GTSize = [6631,18649,2099,3064,1345,5029,1330,3682,947]

    accuracies = [0.0] * 9	# empty array of size 9 filled with zeroes
    labeledAs = [0.0] * 9	# empty array of size 9 filled with zeroes
    counter = 0

    total = GTSize[0]+GTSize[1]+GTSize[2]+GTSize[3]+GTSize[4]+GTSize[5]+GTSize[6]+GTSize[7]+GTSize[8]

    file = open("../../data/paviaUni/test_rndm.txt")
    temp = file.read().splitlines()		# this ignores the newlines

    for line in temp:
        #print(line)

        # every line contains a filename, the filename also contains the true label, extract it
        label = line[len(line)-6:len(line)-5]
        #print(label)
        
        with h5py.File(line, 'r') as f:
            input_image = f['data'][()]
            
        out = net.forward_all(data=input_image)
        pred = int(out['prob'][0].argmax(axis=0))
        #print str(pred) + ' ' + str(counter) + ' ' + label

        if label == str(pred):
            accuracies[int(label)] = accuracies[int(label)] + 1

        labeledAs[pred] = labeledAs[pred] + 1

        counter = counter + 1

        #if counter == 1:
        #	break

    totalAccuracy = (accuracies[0]+accuracies[1]+accuracies[2]+accuracies[3]+accuracies[4]+accuracies[5]+accuracies[6]+accuracies[7]+accuracies[8])/total
    randomAccuracy = (labeledAs[0]*GTSize[0]+labeledAs[1]*GTSize[1]+labeledAs[2]*GTSize[2]+labeledAs[3]*GTSize[3]+labeledAs[4]*GTSize[4]+labeledAs[5]*GTSize[5]+labeledAs[6]*GTSize[6]+labeledAs[7]*GTSize[7]+labeledAs[8]*GTSize[8])/(total*total)

    kappa = (totalAccuracy - randomAccuracy) / (1 - randomAccuracy)

    KappaValues.append(kappa)
    TotalAccuracyValues.append(totalAccuracy)
    RandomAccuracyValues.append(randomAccuracy)
    print 'iter:' + str(iter) + ', kappa: ' + str(KappaValues[iter-1])
    print 'iter:' + str(iter) + ', totalAccuracy: ' + str(totalAccuracy)
    print 'iter:' + str(iter) + ', randomAccuracy: ' + str(randomAccuracy)


plt.plot(Iteration,KappaValues,color='green')
#plt.plot(Iteration,TotalAccuracyValues,color='blue')
#plt.plot(Iteration,RandomAccuracyValues,color='red')
plt.title('Pavia Uni - AlexNet')
plt.grid(True)
plt.xlabel('Iteration #')
plt.ylabel('Kappa')

plt.show()