import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict
import sys
import os

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools

caffe_root = '/home/yavuz/caffe'
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe

caffe.set_mode_gpu()


def plot_confusion_matrix(cm #confusion matrix
                         ,classes 
                          ,normalize=False
                          ,title='Confusion matrix'
                          ,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("confusion matrix is normalized!")
    
    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Modify the paths given below
deploy_prototxt_file_path = '../../models/paviaUni_yama_3x3/deploy.prototxt'
caffe_model_file_path = '../../models/paviaUni_yama_3x3/paviaUni_train_iter_50000.caffemodel'
#caffe_model_file_path = '../models/patches-15/patches-_iter_' + str(sys.argv[1])  +'.caffemodel'

# CNN reconstruction and loading the trained weights
net = caffe.Net(deploy_prototxt_file_path,caffe_model_file_path, caffe.TEST)

#Test siniflarinin sayisi
GTSize = [6631,18649,2099,3064,1345,5029,1330,3682,947]


accuracies = [0.0] * 9	# empty array of size 9 filled with zeroes
labeledAs = [0.0] * 9	# empty array of size 9 filled with zeroes
counter = 0

total = GTSize[0]+GTSize[1]+GTSize[2]+GTSize[3]+GTSize[4]+GTSize[5]+GTSize[6]+GTSize[7]+GTSize[8]

file = open("../../data/paviaUni_yama_3x3/test_rndm.txt")
temp = file.read().splitlines()		# this ignores the newlines



predicted_lables=[]
true_labels = []
class_names = ["0-Asfalt","1-Cimen","2-Cakil","3-Agac","4-MetalSac","5-Toprak","6-Zift","7-Tugla","8-Golge"]

for line in temp:
    #print(line)

    # every line contains a filename, the filename also contains the true label, extract it
    label = line[len(line)-6:len(line)-5]
    #print(label)
    
    with h5py.File(line, 'r') as f:
        input_image = f['data_3x3x103'][()]
        
    out = net.forward_all(data=input_image)
    pred = int(out['prob'][0].argmax(axis=0))

    predicted_lables.append(pred)
    true_labels.append(int(label))



    if label == str(pred):
        accuracies[int(label)] = accuracies[int(label)] + 1

    labeledAs[pred] = labeledAs[pred] + 1

    counter = counter + 1
    print(counter,' processed!')


print(classification_report(y_true=true_labels,
                             y_pred=predicted_lables,
                             target_names=class_names))

cm = confusion_matrix(y_true=true_labels,
                        y_pred=predicted_lables)	



totalAccuracy = (accuracies[0]+accuracies[1]+accuracies[2]+accuracies[3]+accuracies[4]+accuracies[5]+accuracies[6]+accuracies[7]+accuracies[8])/total
randomAccuracy = (labeledAs[0]*GTSize[0]+labeledAs[1]*GTSize[1]+labeledAs[2]*GTSize[2]+labeledAs[3]*GTSize[3]+labeledAs[4]*GTSize[4]+labeledAs[5]*GTSize[5]+labeledAs[6]*GTSize[6]+labeledAs[7]*GTSize[7]+labeledAs[8]*GTSize[8])/(total*total)

kappa = (totalAccuracy - randomAccuracy) / (1 - randomAccuracy)

print 'totalAccuracy ' + str(totalAccuracy)
print 'randomAccuracy ' + str(randomAccuracy)
print 'kappa ' + str(kappa)


print(cm)							
# Compute confusion matrix
cnf_matrix = cm
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
					  title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
					  title='Normalized confusion matrix')
plt.show()


