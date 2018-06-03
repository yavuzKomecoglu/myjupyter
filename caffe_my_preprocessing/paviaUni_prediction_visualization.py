import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict
import sys
import os
import spectral
import cv2

caffe_root = '/home/yavuz/caffe'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

height = 610
width = 340
outputs = np.zeros((height,width))



def Patch2Pixel(patchNo):
    p=0
    r=9
    x=0
    y=0
    for i in range(618 - r + 1):
        for j in range(348 - r + 1):  
            if(p == patchNo):
                print("{i}:{k}, {j}:{l}".format(i=i-4,k=i+r-4, j=j-4,l=j+r-4))
                print("M: {x},{y}".format(x=i,y=j))
                print("Parca: " + str(p))
                
                x=i
                y=j
            p+=1
            
    return x,y


# Modify the paths given below

#PIKSEL
#deploy_prototxt_file_path = '../../models/paviaUni_piksel/deploy.prototxt'
#caffe_model_file_path = '../../models/paviaUni_piksel/paviaUni_piksel_iter_50000.caffemodel'

#9x9 YAMA
#deploy_prototxt_file_path = '../../models/paviaUni_yama/deploy.prototxt'
#caffe_model_file_path = '../../models/paviaUni_yama/paviaUni_train_iter_50000.caffemodel'

#5x5 YAMA
#deploy_prototxt_file_path = '../../models/paviaUni_yama_5x5/deploy.prototxt'
#caffe_model_file_path = '../../models/paviaUni_yama_5x5/paviaUni_train_iter_50000.caffemodel'

#3x3 YAMA
deploy_prototxt_file_path = '../../models/paviaUni_yama_3x3/deploy.prototxt'
caffe_model_file_path = '../../models/paviaUni_yama_3x3/paviaUni_train_iter_50000.caffemodel'


#PCA
#deploy_prototxt_file_path = '../../models/paviaUni_PCA_yama_15x15/Test1-Alexnet/deploy.prototxt'
#caffe_model_file_path = '../../models/paviaUni_PCA_yama_15x15/Test1-Alexnet/paviaUni_PCA_yama_9x9_train_iter_50000.caffemodel'


#KAYNASTIRMA
#deploy_prototxt_file_path = '../../models/paviaUni_PCA_yama_15x15/Test1-Alexnet/deploy.prototxt'
#caffe_model_file_path = '../../models/paviaUni_PCA_yama_15x15/Test1-Alexnet/paviaUni_piksel_ve_yama_iter_50000.caffemodel'


#caffe_model_file_path = '../models/patches-15/patches-_iter_' + str(sys.argv[1])  +'.caffemodel'

# CNN reconstruction and loading the trained weights
net = caffe.Net(deploy_prototxt_file_path,caffe_model_file_path, caffe.TEST)


GTSize = [6631,18649,2099,3064,1345,5029,1330,3682,947]

accuracies = [0.0] * 9	# empty array of size 9 filled with zeroes
labeledAs = [0.0] * 9	# empty array of size 9 filled with zeroes
counter = 0

total = GTSize[0]+GTSize[1]+GTSize[2]+GTSize[3]+GTSize[4]+GTSize[5]+GTSize[6]+GTSize[7]+GTSize[8]

file = open("../../data/paviaUni_yama_3x3/test_rndm.txt")
temp = file.read().splitlines()		# this ignores the newlines

for line in temp:
    #print(line)

    # every line contains a filename, the filename also contains the true label, extract it
    label = line[len(line)-6:len(line)-5]
    #print(label)

    #yavuz - x ve y'i parse et
    #piksel
    #i,j = line.split('_')[3], line.split('_')[2]
    
    #yama
    p = line.split('_')[3]
    print("p",p)
    i,j = Patch2Pixel(int(p))

    print("i",i)
    print("j",j)

    #
    
    with h5py.File(line, 'r') as f:
        input_image = f['data_3x3x103'][()]
        
    out = net.forward_all(data=input_image)
    pred = int(out['prob'][0].argmax(axis=0))
    print str(pred) + ' ' + str(counter) + ' ' + label

    #print("pred",pred)

    #yavuz - resim dolduruldu
    outputs[int(i)][int(j)] = int(pred) +1 


    if label == str(pred):
        accuracies[int(label)] = accuracies[int(label)] + 1

    labeledAs[pred] = labeledAs[pred] + 1

    counter = counter + 1

	#if counter == 1:
	#	break

totalAccuracy = (accuracies[0]+accuracies[1]+accuracies[2]+accuracies[3]+accuracies[4]+accuracies[5]+accuracies[6]+accuracies[7]+accuracies[8])/total
randomAccuracy = (labeledAs[0]*GTSize[0]+labeledAs[1]*GTSize[1]+labeledAs[2]*GTSize[2]+labeledAs[3]*GTSize[3]+labeledAs[4]*GTSize[4]+labeledAs[5]*GTSize[5]+labeledAs[6]*GTSize[6]+labeledAs[7]*GTSize[7]+labeledAs[8]*GTSize[8])/(total*total)

kappa = (totalAccuracy - randomAccuracy) / (1 - randomAccuracy)

print 'totalAccuracy ' + str(totalAccuracy)
print 'randomAccuracy ' + str(randomAccuracy)
print 'kappa ' + str(kappa)

for i in range(0,9):
    print accuracies[i]/GTSize[i]



#spectral.imshow(classes = outputs.astype(int),figsize =(9,9))
print (outputs)

plt.imshow(outputs)
plt.show()
