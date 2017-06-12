import caffe
import cv2
import scipy.io
import numpy as np


w_dim = 340
h_dim = 610
z_dim = 103

samples = 1
channels = 1
height = 1
width = 103
            
            
#Train Image
paviaU_trainImage = cv2.imread('/home/yavuz/myprojects/myjupyter/images/pavia_university/Train_University.bmp')
paviaU = scipy.io.loadmat('/home/yavuz/myprojects/myjupyter/dataset/Pavia_uni/PaviaU.mat')
npPaviaU=np.array(paviaU['paviaU'])

imageCount = 0
for i in range(h_dim):
    for j in range(w_dim):
        
        r = paviaU_trainImage[i][j][0]
        g = paviaU_trainImage[i][j][1]
        b = paviaU_trainImage[i][j][2]
        if(r!=0 or g!=0 or b!=0):
            
            
            vec_hdf5 = np.empty((samples, channels, height, width))
            for z in range(z_dim):
                vec_hdf5[0][0][0][z] = npPaviaU[i][j][z]
                
            if imageCount == 0:
                n = np.int32(vec_hdf5)
            else:
                n = n + np.int32(vec_hdf5)

            #print(n)

            imageCount +=1

avg_img = np.uint8(np.double(n)/imageCount)
print(avg_img)



#create binaryproto
blob = caffe.io.array_to_blobproto(avg_img)
with open("my_mean.binaryproto", 'wb') as f :
    f.write(blob.SerializeToString())

print("DONE!")    


