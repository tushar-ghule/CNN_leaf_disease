
# coding: utf-8

# In[1]:


import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import operator
get_ipython().magic('matplotlib inline')


from keras.models import load_model
from keras.preprocessing import image


import cv2
import numpy as np


model = load_model('vgg_model_class5.h5')
model.load_weights('finetune_VGG_class5.h5')

model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

img_width, img_height = 224,224

#img = cv2.imread('test01.jpg')
img = image.load_img('test_files/blight.jpg', target_size=(img_width, img_height))      

#test_path = 'test_files/test_tomato'


x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)        

#img = cv2.resize(img,(224,224))     try this method as well 

#h,w,l = img.shape
#tuple(img.shape[1::-1])

images = np.vstack([x])             

classes = model.predict_classes(images)  # second parameter batch_size = 10     right one 
predictions = model.predict(images)                                            
print(classes)

print(predictions)

if(classes==0):
    print("Bacterial")
elif(classes==1):
     print("Blight")
elif(classes==2):
     print("Healthy") 
elif(classes==3):
     print("Mosaic")
else :print("Spectorial")    


return [classes]

