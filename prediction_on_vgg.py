
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


# In[2]:


from keras.models import load_model
from keras.preprocessing import image


# In[3]:


import cv2
import numpy as np


# In[ ]:


def getPredict():


# In[4]:


model = load_model('vgg_model_class5.h5')


# In[5]:


model.load_weights('finetune_VGG_class5.h5')


# In[6]:


#model.compile(loss='categorical_crossentropy',
 #             optimizer='adadelta',
  #            metrics=['accuracy'])


# In[7]:


model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# In[8]:


img_width, img_height = 224,224


# In[20]:


#img = cv2.imread('test01.jpg')
img = image.load_img('test_files/blight.jpg', target_size=(img_width, img_height))      


# In[21]:


#test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[22]:


#test_path = 'test_files/test_tomato'


# In[23]:


x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)        


# In[24]:


#img = cv2.resize(img,(224,224))     try this method as well 


# In[25]:


#h,w,l = img.shape
#tuple(img.shape[1::-1])


# In[26]:


images = np.vstack([x])             


# In[27]:


classes = model.predict_classes(images)  # second parameter batch_size = 10     right one 
predictions = model.predict(images)                                            
print(classes)


# In[28]:


#img = np.reshape(img,[1,224,224,3])


# In[29]:


print(predictions)


# In[30]:


if(classes==0):
    print("Bacterial")
elif(classes==1):
     print("Blight")
elif(classes==2):
     print("Healthy") 
elif(classes==3):
     print("Mosaic")
else :print("Spectorial")    


# In[ ]:


return [classes]

