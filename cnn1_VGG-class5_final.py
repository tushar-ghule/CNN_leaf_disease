
# coding: utf-8

# # Set up

# In[21]:


train_path = 'leaf_tomato_final/train'
valid_path = 'leaf_tomato_final/valid'
test_path = 'leaf_tomato_final/test'


# In[22]:


import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import operator
get_ipython().magic('matplotlib inline')


# In[23]:


from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications


# In[24]:


#from tensorflow.python.keras.models import Model, Sequential
#from tensorflow.python.keras.layers import Dense, Flatten, Dropout
#from tensorflow.python.keras.applications import VGG16
#from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.optimizers import Adam, RMSprop


# In[25]:


#train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['bacterial','blight','healthy','mosaic','spectorial'], batch_size=15)
#valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['bacterial','blight','healthy','mosaic','spectorial'], batch_size=10)
#test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['bacterial','blight','healthy','mosaic','spectorial'], batch_size=50)


# In[26]:


# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


# In[27]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[28]:


imgs, labels = next(train_generator)


# In[29]:


plots(imgs, titles=labels)


# In[30]:


img_width,img_height = 224,224


# In[31]:


#batch_size = 10


# In[32]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[33]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
   shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[34]:


test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[35]:


train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    classes=['bacterial','blight','healthy','mosaic','spectorial'],
    batch_size=15,
   class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=(img_width, img_height),
    classes=['bacterial','blight','healthy','mosaic','spectorial'],
    batch_size=10,
    class_mode='categorical'
)


# # Build and train CNN

# ## Build Fine-tuned VGG16 model

# In[36]:


#vgg16_model = keras.applications.vgg16.VGG16()
#model = VGG16(include_top=True, weights='imagenet')
model = applications.VGG16(weights='imagenet')


# In[37]:


model.summary()


# In[38]:


model.layers.pop()
model.layers.pop()
model.layers.pop()


# In[39]:


model.layers.pop()


# In[40]:


model.summary()


# In[41]:


transfer_layer = model.get_layer('block5_pool')


# In[42]:


transfer_layer.output


# In[43]:


conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)


# In[44]:


model.input


# In[45]:


#conv_model.save_weights('finetune_VGG_conv_model_wights.h5')


# In[46]:


new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add the final layer for the actual classification.
new_model.add(Dense(5, activation='softmax'))


# In[47]:


###   function for printing the booleans of trainability 


# In[48]:



def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


# In[49]:


print_layer_trainable()


# In[50]:


##### set trainability = false 


# In[51]:


#conv_model.trainable = False


# In[52]:


#for layer in conv_model.layers:
 #   layer.trainable = False


# In[53]:


print_layer_trainable()


# In[54]:


new_model.summary()


# In[55]:


for layer in new_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


# In[56]:


new_model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
#Adam(lr=1e-5)  next try this 


# ## Train the fine-tuned VGG16 model

# In[57]:


#model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[58]:


new_model.fit_generator(train_generator, steps_per_epoch=180, 
                    validation_data=validation_generator, validation_steps=50, epochs=6, verbose=2)


# ###### old model results
# #model.fit_generator(train_batches, steps_per_epoch=4, 
# #                    validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

# ## Predict using fine-tuned VGG16 model

# In[39]:


test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)


# In[40]:


#test_labels = test_labels[:,0]
test_labels


# In[59]:


new_model.save_weights('finetune_VGG_class5_aug.h5')


# In[41]:


predictions = new_model.predict_generator(test_batches, steps=3, verbose=0)


# In[44]:


#headHM   Saving model
predictions


# In[45]:


#new_model.save_weights('finetune_VGG_class5.h5')


# In[46]:


ar = np.zeros((50,), dtype=np.int)


# In[47]:


for x in range(50):
    max_index, max_value = max(enumerate(predictions[[x]].ravel()), key=operator.itemgetter(1))   
    ar[x] = max_index


# In[48]:


ar_test = np.zeros((50,), dtype=np.int)


# In[49]:


for x in range(50):
    max_index, max_value = max(enumerate(test_labels[[x]].ravel()), key=operator.itemgetter(1))   
    ar_test[x] = max_index


# In[50]:


#cm = confusion_matrix(test_labels, np.round(predictions[:,0]))
cm = confusion_matrix(ar_test, ar)


# In[51]:



cm_plot_labels = ['bacterial','blight','healthy','mosaic','spectorial']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[52]:


#old model results
cm_plot_labels = ['bacterial','blight','healthy','mosaic','spectorial']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[60]:


new_model.save('vgg_model_class5_aug.h5')

