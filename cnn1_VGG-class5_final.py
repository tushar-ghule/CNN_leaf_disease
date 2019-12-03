
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

#from tensorflow.python.keras.models import Model, Sequential
#from tensorflow.python.keras.layers import Dense, Flatten, Dropout
#from tensorflow.python.keras.applications import VGG16
#from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.optimizers import Adam, RMSprop


#train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['bacterial','blight','healthy','mosaic','spectorial'], batch_size=15)
#valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['bacterial','blight','healthy','mosaic','spectorial'], batch_size=10)
#test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['bacterial','blight','healthy','mosaic','spectorial'], batch_size=50)


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


imgs, labels = next(train_generator)
plots(imgs, titles=labels)
img_width,img_height = 224,224


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
   shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


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

model = applications.VGG16(weights='imagenet')

model.summary()

model.layers.pop()
model.layers.pop()
model.layers.pop()

model.layers.pop()

model.summary()

transfer_layer = model.get_layer('block5_pool')

transfer_layer.output

conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)

model.input
#conv_model.save_weights('finetune_VGG_conv_model_wights.h5')


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


###   function for printing the booleans of trainability 

def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


print_layer_trainable()

#for layer in conv_model.layers:
 #   layer.trainable = False

print_layer_trainable()

new_model.summary()

#for layer in conv_model.layers:
 #   layer.trainable = False

for layer in new_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))


new_model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
#Adam(lr=1e-5)  next try this 


# ## Train the fine-tuned VGG16 model
#model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

new_model.fit_generator(train_generator, steps_per_epoch=180, 
                    validation_data=validation_generator, validation_steps=50, epochs=6, verbose=2)

# #model.fit_generator(train_batches, steps_per_epoch=4, 
# #                    validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)
# ## Predict using fine-tuned VGG16 model


test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)
test_labels

new_model.save_weights('finetune_VGG_class5_aug.h5')
predictions = new_model.predict_generator(test_batches, steps=3, verbose=0)

print (predictions)

ar = np.zeros((50,), dtype=np.int)

for x in range(50):
    max_index, max_value = max(enumerate(predictions[[x]].ravel()), key=operator.itemgetter(1))   
    ar[x] = max_index

ar_test = np.zeros((50,), dtype=np.int)

for x in range(50):
    max_index, max_value = max(enumerate(test_labels[[x]].ravel()), key=operator.itemgetter(1))   
    ar_test[x] = max_index


#cm = confusion_matrix(test_labels, np.round(predictions[:,0]))
cm = confusion_matrix(ar_test, ar)

cm_plot_labels = ['bacterial','blight','healthy','mosaic','spectorial']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


#old model results
cm_plot_labels = ['bacterial','blight','healthy','mosaic','spectorial']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

new_model.save('vgg_model_class5_aug.h5')

