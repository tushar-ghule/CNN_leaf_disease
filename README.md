# CNN_leaf_disease
disease ditection 

Data is tomato disease, taken from the imagenet. Folder was too big so I didn't upload it here. 
The data is tobe devide in :

For plant names : c_28, c_30, c_32
now divide data set as following: 

1] test  50 images each
2] train 540 images each 
3] valid  130 images each 

We used pretrained VGG16 model.
we remove last 3 layers and add layer according to our requirement, called transfer learning. 
** Transfer learning : It's the technique by which we fine reuse the already accomplished 
model for our use. [ Using fine tuning ]. 

Fine Tune: When we reuse the VGG16 model, we don't need to train all the layers again.
the starting conv layers are trained by imageNet on 1000 images already, so it has 
good property detection characteristics. We just want to train the last dense layer for 
our output. In our case it is 5 leaf dieases. 

Bottleneck method: we freeze the top layers while training, and use all the same features. But, we do train for the last added 
layers and update the features for the same. 
