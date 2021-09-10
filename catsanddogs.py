#Uses cats and dogs dataset, from: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview
# Need to have dog photos in "/train/dog/" and cat photos in "/train/cat"
#Uses Tensorflow to build CNN (Convolutional Neural Network) model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf

work_directory = "C:/catsanddogs/train"
test_directory= "C:/catsanddogs/test"
categories_list = ["dog", "cat"]

# Now need to iterate over working directory to get images


image_size=100 # set image size. Lower number means lower resolution


trainingdata= [] #create empty training data set

#This function interates over working directory
def create_training_data():
	for category in categories_list:
		path=os.path.join(work_directory, category) # path to separate cats or dogs directory
		class_number = categories_list.index(category) #1 for cat, 0 for dog
		for img in os.listdir(path):
			img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # read as greyscale
			new_array=cv2.resize(img_array, (image_size, image_size))
			trainingdata.append([new_array, class_number])

create_training_data()
print(len(trainingdata))
# need to shuffle data
random.shuffle(trainingdata)
#create sets of variables for features and labels
x=[] #feature set
y=[] #labels

for features, label in trainingdata:
	x.append(features)
	y.append(label)
x=np.array(x).reshape(-1,image_size,image_size,1) #reshape
y=np.array(y)



x=x/255 # scale data
Model = tf.keras.models.Sequential()
Model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=x.shape[1:]))
Model.add(tf.keras.layers.Activation("relu"))
Model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

Model.add(tf.keras.layers.Conv2D(64,(3,3)))
Model.add(tf.keras.layers.Activation("relu"))
Model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# 2x64 CNN layer

Model.add(tf.keras.layers.Flatten())
Model.add(tf.keras.layers.Dense(64)) #64 node dense layer
#3rd layer
Model.add(tf.keras.layers.Dense(1))
Model.add(tf.keras.layers.Activation('sigmoid'))
#output layer

Model.compile(loss="binary_crossentropy",
		optimizer="nadam",
		metrics=['accuracy'])   #Model settings

Mymodel=Model.fit(x,y, epochs=50, validation_split=0.3)  #Compile Model

Model.summary()

#Train and validation loss
plt.figure(figsize=(15,8))
plt.plot(Mymodel.history['loss'][1:], "ro-", label = "Train Loss")
plt.plot(Mymodel.history['val_loss'][1:], "b--", lw=3, label = "Validation Loss")
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.title("Train & Validation Loss (50 epochs)", fontsize=12)
plt.show()

# Train and Validation Accuracy
plt.figure(figsize=(15,8))
plt.plot(Mymodel.history['accuracy'], "ro-", label = "Train Accuracy")
plt.plot(Mymodel.history['val_accuracy'], "b--", lw=3, label = "Validation Accuracy")
plt.legend(loc="lower right", fontsize=12)
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.title("Train & Validation Accuracy (50 epochs)", fontsize=12)
plt.show()

# Test data
testdata=[]
def create_test_data():
	path=os.path.join(test_directory) # path to separate cats or dogs directory
	#class_number = categories_list.index(category) #0 for cat, 1 for dog
	for img in os.listdir(path):
		img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
		new_array=cv2.resize(img_array, (image_size, image_size))
		testdata.append([new_array])
create_test_data()
testdata=np.array(testdata).reshape(-1,image_size,image_size,1) #reshape
testdata=testdata/255 # scale data

preds = Model.predict(testdata) #makepredictions
for i in range(len(preds)):
    if preds[i] >= 0.5:
        preds[i] = 1
    else:
        preds[i] = 0
preds

