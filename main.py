import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import random

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from collections import Counter

row_no         = 0
prob           = 0.26
new_images     = []
steering_angle = []

print('Reading csv file...')
print()

## Creating a numpy array to store the image pixel values

def read_image(row, im):
    tmp_img = plt.imread("data/"+ row[im])
    tmp_img = cv2.resize(tmp_img,(200, 100), interpolation = cv2.INTER_AREA)
    new_images.append(tmp_img)

    if row[im][4] == 'r':
        steering_correction = -0.25
    
    elif row[im][4] == 'l':
        steering_correction = 0.25
        
    else:
        steering_correction = 0.0
    
    steering_angle.append(float(row[3]) + steering_correction)


# Reading the csv file to decide which images to load

with open('./data/driving_log.csv', 'rU') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        if row_no != 0:
            for im in range(3):
                if row[im][0] == ' ':
                    row[im] = row[im][1::]
                
                if abs(float(row[3])) - 0.05 > 0.:
                    read_image(row, im)

                else:
                    
                    if random.random() < prob:
                        read_image(row, im)              
                    
        else:
            row_no = row_no + 1

        
steer_dist = Counter(steering_angle)

x = list(steer_dist.values())
y = list(steer_dist.keys())

# Plotting the distribution of steering angles

fig = plt.figure()
plt.hist(steering_angle, alpha = 0.75)
plt.xticks(np.arange(-1.3, 1.4, 0.1))
plt.xlabel('Streering Angle')
plt.ylabel('No. of Instances')
plt.grid(True)
fig.savefig('figure_3.png')
plt.close(fig)

new_images     = np.array(new_images)
steering_angle = np.array(steering_angle)
print('Total images found',new_images.shape[0])
print() 

## Shuffling the data and dividing the data into training and validation set.

print('Shuffling Data ...')
print()
X_train, y_train = shuffle(new_images, steering_angle)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

nb_classes = len(np.unique(y_train))
ch, row, col = 3, 66, 200
new_row, new_col = 66,200  
lrn_rate = 0.001
batch_size = 32


## Building a keras model

print('Creating Keras model ...')
print()

model = Sequential()

model.add(Cropping2D(cropping=((34, 0), (0, 0)), input_shape=(100,200,3)))

model.add(Lambda(lambda x: x/255. - .5,
       input_shape=(row, col, ch),
       output_shape=(row, col, ch)))

model.add(BatchNormalization(input_shape=(66, 200, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1164, activation='elu'))

model.add(Dense(100, activation='elu'))

model.add(Dense(50, activation='elu'))

model.add(Dense(10, activation='elu'))

model.add(Dense(1, name='y_pred'))
      
## Self defined keras generator

def get_next_batch(data, angle, batch_size = 32):
    while True:
        batch_train = np.zeros((batch_size, 100, 200, 3), dtype = np.float32)
        batch_angle = np.zeros((batch_size,), dtype = np.float32)
        index = np.arange(data.shape[0])
        for i in range(batch_size):
            random = int(np.random.choice(data.shape[0],1))
            batch_train[i] = data[random]
            batch_angle[i] = angle[random]
        yield batch_train, batch_angle
        

training_generator = get_next_batch(X_train, y_train, batch_size)
valid_generator    = get_next_batch(X_validation, y_validation, batch_size)
  

## Training the model
      
print('Training..')
print()

adam = Adam(lr=0.001, decay=1e-6, epsilon=1e-08)
model.compile(loss='mean_squared_error',optimizer=adam, metrics=['accuracy'])

model.fit_generator(training_generator, samples_per_epoch = len(X_train), nb_epoch=10, validation_data=valid_generator, 
								  nb_val_samples = len(X_validation))

model.save('fifth_try.h5') 
print('Model Saved')
