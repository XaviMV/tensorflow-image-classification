import tensorflow as tf
import numpy as np
import time as t
import random
import os
import cv2


categories = ['class1', 'class2']
categories_test = ['class1_test', 'class2_test']

PATH = os.getcwd()

train_x = [] # arrays of images
train_y = [] # array of the images' labels

test_x = [] # arrays of images
test_y = [] # array of the images' labels

# Save images from each category (class1 and class2)
for categ in categories:
    nou_path = os.path.join(PATH, categ)
    for path_imatge in os.listdir(nou_path):
        img = cv2.imread(os.path.join(PATH, categ, path_imatge), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (150, 150))
        train_x.append(img)
        if (categ == categories[0]):
            train_y.append(0)
        else:
            train_y.append(1)

# Save images from each category (class1_test and class2_test)
for categ in categories_test:
    nou_path = os.path.join(PATH, categ)
    for path_imatge in os.listdir(nou_path):
        img = cv2.imread(os.path.join(PATH, categ, path_imatge), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (150, 150))
        test_x.append(img)
        if (categ == categories_test[0]):
            test_y.append(0)
        else:
            test_y.append(1)


print("TRAIN SAMPLES: ", len(train_x))
print("TEST SAMPLES: ", len(test_x))


train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

test_x = np.asarray(test_x)
test_y = np.asarray(test_y)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (5,5), activation = "relu", input_shape=(150, 150, 1), padding='same'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Conv2D(64, (5,5), activation = "relu", padding='same'),
    tf.keras.layers.MaxPooling2D((5, 5)),
    tf.keras.layers.Conv2D(64, (2, 2), activation = "relu", padding='same'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation = "relu"),
    tf.keras.layers.Dense(32, activation = "relu"),
    tf.keras.layers.Dense(2, activation = "softmax")
    ])

model.summary()

t.sleep(1)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=15, batch_size=10, verbose=1, shuffle=True)

model.save('/tmp/model') # The model is saved so that it can be loaded with the function: tf.keras.models.load_model('/tmp/model')

print("TRAINING FINISHED, STARTING TEST:")

model.evaluate(test_x, test_y, verbose=2)