import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import time as t
import random
import os
import cv2

categories = ['class1', 'class2']

PATH = os.getcwd()

train_x = [] # arrays de les imatges
train_y = [] # labels de les imatges

test_x = [] # arrays de les imatges
test_y = [] # labels de les imatges

percentage_test_images = 10 # Percentage of the overall images that will be used only for the testing of the neural network

# Save images from each category (class1 and class2)
for categ in categories:
    nou_path = os.path.join(PATH, categ)
    count = 0
    for path_imatge in os.listdir(nou_path):
        img = cv2.imread(os.path.join(PATH, categ, path_imatge))
        if count%int(100/percentage_test_images) != 0:
            train_x.append(img)
            if (categ == categories[0]):
                train_y.append(0)
            else:
                train_y.append(1)
        else:
            test_x.append(img)
            if (categ == categories[0]):
                test_y.append(0)
            else:
                test_y.append(1)
        count += 1


train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

test_x = np.asarray(test_x)
test_y = np.asarray(test_y)

model = tf.keras.models.load_model("xarxa_neuronal")

print("Loaded!")

model.evaluate(test_x, test_y, verbose=2)

while True:
    num = random.randint(0, len(test_x)-1)
    plt.imshow(test_x[num], cmap='gray')
    plt.show()
    imatge_test = test_x[num]
    imatge_test = np.expand_dims(imatge_test, axis=0)
    print("Guess: ", categories[np.argmax(model.predict(imatge_test, verbose = 0))], ", is ", categories[test_y[num]])
    print()