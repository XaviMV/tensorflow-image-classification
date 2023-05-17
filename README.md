# tensorflow-image-classification
Simple code to create an image classification model with Tensorflow.

# cats and dogs
As an example I used this code to differentiate pictures of cats and dogs. When the model is given a picture its objective is to determine whether there is a cat or a dog present.

I used a dataset of 10.000 pictures, 5.000 pictures of each animal. Of those pictures I dedicated 8.000 to the training of the model and 2.000 to its testing. Those testing images were not present in the training, so it would be the model's first time seeing them, this is done to ensure that the model does not just memorize the training images and give a false accuracy.

The neural network sees images as 150x150 pixels and in grayscale, this means that if the images given are not perfectly square they will get a bit deformed in order to fit the network's input layer. An example of what the network sees:

![image](https://github.com/XaviMV/tensorflow-image-classification/assets/70759474/3e51d8c9-7735-49d2-98bf-33952cd0ee23)


After 30 minutes of training the neural network ended up with an accuracy of 80.3% on the testing images, however it's training images' accuracy was 90.36%, this means that even though the model did learn to differentiate between the two classes quite successfully (8 correct guesses out of 10 images) it did end up "memorizing" it's training images. This could be for a variety of reasons, but I think the following are the most likely:
  - The model iterated through the training data too many times. The epochs (how many times the model iterates through the training images) were set to 15, and it may have been too much, the "memorizing" could have happened because it saw the same images too many times.


  - There were not enough training images. If the number of images was too low the model may not have been able to find the patterns that define each class, and so it resorted to "memorizing" the images that didn't fall into the patterns.


  - Network complexity. If the network was too complex the model would not have been able to create such complex patterns and it would have resorted to the memorizing. If the network was too simple it also wouldn't have been able to find the more complex patterns that define each class and would have resorted to the memorizing too.

This reasons go hand in hand, for example, if the dataset was bigger the model probably would have been able to do more iterations over the training images, that is because there would have been a bigger pool of images and hence way more examples to memorize, that would have made the model more likely to find a pattern that fits most of the examples instead of memorizing each and every one of the images.

This also goes for the network complexity, if the dataset was bigger the network could get away with being more complex, since the patterns that used to be too complex or vague would have been clearer with more sample images.

The network used convolutional layer as well as dense layers in its structure.
