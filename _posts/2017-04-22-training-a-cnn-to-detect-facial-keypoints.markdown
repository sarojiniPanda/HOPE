---
title:  "Training a CNN to Detect Facial Keypoints"
date:   2017-04-22 11:39:23
categories: [keras] 
tags: [keras, regression]
---

In this blog post, I will outline a procedure for using deep learning to detect facial keypoints in images.  The dataset was featured on [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection) a few months ago, and can be downloaded [here](https://www.kaggle.com/c/facial-keypoints-detection/data).

The data is composed of $$96 \times 96$$ grayscale images of cropped human faces, along with 15 corresponding facial landmarks, reported in ($$x, y$$) coordinates.  I've visualized some of the data below, and you can see that there are two landmarks per eyebrow (__four__ total), three per eye (__six__ total), __four__ for the mouth, and __one__ for the tip of the nose.   

![facial keypoints]({{ site.url }}/assets/facial_keypoints.png)

The goal is to create an automated system that can learn from the patterns in this data, in order to predict the facial landmarks in new images.

We'll train a convolutional neural network (CNN) composed of stacks of layers.  The CNN will take an image as input, and its final layer will output a vector with 30 entries, corresponding to the model's predicted locations of each of the 15 facial keypoints.  The CNN contains over 7 million weights that we'll fit to the training data.  

![keypoints CNN]({{ site.url }}/assets/keypoints_CNN.png)

We assess the suitability of any candidate collection of model weights according to the mean squared error (MSE) loss function.  [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) is minimized when the weights yield a model with predictions that line up well with the true values of the facial keypoints.  Thus, during the training process, we'll search for the weights that minimize MSE.  

There are many different optimizers in Keras that can be used to find the weights to minimize MSE; I use stochastic gradient descent, but feel free to experiment with the [others](https://keras.io/optimizers/).

Once we have the weights, we'll use them to predict the locations of the facial landmarks on new images!  The [code](https://github.com/alexisbcook/facial-keypoints-CNN) in the repository should just be used as a starting point, to get you up and running.  After all, I only run the notebook for 5 epochs.  In practice, you'll have to train the network for much longer.

I am a __huge advocate__ of constructing an end-to-end pipeline before engaging in extensive hyperparameter tuning.  Don't limit yourself to this clean dataset of cropped grayscale faces!  Wouldn't it be much more _rewarding_ to build an algorithm that could extract meaningful information from uncropped color images, or (if you're feeling slightly more ambitious ;)) your webcam?

In this case, the pipeline could work something like this:
1. Accept a color image.
2. Convert the image to grayscale.
3. Detect and crop the face contained in the image.
4. Locate the facial keypoints in the cropped image.
5. Overlay the facial keypoints in the original (color, uncropped) image.

The CNN really only gives you Step 4, but the others are quick to fill in with OpenCV.  Steps 1-2 are quick one-liners, and Steps 3-5 use a pre-trained [Haar cascade classifier](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) that is easily accessed [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).

I'm planning to add this additional code to the [repository](https://github.com/alexisbcook/facial-keypoints-CNN) sometime soon, but if you're just interested in the CNN, the code is ready for you!


