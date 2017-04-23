---
title:  "Training a CNN to Detect Facial Keypoints"
date:   2017-04-22 11:39:23
categories: [keras] 
tags: [keras, regression]
---

In this blog post, I will show you how to use deep learning to detect facial keypoints in images.  The dataset we'll use was featured on [Kaggle](link) a few months ago, and can be downloaded [here](link).

The data is composed of $$96 \times 96$$ grayscale images of cropped human faces, along with 15 corresponding facial landmarks, reported in ($$x, y$$) coordinates.  I've visualized some of the data below, where you can see that there are two landmarks per eyebrow (__four__ total), three per eye (__six__ total), __four__ for the mouth, and __one__ for the tip of the nose.   

![facial keypoints]({{ site.url }}/assets/facial_keypoints.png)

The goal is to create an automated system that can learn from the patterns in this data, in order to predict the facial landmarks in new images.

We'll train a convolutional neural network (CNN) composed of stacks of layers.  The CNN will take an image as input, and its final layer will output a vector with 30 entries, corresponding to the model's predicted locations of each of the 15 facial keypoints.  The CNN will contain over 7 million weights that we'll fit to discovered patterns in the training data.  

![keypoints CNN]({{ site.url }}/assets/keypoints_CNN.png)

We assess the suitability of any candidate collection of model weights according to the mean squared error (MSE) loss function.  MSE is a function of the model weights and is calculated as follows:

This is a draft :)
(write as a function of the model weights)

where x is blah and y is blah.  MSE is minimized when the weights yield a model with predictions that line up well with the true values of the facial keypoints.  Thus, during the training process, we'll search for the weights that minimize the MSE.  

There are many different optimizers in Keras that can be used to find the weights to minimize MSE; I use stochastic gradient descent, but feel free to experiment with the [others](link).

Once we have the weights, we'll use them to predict the locations of the facial landmarks on new images!  The [code](link) in the repository should just be used as a starting point, to get you up and running.  After all, I only run the notebook for 5 epochs.  In practice, you'll have to train the network for much longer.

I am a __huge advocate__ of constructing an end-to-end pipeline before engaging in extensive hyperparameter tuning.  Don't limit yourself to this clean dataset of cropped grayscale faces!  Wouldn't it be much more _rewarding_ to build an algorithm that could extract meaningful information from uncropped color images, or (if you're feeling slightly more ambitious ;)) your webcam?

In this case, the pipeline could work something like this:
1. Accept a color image.
2. Convert the image to grayscale.
3. Detect and crop the face contained in the image.
4. Locate the facial keypoints in the cropped image.
5. Overlay the facial keypoints in the original (color, uncropped) image.

The CNN really only gives you Step 4, but the others are quick to fill in with OpenCV.  Steps 1-2 are quick one-liners, and Steps 3-5 use a pre-trained Haar cascade classifier that is easily accessed [here](link).

I've added this code to the [repository](link), and I hope you enjoy!  Please let me know if you end up using it to add a Snapchat-like filter :).

To get up-and-running quickly, you can run these commands in the terminal:
```
git clone path-to-repo
cd repo
python face_keypoints.py name-of-image
```


