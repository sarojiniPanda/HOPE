---
title:  "Using Transfer Learning to Classify Images with Keras"
date:   2017-04-08 11:39:23
categories: [keras] 
tags: [keras, classification, transfer-learning]
---

In this blog post, I will show you how to _efficiently_ use deep learning to train an algorithm to perform object classification.  This blog post is inspired by a [recent Medium post](https://medium.com/@st553/using-transfer-learning-to-classify-images-with-tensorflow-b0f3142b9366) that made use of Tensorflow.  I will adapt the code to Keras (version 2.0.2), and all code will be written in Python 3.5.  

I will assume that you are already familiar with the ideas behind convolutional neural networks (CNNs) and transfer learning, and we'll focus on discussing the details of my code in Keras. 

If you need to learn more about CNNs, I recommend reading the notes for the [CS231n](http://cs231n.github.io/convolutional-networks/) course at Stanford.  All lectures are also available [online](https://www.youtube.com/watch?v=LxfUGhug-iQ&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=7).  You are also encouraged to check out Term 2 of Udacity's [Artificial Intelligence Nanodegree](https://www.udacity.com/course/artificial-intelligence-nanodegree--nd889), where you can find a comprehensive introduction to neural networks (NNs), CNNs (including transfer learning), and recurrent neural networks (RNNs).

#### The Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a popular dataset composed of 60,000 tiny color images that each depict an object from one of ten different categories.

![cifar-10 dataset]({{ site.url }}/assets/cifar10.png)

This [dataset](https://keras.io/datasets/) is simple to load in Keras.
``` python
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

#### Extracting the InceptionV3 Bottleneck Features

We won't build or train our own CNNs.  Instead, we will use __transfer learning__ to leverage a pre-trained CNN that has demonstrated state-of-the-art performance in object classification tasks. 

Keras makes it very easy to access several pre-trained [CNN architectures](https://keras.io/applications/).  For now, we will focus on the InceptionV3 architecture. 

![inception architecture]({{ site.url }}/assets/inception.png)

After importing the necessary Python class, it's only one line of code to get the model, along with the pre-trained weights.

``` python
from keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=True)
```

The pre-trained InceptionV3 architecture is now stored in the variable `base_model`.  The final layer of this network is a densely connected layer designed to distinguish between the [1000 different object categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) in the ImageNet database.  We will remove this final layer and save the resultant network in a new model.  

``` python
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
```

This new model will no longer return a predicted image class, since the classification layer has been removed; however, the CNN now stored in `model` still provides us with a useful way to extract features from images.  By passing each of the CIFAR-10 images through this model, we can convert each image from its 32x32x3 array of raw image pixels to a vector with 2048 entries.  In practice, we refer to this dataset of 2048-dimensional points as InceptionV3 bottleneck features.  

#### Using t-SNE to Visualize Bottleneck Features

Towards visualizing the bottleneck features, we will use a dimensionality reduction technique called [t-SNE](http://distill.pub/2016/misread-tsne/) (aka t-Distributed Stochastic Neighbor Embedding).  t-SNE reduces the dimensionality of each point, in a way where the points in the lower-dimensional space preserve the pointwise distances from the original, higher-dimensional space.  Scikit-learn [has an implementation](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) of t-SNE, but it is too slow for our purposes.  Instead, to work with our very large, highly dimensional dataset, you'll need to work with a faster implementation.  The implementation we encourage you to use can be found [on github](https://github.com/alexisbcook/tsne), and can be installed by running `pip install git+https://github.com/alexisbcook/tsne.git` in the terminal.

After plotting the resulting 2-dimensional points, color-coded by label, we get the plot below.

![t-sne plot for transfer learning on cifar-10]({{ site.url }}/assets/tsne.png)

InceptionV3 does an amazing job with teasing out the content in the image, where points containing similar objects are mostly confined to nearby regions in the 2D plot.

#### Performing Classification with Transfer Learning

When we train a very shallow NN on the bottleneck features, we attain a test accuracy of 80 percent!  That's amazing! :)

#### Play with the Code!

Can we do better with other pre-trained architectures?  Feel free to download the code on Github and try your own hand at transfer learning!  __Link to repository coming soon ~__