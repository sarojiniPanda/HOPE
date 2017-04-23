---
title:  "Global Average Pooling Layers for Object Localization"
date:   2017-04-09 11:39:23
categories: [keras] 
tags: [keras, localization]
use_math: true
---

For image classification tasks, a common choice for CNN architecture is repeated blocks of convolution and maxpooling layers, followed by two or more densely connected layers.  The final dense layer has a softmax activation function and a node for each potential object category.  

As an example, consider the VGG-16 model architecture, depicted in the figure below.

![vgg-16 model]({{ site.url }}/assets/vgg16.png)

We can also summarize the layers of the VGG-16 model by executing the following line of code in the terminal:

```	python
python -c 'from keras.applications.vgg16 import VGG16; VGG16().summary()'
```

Your output should appear as follows:

![vgg-16 layers in Keras]({{ site.url }}/assets/vgg16_keras.png)

You will notice five blocks of (two to three) convolutional layers followed by a max pooling layer.  The final max pooling layer is then flattened and followed by three densely connected layers.  Notice that most of the parameters in the model belong to the fully connected layers!

As you can probably imagine, an architecture like this has the risk of overfitting to the training dataset.  In practice, judicious use of dropout laters is used to avoid overfitting.

#### Global Average Pooling

In the last few years, experts have used global average pooling (GAP) layers to reduce the total number of parameters in the model.  The [first paper](https://arxiv.org/pdf/1312.4400.pdf) to propose GAP layers designed an architecture where the final max pooling layer contained one activation map for each image category in the dataset.  The max pooling layer was then fed to a GAP layer, which yielded a vector with a single entry for each possible object in the classification task.  The authors then applied a softmax activation function to yield the predicted probability of each class.  If you peek at the [original paper](https://arxiv.org/pdf/1312.4400.pdf), I especially recommend checking out Section 3.2, titled "Global Average Pooling".

The [ResNet-50 model](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) takes a less extreme approach; instead of getting rid of dense layers altogether, its final convolutional layer is fed to a GAP layer, followed by one densely connected layer with a softmax activation function that yields the predicted object classes.  

In mid-2016, [researchers at MIT](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) demonstrated that CNNs with GAP layers (a.k.a. GAP-CNNs) that have been trained for a classification task can also be used for [object localization](https://www.youtube.com/watch?v=fZvOy0VXWAI).  That is, a GAP-CNN not only tells us *what* object is contained in the image - it also tells us *where* the object is in the image, and through no additional work on our part!  The localization is expressed as a heat map (henceforth referred to as a __class activation map__), where the color-coding scheme identifies regions that are relatively important for the GAP-CNN to perform the object identification task.  Please check out the YouTube video below for an _awesome_ demo!

<iframe width="560" height="315" style="padding:0px 0px 20px 0px;" src="https://www.youtube.com/embed/fZvOy0VXWAI?rel=0" frameborder="0" allowfullscreen></iframe>

In the [repository](https://github.com/alexisbcook/ResNetCAM-keras), I have explored the localization ability of the pre-trained ResNet-50 model, using the technique from [this paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf).  The main idea is that each of the activation maps in the final convolutional layer acts as a detector for a different pattern in the image, localized in space.  To get the _class_ activation map corresponding to an image, we need only to translate these detected patterns to detected objects.  This translation is done by noticing each node in the GAP layer corresponds to a different activation map, and that the weights in the final dense layer encode each activation map's contribution to the predicted object class.  We can thus sum the contributions of each of the detected patterns in the activation maps (from the final convolutional layer), where detected patterns that are more important to the predicted object class are given more weight.  

#### How the Code Operates

Let $$f_k$$ represent the $$k$$-th activation map in the final convolutional layer.  In the pre-trained ResNet-50 model, the last convolutional layer contains 2048 activation maps, each 7 pixels high and 7 pixels wide.  So, $$f_0$$ is $$7\times7$$ pixels and is the first activation map in the convolutional layer, $$f_1$$ is also $$7\times7$$ pixels and is the second activation map, and so on, where $$f_{2047}$$ is likewise $$7\times7$$ pixels and is the final activation map.  In order to permit comparison to the original image, [bilinear upsampling](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom) is used to resize each activation map to $$224 \times 224$$.

Next, we look at the class that is predicted by the model.  The output node corresponding to the predicted class is connected to every node in the GAP layer.  Let $$w_i$$ represent the weight connecting the $$i$$-th node in the GAP layer to the output node corresponding to the predicted dog breed.  Then, in order to obtain the class activation map, we need only compute the sum

$$w_0 \cdot f_0 + w_1 \cdot f_1 + \ldots + w_{2047} \cdot f_{2047}$$.

This sum is a $$224\times 224$$ array that is then plotted to produce the class activation map.  If you'd like to use this code to do your own object localization, you need only download the [repository](https://github.com/alexisbcook/ResNetCAM-keras).