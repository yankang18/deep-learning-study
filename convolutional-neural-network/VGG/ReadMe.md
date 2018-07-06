# VGG (Visual Geometry Group)

* [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)


## 1. Introduction

VGG is a convolutional neural network, which is a type of deep neural network that is most powerful in image processing tasks, such as sorting images into groups. CNN's consist of layers that process visual information. A CNN first takes in an input image and then passes it through these layers. There are a few different types of layers, the three fundamental layers are: 

* convolutional layer, 
* pooling layer, and
* fully-connected layer.

VGG is a group of CNN architectures that each with number of layers. Figure below shows the configurations for different VGG architectures.

<img src='images/vgg_architectures.png'>

below is a network called VGG-16, which has been trained to recognize a variety of image classes. It takes in an image as input, and outputs a predicted class for that image. Also, it has practically the same performance of VGG-19, but with a simpler architecture.

<img src='images/vgg_16.png'>

## 2. Convolutional Layer

A convolutional layer processes the input image directly:

* A convolutional layer takes in an image as input.
* A convolutional layer is made of a set of convolutional filters. Each filter extracts a specific kind of feature, ex. a high-pass filter is often used to detect the edge of an object.
* The output of a given convolutional layer is a set of feature maps, which are filtered versions of an original input image.

### Activation Function

VGG architectures treat a convolutional layer plus activation as one layer. VGG use RELU actication function. The `ReLu` stands for Rectified Linear Unit (ReLU) activation function. This activation function is zero when the input x <= 0 and then linear with a slope = 1 when x > 0. 

ReLu's, and other activation functions, are typically placed after a convolutional layer to slightly transform the output so that it's more efficient to perform backpropagation and effectively train the network.

## 2. Pooling Layer

After a couple of convolutional layers (+ReLu's), in the VGG-16 network, you will see a maxpooling layer.

* Pooling layers take in an image (usually a filtered image) and output a reduced version of that image
* Pooling layers reduce the dimensionality of an input
* Maxpooling layers look at areas in an input image (like the 4x4 pixel area pictured below) and choose to keep the maximum pixel value in that area, in a new, reduced-size area.
* Maxpooling is the most common type of pooling layer in CNN's, but there are also other types such as average pooling.

<img src='images/pooling.png'>

## 4. Fully-Connected Layer

A fully-connected layer's job is to connect the input it sees to a desired form of output. Typically, this means converting a matrix of image features into a feature vector whose dimensions are 1xC, where C is the number of classes. 

As an example, say we are sorting images into ten classes, you could give a fully-connected layer a set of (activated, pooled) feature maps as input and tell it to use a combination of these features (multiplying them, adding them, combining them, etc.) to output a 10-item long feature vector. This vector compresses the information from the feature maps into a single feature vector.

### Softmax

The very last layer you see in this network is a softmax function. The softmax function, can take any vector of values as input and returns a vector of the same length whose values are all in the range (0, 1) and, together, these values will add up to 1. This function is often seen in classification models that have to turn a feature vector into a probability distribution.

Consider the same example again: a network that groups images into one of 10 classes. The fully-connected layer can turn feature maps into a single feature vector that has dimensions 1x10. Then the softmax function turns that vector into a 10-item long probability distribution in which each number in the resulting vector represents the probability that a given input image falls in class 1, class 2, class 3, ... class 10. This output is sometimes called the `class scores` and from these scores, you can extract the most likely class for the given image!

### Overfitting

Convolutional, pooling, and fully-connected layers are all you need to construct a complete CNN, but there are additional layers that you can add to avoid overfitting. One of the most common layers to add to prevent overfitting is a `dropout layer`.

Dropout layers essentially turn off certain nodes in a layer with some probability, p. This ensures that all nodes get an equal chance to try and classify different images during training, and it reduces the likelihood that only a few, heavily-weighted nodes will dominate the process.