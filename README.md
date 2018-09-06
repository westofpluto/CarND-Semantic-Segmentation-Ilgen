# Semantic Segmentation
Semantic Segmentation Project for Udacy SDC Nanodegree

### Overview
The objective of this project was to label the pixels of a road in images from the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). Pixel labeling was done for two classes: road and not-road. The sementic segmentation code was written using Tensorflow to implement a Fully Convolutional Network (FCN) built on the well-known VGG16 network described [here](https://arxiv.org/abs/1605.06211). My implementation used the VGG layers 3, 4 and 7 outputs and added 1x1 convolution, upsample deconvolution, and skip layers. I also used L2 regularization to limit weight size and improve training results.

![VGG16][https://github.com/westofpluto/CarND-Semantic-Segmentation-Ilgen/images/fcnarchvgg16.png]

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform. For training this network, I used a p2.xlarge instance on AWS and used the Udacity Advanced Deep Learning AMI. Note that there are some installation problems for tensorflow when using this AMI on g2 instances, which is why (after some trial and error) I settled on using the p2 instance instead.

##### Frameworks and Packages
The following packages must be installed to run this project:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
This project trains a network on the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) which can be downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Code Description
The code for training the network is contained inside main.py. At the top of the file, I set some hyperparameters including learning rate (1.0E-4), number of epochs (20) and batch size (16). The network specification is done in the function layers() which takes the VGG layers 3, 4 and 7 as inputs. To VGG layer 7 I added a 1x1 convolution layer and a 4x4 upsample deconvolution layer. To VGG layer 4 I added a 1x1 convolution layer, a skip layer (adding the upsampled layer 7 and the convolution layer 4), and a 4x4 upsample deconvolution layer. To VGG layer 3 I added a 1x1 convolution layer, a skip layer (adding the upsampled skip layer from VGG4 and the convolution layer 3), and a 16x16 upsample deconvolution layer. For all these I used an L2 regularizer with regularization constant 1.0E-3.

The optimizer is defined in function optimize. Here I use cross entropy plus regularization as the loss function and use the AdamAoptimizer to train.

Finally, training is performed in the train\_nn function, which loops over epochs and batches per epoch to call the optimize function. The train\_nn function is called by the top level run function that runs when the script is executed.

### Results
I trained the network for 20 epochs using a batch size of 16 and a learning rate of 1.0E-4. The loss decreased in training as follows:

* After epoch 1: 0.2924
* After epoch 4: 0.1368
* After epoch 8: 0.1920
* After epoch 12: 0.0729
* After epoch 16: 0.0598
* After epoch 20: 0.0400

The decrease in loss was not monotonic but obviously the network was indeed training properly and effectively.

After training, sample test images were saved where pixels had been labeled. Road pixels were colored green and not-road pixels were left as-is. From the sample images below, we can state that the training was indeed effective in labeling road pixels.

### Sample Images

![sample1][runs/1536206499.5971024/um_000008.png]

![sample2][runs/1536206499.5971024/um_000053.png]

![sample3][runs/1536206499.5971024/um_000080.png]

![sample4][runs/1536206499.5971024/umm_000039.png]

![sample5][runs/1536206499.5971024/umm_000088.png]



