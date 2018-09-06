#!/usr/bin/env python3
print("Starting")
import os.path
print("before tf import")
import tensorflow as tf
print("after tf import")
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

print("setting hyperparameters")
##########################################
# set some hyperparameters
##########################################
REGULARIZER_SCALE                = 1.0e-3
REGULARIZER_STDEV                = 1.0e-3
REGULARIZATION_LOSS_CONSTANT     = 1.0e-2
LEARNING_RATE                    = 1.0e-4
KEEP_PROB                        = 0.8
NUM_EPOCHS                       = 20
BATCH_SIZE                       = 16

##########################################

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    #
    # MRI: my implementation is from the walkthru
    #
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, layer3, layer4, layer7    

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    #
    # set up regulizer and initializer
    #
    kernel_reg = tf.contrib.layers.l2_regularizer(1.e-3)
    kernel_init = tf.random_normal_initializer(stddev=1.e-3)

    #
    # process VGG7 layer: 1x1 convolution followed by upsample deconvolution
    #
    conv_vgg7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes,kernel_size=1, strides=(1, 1), padding='same', 
                                    kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    deconv_vgg7_upsample = tf.layers.conv2d_transpose(conv_vgg7_1x1, num_classes, kernel_size=4, strides=(2, 2), padding='same',
                                    kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    #
    # process VGG4 layer: 1x1 convolution followed by adding to VGG7 followed by upsample deconvolution
    #
    conv_vgg4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=(1, 1), padding='same',
                                    kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    skip_layer_4_7 = tf.add(deconv_vgg7_upsample, conv_vgg4_1x1)

    deconv_skip_4_7_upsample = tf.layers.conv2d_transpose(skip_layer_4_7, num_classes, kernel_size=4, strides=(2, 2), padding='same',
                                    kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    #
    # process VGG3 layer: 1x1 convolution followed by adding to skip layer 4/7 followed by upsample deconvolution
    # We upsample by 8 to get original image size
    #
    conv_vgg3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, strides=(1, 1), padding='same',
                                    kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)

    skip_layer_3_4_7 = tf.add(deconv_skip_4_7_upsample, conv_vgg3_1x1)

    output = tf.layers.conv2d_transpose(skip_layer_3_4_7, num_classes, kernel_size=16, strides=(8, 8), padding='same',
                                    kernel_regularizer=kernel_reg, kernel_initializer=kernel_init)
    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, overall_loss)
    :NOTE: the overall loss includes cross entropy loss plus some factor * L2 regularization loss
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    #
    # We want to minimize a cost function that is cross entropy loss + (some constant)*regularization loss
    # The regularization constant is a hyperparameter
    #
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    overall_loss = cross_entropy_loss + REGULARIZATION_LOSS_CONSTANT * sum(regularization_losses)

    #
    # minimize overall loss
    #
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(overall_loss)

    return logits, train_op, overall_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())

    #
    # loop over epochs
    #
    for epoch in range(epochs):
        # 
        # print out epoch
        #
        epoch_plus_1 = epoch+1
        print("*****************************************************") 
        print("Epoch %d / %d " % (epoch_plus_1,epochs))
        print("*****************************************************") 

        #
        # loop over all batches this epoch
        #
        batch = 1
        for image, label in get_batches_fn(batch_size):
            #
            # print out and increment batch number 
            #
            print("Batch %d ..." % batch)
            batch = batch + 1

            #
            # train the model
            #
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE})

        #
        # print loss for each epoch
        #
        print("Epoch %d: loss is %.4f" % (epoch_plus_1,loss))

    print("Finished with training!")

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        #
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        #

        #
        # begin by setting placeholders variables
        #
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        #
        # create the FCN model and optimizer
        #
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, overall_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # 
        # train the network
        #
        epochs = NUM_EPOCHS
        batch_size = BATCH_SIZE

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, overall_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples KK-DONE
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
