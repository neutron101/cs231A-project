from __future__ import print_function

import argparse
from datetime import datetime
import os, sys, time
from PIL import Image
#1.1.0
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 11
SAVE_DIR = './output/'
MODEL_WEIGHTS = 'coco-food-model/model.ckpt-1510'

def infer(img_path, save_dir=SAVE_DIR, model_weights=MODEL_WEIGHTS, num_classes=NUM_CLASSES):

    """Create the model and start the evaluation process."""

    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 

    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=num_classes)


    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)


    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, model_weights)

    # for path in img_path:
    #     print("img_path {}, save path {}".format(path, save_dir))
    #     pred = prepare_pred(path, num_classes)

    # Perform inference.
    preds = sess.run(pred)

    msk = decode_labels(preds, num_classes=num_classes)
    im = Image.fromarray(msk[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    im.save(save_dir + os.path.basename(img_path))
    print('The segmentation mask has been saved to {}'.format(save_dir + os.path.basename(img_path)))

    tf.reset_default_graph()
    sess.close()


def prepare_pred(img_path, num_classes):

    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 

    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=num_classes)


    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    return pred

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Inference with checkpoint")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("model_weights", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

# def main():

#     args = get_arguments()
#     infer(args.img_path, args.save_dir, args.model_weights, args.num_classes)

    
# if __name__ == '__main__':
#     main()
