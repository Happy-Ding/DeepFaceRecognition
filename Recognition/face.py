import os
import time
import random
import pickle
import logging
import skimage.io
import skimage.transform
import scipy.io as sio
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# logging
logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# Flags
tf.app.flags.DEFINE_boolean('augmentation', True, "Whether shuffle the training data")
tf.app.flags.DEFINE_boolean('random_flip', True, "Whether to random flip left and right")
tf.app.flags.DEFINE_boolean('random_brightness', False, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', False, "whether to random constrast")

tf.app.flags.DEFINE_integer('image_width', 96//2, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_integer('image_height', 112//2, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_boolean('shuffle', True, "Whether shuffle the training data")

tf.app.flags.DEFINE_integer('classes', 10483, 'the max training classes ')
tf.app.flags.DEFINE_integer('display_steps', 200, "the step num to display")
tf.app.flags.DEFINE_integer('eval_steps', 50, "the step num to eval")
tf.app.flags.DEFINE_integer('test_steps', 1000, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 3000, "the steps to save")

tf.app.flags.DEFINE_string('model_name', datetime.strftime(datetime.now(), '%Y%m%d_%H%M'), 'the model name')

tf.app.flags.DEFINE_string('train_data_dir', 'Z:\\CASIA-WebFace-Clean-align\\', 'the train dataset dir') #
tf.app.flags.DEFINE_string('lfw_data_dir', 'Z:\\lfw-deepfunneled-clean-align\\', 'the test lfw dataset dir')

tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "inference"}')
tf.app.flags.DEFINE_integer('epoch_size', 40, 'Validation batch size')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_float('base_lr', 0.01, 'Base Learning Rate')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Learning Rate Decay Steps')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Learning Rate Decay Rate')

#tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
#tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
#tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_string('restore', None, 'the checkpoint')
FLAGS = tf.app.flags.FLAGS

# Image Data Iterator
class DataIterator:
  def __init__(self, data_dirs, shuffle=False):
    logger.info('Start DataIterator')
    records = []
    label = 0
    for data_dir in data_dirs.split("#"):
      logger.info('Building Dataset '+data_dir)
      for folder in os.listdir(data_dir):
        base_path = os.path.join(data_dir+folder)
        for item in os.listdir(base_path):
          records.append((os.path.join(base_path, item), label))
        label += 1
    if shuffle:
      random.shuffle(records)
    self.shuffle = shuffle
    self.image_path = [item[0] for item in records]
    self.labels = [item[1] for item in records]
    del records

  @property
  def size(self):
    return len(self.labels)

  @staticmethod
  def data_augmentation(images):
    if FLAGS.random_flip:
      images = tf.image.random_flip_left_right(images)
    if FLAGS.random_brightness:
      images = tf.image.random_brightness(images, max_delta=0.3)
    if FLAGS.random_contrast:
      images = tf.image.random_contrast(images, 0.9, 1.1)
    return images

  def input_pipeline(self, batch_size, augmentation=False):
    with tf.variable_scope('image_batch_loader'):
      image_path_tensor = tf.convert_to_tensor(self.image_path, dtype=tf.string)
      labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
      input_queue = tf.train.slice_input_producer([image_path_tensor, labels_tensor], num_epochs=FLAGS.epoch_size, shuffle=FLAGS.shuffle)
      
      image_path_input = input_queue[0]
      labels = input_queue[1]
      images_content = tf.read_file(image_path_input)
      im = tf.image.decode_jpeg(images_content, channels=(1 if FLAGS.gray else 3))
      images = tf.to_float(im)
      #images = tf.image.convert_image_dtype(im, tf.float32)
      #images.set_shape((FLAGS.image_height, FLAGS.image_width, (1 if FLAGS.gray else 3)))
      #images = tf.subtract(images, tf.constant(127.5, dtype=tf.float32))
      #images = tf.image.convert_image_dtype(im, tf.float32)
      
      new_size = tf.constant([FLAGS.image_height, FLAGS.image_width], dtype=tf.int32)
      images = tf.image.resize_images(im, new_size)
      images = images - 127.5
      images = images * 0.0125
      if augmentation:
        images = self.data_augmentation(images)
      #labels = tf.one_hot(labels_input, FLAGS.classes)
      if self.shuffle:
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=51200, min_after_dequeue=25600, num_threads=256, allow_smaller_final_batch=True)
      else:
        image_batch, label_batch = tf.train.batch_join([images, labels], batch_size=batch_size, capacity=51200, allow_smaller_final_batch=True)
    return image_batch, label_batch

def residual_block(inputs, depth, reuse=None, scope=None):
  with tf.variable_scope(scope, 'residual_block', [inputs], reuse=reuse):
    net = slim.conv2d(inputs, depth, [1, 1], 1, padding='SAME')
    net = slim.conv2d(net, depth, [1, 1], 1, padding='SAME')
    return net + inputs
    
# Build Network Graph
def build_graph(phase_train=True, weight_decay=0.0001, reuse=None):
  images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_height, FLAGS.image_width, 1 if FLAGS.gray else 3), name='image_batch')
  labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
  with slim.arg_scope([slim.conv2d], 
      weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
      weights_regularizer=slim.l2_regularizer(weight_decay)):
        net = slim.conv2d(images, 32, [3, 3], 1, scope='conv1_1')
        net = slim.conv2d(net, 64, [3, 3], 1, scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], stride=2)
        net = residual_block(net, 64)
        net = residual_block(net, 64)
        net = slim.conv2d(net, 128, [3, 3], 1)
        net = slim.max_pool2d(net, [2, 2], stride=2)
        net = residual_block(net, 128)
        net = residual_block(net, 128)
        net = residual_block(net, 128)
        net = slim.conv2d(net, 256, [3, 3], 1)
        net = slim.max_pool2d(net, [2, 2], stride=2)
        net = residual_block(net, 256)
        net = residual_block(net, 256)
        net = residual_block(net, 256)
        net = residual_block(net, 256)
  features = slim.fully_connected(slim.flatten(net), 512, activation_fn=None)
  logits = slim.fully_connected(features, FLAGS.classes, activation_fn=None, scope='classification')
  #softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
  softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  loss = tf.add_n([softmax_loss] + regularization_losses, name='total_loss')

  global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
  learning_rate = tf.train.exponential_decay(FLAGS.base_lr, global_step, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate, staircase=True)
  train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step) #  AdamOptimizer

  tf.summary.scalar('loss', loss)
  tf.summary.scalar('learning_rate', learning_rate)
  merged_summary_op = tf.summary.merge_all()

  return {'images': images,
          'labels': labels,
          'features': features,
          'loss': loss,
          'global_step': global_step,
          'train_op': train_op,
          'merged_summary_op': merged_summary_op}

def cosine_similarity(feat1, feat2):
  return np.dot(feat1, feat2) / np.linalg.norm(feat1) / np.linalg.norm(feat2)

def histogram(scores, labels, display=True):
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.add_subplot(111)
  h1 = ax.hist(scores[labels==1], bins=128, color='red', alpha=0.7)
  h2 = ax.hist(scores[labels!=1], bins=128, color='black', alpha=0.7)
  #ax.legend( (h1[0], h2[0]), ('Positive Pair', 'Negative Pair') )
  fig.tight_layout()
  if display:
    plt.show()
  else:
    plt.savefig('lfw_histogram.jpg')

def train():
  logger.info('==== Train Mode ====')
  if FLAGS.lfw_data_dir:
    logger.info('LFW directory: %s' % FLAGS.lfw_data_dir)
    lfw_paths = [item.strip() for item in open('data\\lfw_list.txt', 'r')]
    lfw_pairs = []
    lfw_labels = []
    for line in open('data\\lfw_pairs.txt').readlines():
      items = line.strip().split()
      lfw_pairs.append((int(items[0])-1, int(items[1])-1))
      lfw_labels.append(int(items[2]))
    lfw_labels = np.array(lfw_labels)
    lfw_path_length = len(lfw_paths)
    logger.info('Loading LFW images')
    lfw_images = np.zeros((lfw_path_length, FLAGS.image_height, FLAGS.image_width, 1 if FLAGS.gray else 3))
    for i in range(lfw_path_length):
      im = skimage.io.imread(lfw_paths[i], True if FLAGS.gray else False)
      im = skimage.transform.resize(im, (FLAGS.image_height, FLAGS.image_width)) * 1.0
      im = im - 127.5
      im = im * 0.0125
      if FLAGS.gray:
        im = np.reshape(im, (FLAGS.image_height, FLAGS.image_width, 1))
      lfw_images[i, :, :, :] = im
    lfw_features = np.zeros((lfw_path_length, 512))
    logger.info('LFW Loaded.')
  train_feeder = DataIterator(data_dirs=FLAGS.train_data_dir, shuffle=FLAGS.shuffle)
  with tf.Session() as sess:
    train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, augmentation=FLAGS.augmentation)
    logger.info('Init Graph')
    graph = build_graph()
    logger.info('Init Variables')
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    logger.info('Start Coordinator And Queue Runners')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    logger.info('Start Summary')
    summary_writer = tf.summary.FileWriter('.\\logs\\'+FLAGS.model_name, sess.graph)
    logger.info('Check Restore')
    saver = tf.train.Saver(max_to_keep=9999)
    if FLAGS.restore:
      ckpt = tf.train.latest_checkpoint(FLAGS.restore)
      if ckpt:
        saver.restore(sess, ckpt)
        print("restore from {0}".format(ckpt))
        graph['global_step'] += int(ckpt.split('-')[-1])
    logger.info(':::Train Start:::')
    step = 0
    try:
      while not coord.should_stop():
        train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
        feed_dict = {graph['images']: train_images_batch,
                     graph['labels']: train_labels_batch}
        if step % FLAGS.eval_steps != 0:
          _, step = sess.run([graph['train_op'], graph['global_step']], feed_dict=feed_dict)
        else:
          _, loss_val, train_summary, step = sess.run([graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']], feed_dict=feed_dict)
          summary_writer.add_summary(train_summary, step)
        if step % FLAGS.display_steps == 0:
          logger.info("{0} [Train] Step {1} loss {2:.3}".format(datetime.strftime(datetime.now(), '%H:%M'), int(step), loss_val))
        if step % FLAGS.test_steps == 0 and FLAGS.lfw_data_dir:
          for i in range((lfw_path_length//FLAGS.batch_size)+1):
            start = i*FLAGS.batch_size
            end   = (i+1)*FLAGS.batch_size
            if start >= lfw_path_length:
              continue
            if end > lfw_path_length:
              end = lfw_path_length
            lfw_features[start:end, :] = sess.run(graph['features'], {graph['images']: lfw_images[start:end, :, :, :]})
          #diff = np.subtract(np.array([lfw_features[left, :] for (left, right) in lfw_pairs]), np.array([lfw_features[right, :] for (left, right) in lfw_pairs]))
          #scores = np.int32(np.sum(np.square(diff),1))
          scores = np.array([cosine_similarity(lfw_features[left, :], lfw_features[right, :]) for (left, right) in lfw_pairs])
          best_accuracy = 0
          for score in scores:
            predict_labels = 2*(scores >= score) - 1
            accuracy = np.mean(predict_labels == lfw_labels)
            if accuracy > best_accuracy:
              best_accuracy = accuracy
          summary = tf.Summary()
          #pylint: disable=maybe-no-member
          summary.value.add(tag='lfw/accuracy', simple_value=best_accuracy)
          summary_writer.add_summary(summary, step)
          logger.info("{0} [LFW] Accuracy {1}".format(datetime.strftime(datetime.now(), '%H:%M'), best_accuracy))
          histogram(scores, lfw_labels, display=False)
        if (step+1) % FLAGS.save_steps == 0:
          save_path = '.\\models\\'+FLAGS.model_name
          print("save to {0}".format(save_path))
          saver.save(sess, save_path, global_step=graph['global_step'])
    except tf.errors.OutOfRangeError:
        logger.info('==================Train Finished================')
        save_path = '.\\models\\'+FLAGS.model_name
        print("save to {0}".format(save_path))
        saver.save(sess, save_path, global_step=graph['global_step'])
    finally:
        coord.request_stop()
    coord.join(threads)

def main(_):
  if FLAGS.mode == "train":
    train()
  else:
    logger.info("Wrong Mode")

if __name__ == "__main__":
  tf.app.run()