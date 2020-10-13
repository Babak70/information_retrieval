from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf



Flags_data_dir='.'






def train_inputs(NUMTRAIN_SAMPLES,batch_size,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label):

  if not Flags_data_dir:
    raise ValueError('Please supply a data_dir')

#  data_dir = os.path.join(FLAGS.data_dir, 'fiber-10-batches-bin')
  data_dir=Flags_data_dir
  
  filenames = [os.path.join(data_dir, 'fiber_train_data_combined.bin')]

  print(filenames)
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)


  filename_queue = tf.train.string_input_producer(filenames)

  with tf.name_scope('data_augmentation'):

    read_input = read_fiber10(filename_queue,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label)

    reshaped_image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32,saturate=False)
    reshaped_label = tf.image.convert_image_dtype(read_input.uint8label, tf.float32,saturate=False)

    
    reshaped_image.set_shape([HEIGHT, WIDTH, read_input.depth])
    reshaped_label.set_shape([HEIGHT_label, WIDTH_label, read_input.depth])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUMTRAIN_SAMPLES *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d Fiber images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)


  images, labels=_generate_image_and_label_batch(reshaped_image, reshaped_label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

  return images, labels






def eval_inputs(NUMEVAL_SAMPLES,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label):
  if not Flags_data_dir:
    raise ValueError('Please supply a data_dir')

#  data_dir = os.path.join(FLAGS.data_dir, 'fiber-10-batches-bin')
  data_dir=Flags_data_dir

  filenames = [os.path.join(data_dir, 'fiber_test_data_combined.bin')]
  num_examples_per_epoch = NUMEVAL_SAMPLES

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with tf.name_scope('input'):

    filename_queue = tf.train.string_input_producer(filenames,shuffle=False)


    read_input = read_fiber10(filename_queue,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label)
    reshaped_image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32,saturate=False)
    reshaped_label = tf.image.convert_image_dtype(read_input.uint8label, tf.float32,saturate=False)



    reshaped_image.set_shape([HEIGHT, WIDTH, read_input.depth])
    reshaped_label.set_shape([HEIGHT_label, WIDTH_label, read_input.depth])


    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)


  return _generate_image_and_label_batch(reshaped_image, reshaped_label,
                                         min_queue_examples,  num_examples_per_epoch,
                                         shuffle=False)
  
  
  
def read_fiber10(filename_queue,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label):

  class Record(object):
    pass
  result = Record()


  result.height =HEIGHT 
  result.width = WIDTH
  result.height_label =HEIGHT_label 
  result.width_label = WIDTH_label
  result.depth = 1
  result.depth_label = 1
  image_bytes = result.height * result.width * result.depth 
  label_bytes = result.height_label * result.width_label * result.depth_label
#  label_bytes=1

  record_bytes = label_bytes + image_bytes


  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes_label = tf.decode_raw(value, tf.uint8)
  record_bytes_image = tf.decode_raw(value, tf.uint8)

  depth_label_major = (tf.reshape(
      tf.slice(record_bytes_label, [0],
                       [label_bytes]),
      [result.depth,result.height_label, result.width_label]))
    
  depth_label_major=tf.transpose(depth_label_major,[1,2,0])
          
  
  depth_image_major = (tf.reshape(
      tf.slice(record_bytes_image, [label_bytes],
                       [image_bytes]),
      [result.depth,result.height, result.width]))
      
  depth_image_major=tf.transpose(depth_image_major,[1,2,0])
      
           
#  result.uint8label = tf.transpose(depth_label_major, [1, 2, 0])
#  result.uint8image=tf.transpose(depth_image_major, [1, 2, 0])
  
  result.uint8label = depth_label_major
  result.uint8image=depth_image_major
  # Convert from [depth, height, width] to [height, width, depth].
#  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def _generate_image_and_label_batch(image0, label0, min_queue_examples,
                                    batch_size, shuffle):
  
  num_preprocess_threads = 1
  num_preprocess_threads_eval = 1
  if shuffle:
    images, labels = tf.train.shuffle_batch(
        [image0, label0],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, labels = tf.train.batch(
        [image0, label0],
        batch_size=batch_size,
        num_threads=num_preprocess_threads_eval,
        capacity=min_queue_examples + 3 * batch_size)


  return images, labels  


