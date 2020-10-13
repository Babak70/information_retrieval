from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import numpy as np
import tensorflow as tf

import eye_input
import eye




Flags_eval_dir='./fiber10_eval'
Flags_checkpoint_dir='./fiber10_train'
Flags_eval_interval_secs=30
Flags_run_once=True
MOVING_AVERAGE_DECAY = 0.9999 


def eval_once(saver, summary_writer, loss_eval, logits,summary_op,NUMEVAL_SAMPLES,logits_cast_back,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label):

  
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(Flags_checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)

      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      res_out=sess.run(logits_cast_back)
      rr=np.reshape(res_out, [HEIGHT_label,WIDTH_label,NUMEVAL_SAMPLES])
      rr.tofile("./predicted_output.bin")   

      print(sess.run(loss_eval))
 


      
      summary = tf.Summary()            
      summary.ParseFromString(sess.run(summary_op))
      summary_writer.add_summary(summary, global_step)
    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def write_files(str1,str2,NUMEVAL_SAMPLES,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label):
    
   
   eval_dataR=np.fromfile(str1,dtype=np.uint8)
   eval_data=np.reshape(eval_dataR, [NUMEVAL_SAMPLES,HEIGHT*WIDTH])
   eval_labelsR=np.fromfile(str2,dtype=np.uint8)
   eval_labels=np.reshape(eval_labelsR, [NUMEVAL_SAMPLES,HEIGHT_label*WIDTH_label])
   

   outdata_eval2 = np.concatenate((eval_labels,eval_data), axis = 1) 
   outdata_eval2.tofile("fiber_test_data_combined.bin")


def evaluate(str1,str2,NUMEVAL_SAMPLES,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label):
    

  write_files(str1,str2,NUMEVAL_SAMPLES,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label)

  with tf.Graph().as_default() as g:
      
      


    
    images, labels = eye_input.eval_inputs(NUMEVAL_SAMPLES,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label)

    
    tf.summary.image('eval_labels',labels,max_outputs=5)
    tf.summary.image('eval_images', images,max_outputs=5)

    



    with tf.variable_scope('G'):    
     logits = eye.inference(images,False)
     
    tf.summary.image('eval_logits', logits,max_outputs=5)  
    logits_cast_back=tf.image.convert_image_dtype(logits, tf.uint8,saturate=True)
    loss_eval=tf.reduce_mean(tf.squared_difference(logits, labels))   
    tf.summary.scalar('loss',loss_eval)      
    r2=eye.corr(logits,labels)
    tf.summary.scalar('corr_eval', r2)



    # Restore the moving average version of the learned variables for eval.
#    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
#    variables_to_restore = variable_averages.variables_to_restore()
#    saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(Flags_eval_dir, g,flush_secs=7200)

    while True:
      eval_once(saver, summary_writer, loss_eval,logits,summary_op,NUMEVAL_SAMPLES,logits_cast_back,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label)
      if Flags_run_once:
        break
      time.sleep(Flags_eval_interval_secs)





def main(argv=None):
    # Run this for evaluation after training. The evaluation while training is carried out automatically.

   IMAGE_SIZE = 51 # lenght and width of the input images to the network  
   IMAGE_SIZE_label = 48 # lenght and width of the label images to the network
   NUMEVAL_SAMPLES=20  # Number of image pairs in the test dataset

   str1='./eval_dataF.bin'
   str2='./eval_labelsF.bin'
   evaluate(str1,str2,NUMEVAL_SAMPLES, IMAGE_SIZE, IMAGE_SIZE,IMAGE_SIZE_label,IMAGE_SIZE_label)
   


if __name__ == '__main__':
 tf.app.run()