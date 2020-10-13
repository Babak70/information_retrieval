from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import eye
import eye_input
import eye_eval





Flags_train_dir='./fiber10_train'
Flags_log_device_placement=1000000
Flags_log_frequency=15



def write_files(NUMTRAIN_SAMPLES,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label):

   train_dataR=np.fromfile('train_dataF.bin',dtype=np.uint8)
   train_data=np.reshape(train_dataR, [NUMTRAIN_SAMPLES,HEIGHT*WIDTH])


   train_labelsR=np.fromfile('train_labelsF.bin',dtype=np.uint8)
   train_labels=np.reshape(train_labelsR, [NUMTRAIN_SAMPLES,HEIGHT_label*WIDTH_label])

   
   outdata_train = np.concatenate((train_labels,train_data), axis = 1)
   outdata_train.tofile("fiber_train_data_combined.bin")

def train(step0,NUMTRAIN_SAMPLES,batch_size,learning_rate,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label):
  """Train for a number of steps."""
  
  
  write_files(NUMTRAIN_SAMPLES,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label)
  with tf.Graph().as_default():      
         
    global_step = tf.train.get_or_create_global_step()


    with tf.device('/cpu:0'):
      images, labels = eye_input.train_inputs(NUMTRAIN_SAMPLES,batch_size,HEIGHT,WIDTH,HEIGHT_label,WIDTH_label)      
      

  
    tf.summary.image('labels', labels,max_outputs=5) 
    tf.summary.image('images', images,max_outputs=5)



    
    
    with tf.variable_scope('G'):    
     logits = eye.inference(images,True)

    
    tf.summary.image('logits', logits,max_outputs=5)
    r=eye.corr(logits,labels)
    tf.summary.scalar('corr_train', r)
    


    loss=(tf.reduce_mean(tf.squared_difference(logits, labels)))       
    tf.summary.scalar('loss',loss)
    train_op=eye.train(loss, global_step,'G/generator',NUMTRAIN_SAMPLES,learning_rate)

      
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % Flags_log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = Flags_log_frequency * batch_size / duration
          sec_per_batch = float(duration / Flags_log_frequency)

          format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    scaffold=tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=Flags_train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=1000000000),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=Flags_log_device_placement,gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)),scaffold=scaffold,save_checkpoint_steps=195) as mon_sess:
      while not mon_sess.should_stop():
     
        
       GS=mon_sess.run(global_step)
       
       
       if    (GS <=step0):     
            mon_sess.run(train_op)
            print(GS)      
        
       else:
           break
       
        


def main(argv=None): 
#########################################

    batch_size=32 # 
    IMAGE_SIZE = 51 # lenght and width of the input images to the network  
    IMAGE_SIZE_label = 48 # lenght and width of the label images to the network
    learning_rate=0.001  
    step0m=1000 # Number of steps in one epoch. Could also be set to int(NUMTRAIN_SAMPLES/batch_size)
    NUMTRAIN_SAMPLES=60000  # Number of image pairs in the training dataset
    NUMEVAL_SAMPLES=20  # Number of image pairs in the test dataset
#########################################    

    countee=1
    for ij in range(2000000):
     print(countee)     
     step0=step0m*(countee)
     countee+=1
     train(step0,NUMTRAIN_SAMPLES,batch_size,learning_rate,IMAGE_SIZE, IMAGE_SIZE,IMAGE_SIZE_label,IMAGE_SIZE_label)
     print('iteration {} done. Starting Evaluation'.format(countee))
     
     str1='./eval_dataF.bin'
     str2='./eval_labelsF.bin'       
     eye_eval.evaluate(str1,str2,NUMEVAL_SAMPLES, IMAGE_SIZE, IMAGE_SIZE,IMAGE_SIZE_label,IMAGE_SIZE_label)

 
     

    
if __name__ == '__main__':
 tf.app.run()