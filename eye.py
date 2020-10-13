from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf




def inference(features,is_training):  
  
  with tf.variable_scope('generator'): 
      

      
      with tf.variable_scope('layer1'):
       mOut1=Res_block1(features,1,64,64) # Input featurs number of layers:1, intermediate features number of layers:64, output features nukmber of layers: 64
       mOut2=tf.layers.max_pooling2d(tf.add(mOut1,features), 2, 2, padding='valid', data_format='channels_last', name=None)
      with tf.variable_scope('layer2'):
       mOut3=Res_block1(mOut2,64,64,64)
       mOut4=tf.layers.max_pooling2d(tf.add(mOut3,mOut2), 2, 2, padding='valid', data_format='channels_last', name=None)
      with tf.variable_scope('layer3'):
       mOut5=Res_block1(mOut4,64,64,64)
#       mOut6=tf.layers.max_pooling2d((tf.add(mOut5,tf.concat(axis=3, values=[mOut4, tf.zeros([tf.shape(mOut4)[0],tf.shape(mOut4)[1],tf.shape(mOut4)[2],0])]))), 2, 2, padding='valid', data_format='channels_last', name=None
       mOut6=tf.layers.max_pooling2d(tf.add(mOut5,mOut4), 2, 2, padding='valid', data_format='channels_last', name=None)
      with tf.variable_scope('layer4'):
       mOut7=Res_block1(mOut6,64,64,64)
       mOut8=tf.layers.max_pooling2d(tf.add(mOut7,mOut6), 2, 2, padding='valid', data_format='channels_last', name=None)
 
       
      with tf.variable_scope('layer6'):
       mOut11=Up_conv1(mOut8,64,6)
       mOut12=tf.add(mOut11,Res_block1(mOut11,64,64,64))   
           
      with tf.variable_scope('layer7'):
       mOut13=Up_conv1(mOut12,64,12)
       mOut14=(Res_block1(mOut13,64,64,64))              
           
      with tf.variable_scope('layer8'):
       mOut15=Up_conv1(mOut14,64,24)
       mOut16=tf.add(mOut15,Res_block1(mOut15,64,64,64)) 
       
      with tf.variable_scope('layer9'):
       mOut17=Up_conv1(mOut16,64,48)
       mOut18=(Res_block1(mOut17,64,64,1))
       

      return mOut18

def Res_block1(input_images,s_in,s_middle,s_out):

   conv1_weights = tf.get_variable("conv1_weights",[3, 3,s_in, s_middle],initializer=tf.initializers.truncated_normal(stddev=0.05),
        )
   conv1_biases = tf.get_variable("conv1_biases",[s_middle],initializer=tf.constant_initializer(0.0))
   conv1 = tf.nn.conv2d(input_images, conv1_weights,
        strides=[1, 1, 1, 1], padding='SAME', name="conv1_b")
   relu1 = tf.nn.sigmoid((conv1 + conv1_biases))

   conv2_weights = tf.get_variable("conv2_weights",[3, 3,s_middle,s_out],initializer=tf.initializers.truncated_normal(stddev=0.05),
        )
   conv2_biases = tf.get_variable("conv2_biases",[s_out],initializer=tf.constant_initializer(0.0))
   conv2 = tf.nn.conv2d(relu1, conv2_weights,
        strides=[1, 1, 1, 1], padding='SAME',name="conv2_b")
   
   relu2=(conv2 + conv2_biases)
   
   return tf.nn.sigmoid(relu2)





def Up_conv1(input_Upconv1,s_filter,s_out):
    
   conv1_weights_res = tf.get_variable("up_conv1_weights",[3, 3,s_filter, s_filter],initializer=tf.initializers.truncated_normal(stddev=0.05))
   conv1_biases_res = tf.get_variable("up_biases1_weights",[s_filter],initializer=tf.constant_initializer(0.0))
   conv1_res = tf.nn.conv2d_transpose(input_Upconv1, conv1_weights_res,output_shape=[tf.shape(input_Upconv1)[0],s_out,s_out,s_filter],
        strides=[1, 2, 2, 1], padding='SAME')

  
   return tf.nn.sigmoid(conv1_res + conv1_biases_res)





def fully1(input_images,sizA,sizB,is_training):

   weights = tf.get_variable("weights", [sizA**2,sizB**2],
        initializer=tf.truncated_normal_initializer(stddev=0.05),trainable=is_training)
  
   biases = tf.get_variable("biases", [sizB**2],
        initializer=tf.constant_initializer(0.0),trainable=is_training)


   
   out1 =(tf.matmul(input_images,weights))
        
   relu1 = (out1+biases)
   
#   relu1 = (out1+biases)

   return (relu1)

def train(total_loss, global_step,training_scope,NUMTRAIN_SAMPLES, learning_rate):

  
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):

     opt=tf.train.AdamOptimizer(learning_rate)
     #opt=tf.train.AdamOptimizer(lr)

     apply_gradient_op = opt.minimize(total_loss,global_step=global_step,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=training_scope))


  return apply_gradient_op




def corr(logits,labels):   
    
  m1=tf.reduce_mean(logits)
  m2=tf.reduce_mean(labels)
  n1=tf.reduce_mean(tf.multiply(tf.subtract(logits,m1),tf.subtract(labels,m2)))
  dn1=tf.sqrt((tf.reduce_mean(tf.squared_difference(logits, m1))))
  dn2=tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, m2)))
  return tf.divide(n1,dn1*dn2,name="correlation2D")

def corr_train(logits,labels):   
    
  m1=tf.reduce_mean(logits)
  m2=tf.reduce_mean(labels)
  n1=tf.reduce_mean(tf.multiply(tf.subtract(logits,m1),tf.subtract(labels,m2)))
  dn1=tf.sqrt((tf.reduce_mean(tf.squared_difference(logits, m1))))
  dn2=tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, m2)))
  return tf.divide(n1,dn1*dn2,name="correlation2D")
