
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm # ProgressBar for loops
                                                                                                            
# from tensorflow.python.ops import rnn_cell, seq2seq

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""

# In[2]:

import arch
from arch import (
    model_name, num_train_batches, data_loader, 
    encode_input, seq_length, labels, keep_prob_val,
    loss, num_valid_batches, learning_rate, keep_prob
)

# In[7]:

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

num_epochs = 100
#num_epochs = 1
verbose = 1      # Display every <verbose> epochs

# In[8]:

init = tf.global_variables_initializer()
saver = tf.train.Saver()

config = arch.config

sess = tf.InteractiveSession(config=config)
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('logs/' + model_name , sess.graph)

sess.run(init)
#saver.restore(sess, 'models/' + model_name)

# In[9]:

step = 0
try:
    for epoch in range(num_epochs):
        train_losses = []
        valid_losses = []

        # Training on train set
        for i in tqdm(range(num_train_batches)):
            batch_inp, batch_outp = data_loader.next_batch()

            input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
            input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
            input_dict[keep_prob] = keep_prob_val

            _, loss_val, summary = sess.run([train, loss, merged], feed_dict=input_dict)
            train_losses.append(loss_val)

            summary_writer.add_summary(summary, step)
            step += 1

        # Testing on valid set
        for i in range(num_valid_batches):
            batch_inp, batch_outp = data_loader.next_batch(data_type='valid')

            input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
            input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
            input_dict[keep_prob] = 1.0

            loss_val = sess.run(loss, feed_dict=input_dict)
            valid_losses.append(loss_val)

        if epoch % verbose == 0:
            log_txt = "Epoch: " + str(epoch) + " Steps: " + str(step) + " train_loss: " + str(round(np.mean(train_losses),4)) + '+' + str(round(np.std(train_losses),2)) +                 " valid_loss: " + str(round(np.mean(valid_losses),4)) + '+' + str(round(np.std(valid_losses),2)) 
            print(log_txt)

            f = open('log.txt', 'a')
            f.write(log_txt + '\n')
            f.close()

            saver.save(sess, 'models/' + model_name)
except KeyboardInterrupt:
    print("Stopped at epoch: " + str(epoch) + ' and step: ' + str(step))

print("Training completed")