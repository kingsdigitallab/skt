
# coding: utf-8

# In[1]:
																											
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm # ProgressBar for loops
from utils.data_loader import SKTDataLoader

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""

import arch
from arch import (
    model_name, num_train_batches, data_loader, 
    encode_input, seq_length, labels, keep_prob_val,
	keep_prob,
    loss, num_valid_batches, learning_rate,
	test_set_size, decode_outputs_test,
	vocab_size, num_layers
)

batch_size = 1
vocab_size =  data_loader.vocab_size   # Number of unique words in dataset

test_split = 0.01

data_loader = SKTDataLoader(
	'data/dcs_data_input_test_sent.txt',
	'data/dcs_data_output_test_sent.txt',
	batch_size, seq_length, 
	split=[1-test_split, 0, test_split]
)

# In[2]:
num_test_batches = int(data_loader.test_size*1.0/batch_size)

# In[8]:

init = tf.initialize_all_variables()
saver = tf.train.Saver()

sess = tf.InteractiveSession()
#merged = tf.merge_all_summaries()
merged = tf.summary.merge_all()
# summary_writer = tf.train.SummaryWriter('logs/' + model_name , sess.graph)
summary_writer = tf.summary.FileWriter('logs/' + model_name , sess.graph)

sess.run(init)
saver.restore(sess, 'models/' + model_name)

log_path = 'logs/{}_test.log'.format(model_name)

# In[10]:

test_losses = []

# Testing on test set
for i in range(num_test_batches):
	batch_inp, batch_outp = data_loader.next_batch(data_type='test')

	input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
	input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
	input_dict[keep_prob] = 1.0

	loss_val = sess.run(loss, feed_dict=input_dict)
	test_losses.append(loss_val)


log_txt = "Test_loss: " + str(round(np.mean(test_losses), 4)) + '+' + str(round(np.std(test_losses), 2)) 
print(log_txt)

f = open(log_path, 'a')
f.write('-' * 40 + '\n')
f.write(log_txt + '\n')
f.close()


# ### Calculating precision, recall

# #### Getting outputs on entire test set

# In[ ]:

data_loader.reset_index(data_type='test')

X_test = []
y_test = []
y_out = []

for i in tqdm(range(num_test_batches)):
	
	batch_inp, batch_outp = data_loader.next_batch(data_type='test')

	input_dict = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
	input_dict.update({labels[t]: batch_outp[t] for t in range(seq_length)})
	input_dict[keep_prob] = 1.0

	loss_val, outputs = sess.run([loss, decode_outputs_test], feed_dict = input_dict)

	decoded_outputs = np.array(outputs).transpose([1,0,2])
	decoded_outputs = np.argmax(outputs, axis = 2)

	inps = np.swapaxes(batch_inp, 0, 1)
	outps = np.swapaxes(batch_outp, 0, 1)
	gens = np.swapaxes(decoded_outputs, 0, 1)

	for index in range(batch_size):
		inp = ''.join([data_loader.idx2word[x] for x in inps[index] if x != vocab_size-1][::-1])
		outp = ''.join([data_loader.idx2word[x] for x in outps[index] if x != vocab_size-1])
		gen = ''.join([data_loader.idx2word[x] for x in gens[index] if x != vocab_size-1])

		X_test.append(inp.split('\xe2\x96\x81'))
		y_test.append(outp.split('\xe2\x96\x81'))
		y_out.append(gen.split('\xe2\x96\x81'))


# In[ ]:

precisions = []
recalls = []
accuracies = []

f = open(log_path, 'a')

f.write('-' * 40 + '\n')

for inp, outp, gen in zip(X_test, y_test, y_out):

	inp_raw = ' '.join(inp)
	outp_raw = ' '.join(outp)
	gen_raw = ' '.join(gen)

	intersection = set(outp).intersection(gen)
	prec = len(intersection)*1.0/len(gen)
	recall = len(intersection)*1.0/len(outp)

	if outp == gen:
		accuracies.append(1.0)
	else:
		accuracies.append(0.0)
	
	precisions.append(prec)
	recalls.append(recall)

	log_line = '%2i:%2i %s\n      %s\n      %s\n' % (
		int(prec * 100),
		int(recall * 100),
		str(inp_raw).replace('\n', '').lstrip(),
		str(outp_raw).replace('\n', '').lstrip(),
		str(gen_raw).replace('\n', '').lstrip(),
	)

	f.write(log_line)


# In[ ]:

avg_prec = np.mean(precisions)*100.0
avg_recall = np.mean(recalls)*100.0
f1_score = 2*avg_prec*avg_recall/(avg_prec + avg_recall)
avg_acc = np.mean(accuracies) * 100.0

# In[ ]:

summary_lines = '\n'.join([
	'%10s: %2.2f' % (m[0], m[1])
	for m
	in [
		["Precision", avg_prec],
		["Recall", avg_recall],
		["F1_score", f1_score],
		["Accuracy", avg_acc],
	]
])

print(summary_lines)
f.write(summary_lines)

f.close()

print('Logged (input, output, generated, prec, rec) into %s' % log_path)
