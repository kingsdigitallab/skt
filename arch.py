from utils.data_loader import SKTDataLoader
import tensorflow as tf
import numpy as np

# NN architecture

num_layers = 3 # Number of layers of RNN
num_hidden = 128 # Hidden size of RNN cell
batch_size = 128 # Number of sentences in a batch
seq_length = 35 # Length of sequence
split = [0.9, 0.1, 0] # Splitting proportions into train, valid, test
learning_rate = 0.001 # Initial learning rate
keep_prob_val = 0.8 # keep_prob is 1 - dropout i.e., if dropout = 0.2, then keep_prob is 0.8
float_type=tf.float32
int_type=np.int32

if 1:
    num_layers = 2 # Number of layers of RNN
    num_hidden = 64 # Hidden size of RNN cell
    batch_size = 32 # Number of sentences in a batch
    seq_length = 35 # Length of sequence
    split = [0.02, 0.01, 0] # Splitting proportions into train, valid, test

# Data loading

# In[3]:
data_loader = SKTDataLoader('data/dcs_data_input_train_sent.txt','data/dcs_data_output_train_sent.txt',batch_size,seq_length, split=split)
vocab_size =  data_loader.vocab_size   # Number of unique words in dataset

data_size = data_loader.data_size      # Number of paris in the entire dataset
train_set_size = data_loader.train_size# Number of pairs in train set
valid_set_size = data_loader.valid_size# Number of pairs in valid set
test_set_size = data_loader.test_size  # Number of pairs in test set

num_train_batches = int(train_set_size*1.0/batch_size) # Number of train batches1
num_valid_batches = int(valid_set_size*1.0/batch_size)
num_test_batches = int(test_set_size*1.0/batch_size)

# model_name = 'attn_3_8000_0.8_trainonly' # Name is <num_layers>_<sentencepiece_vocabsize>_<keep_prob>
model_name = 'attn_{}_{}x{}_{:.2f}_{:.3f}_trainonly'.format(vocab_size, num_layers, num_hidden, keep_prob_val, learning_rate)

print('-' * 10)
print("Model name: %s" % (model_name))
print("Vocabulary: " + str(vocab_size))
print("Data set  : %d -> tra: %d + val: %d + tst: %d" % (data_size, train_set_size, valid_set_size, test_set_size))
print("Arch      : %d layers x %d units ; seq length: %d" % (num_layers, num_hidden, seq_length))
print("Batch size: %d" % (batch_size,))
print("Learning  : rate: %.3f ; keep_prob: %.3f" % (learning_rate, keep_prob_val))
print('-' * 10)


# Define the compute graph

from tensorflow.contrib.rnn import DropoutWrapper, BasicLSTMCell, MultiRNNCell
from tensorflow.contrib import legacy_seq2seq as seq2seq


with tf.name_scope('encode_input'):
    encode_input = [tf.placeholder(int_type, shape=(None,), name = "ei_%i" %i) for i in range(seq_length)]

with tf.name_scope('labels'):
    labels = [tf.placeholder(int_type, shape=(None,), name = "l_%i" %i) for i in range(seq_length)]

with tf.name_scope('decode_input'):
    decode_input = [tf.zeros_like(encode_input[0], dtype=int_type, name="GO")] + labels[:-1]
    
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder("float", name='keep_prob')


# In[5]:

cells = [DropoutWrapper(
        BasicLSTMCell(num_hidden), output_keep_prob=keep_prob_val
    ) for i in range(num_layers)]

stacked_lstm = MultiRNNCell(cells)

with tf.variable_scope("decoders") as scope:
    decode_outputs, decode_state = seq2seq.embedding_attention_seq2seq(encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, num_hidden, dtype=float_type)

    scope.reuse_variables()

    decode_outputs_test, decode_state_test = seq2seq.embedding_attention_seq2seq(encode_input, decode_input, stacked_lstm, vocab_size, vocab_size, num_hidden, dtype=float_type, feed_previous=True)
    

# In[6]:

with tf.name_scope('loss'):
    loss_weights = [tf.ones_like(l, dtype=float_type) for l in labels]
    loss = seq2seq.sequence_loss(decode_outputs, labels, loss_weights, vocab_size)

tf.summary.scalar('loss', loss)


config = None
if 0:
    config = tf.ConfigProto()
    #config.gpu_options.visible_device_list = "0" # (the gpu device that can be used)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5 # (the percentage of memory used)
    #config.intra_op_parallelism_threads = 2
    #config.inter_op_parallelism_threads = config.intra_op_parallelism_threads
  