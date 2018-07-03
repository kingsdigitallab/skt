
# coding: utf-8

# In[1]:

def train():

    import numpy as np
    import tensorflow as tf
    import random
    from tqdm import tqdm # ProgressBar for loops
    from utils.data_loader import pt, load_dict, save_dict
                                                                                                                
    # from tensorflow.python.ops import rnn_cell, seq2seq

    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"]=""

    # In[2]:

    import arch
    from arch import (
        model_name, num_train_batches, data_loader, 
        encode_input, seq_length, labels, keep_prob_val,
        loss, num_valid_batches, learning_rate, keep_prob,
        save_model,
    )

    # In[7]:

    pt('model defined')

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    pt('training defined')

    num_epochs = 200
    #num_epochs = 1
    verbose = 20      # Display every <verbose> epochs

    # In[8]:

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    pt('initialised')

    config = arch.config

    sess = tf.InteractiveSession(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('logs/' + model_name , sess.graph)

    pt('session created')

    model_path = 'models/' + model_name

    sess.run(init)
    import os
    try:
        if save_model:
            saver.restore(sess, model_path)
    except tf.errors.NotFoundError:
        pass

    training_context = {
        'epoch': 0,
    }
    training_context_path = '%s_training_context.json' % (model_path)
    if save_model:
        training_context = load_dict(training_context_path, training_context)
    print(training_context)

    # In[9]:
    train_losses = [0] * num_train_batches
    valid_losses = [0] * num_valid_batches

    def get_feed_from_bath(data_type):
        batch_inp, batch_outp = data_loader.next_batch(data_type=data_type)

        ret = {encode_input[t]: batch_inp[t] for t in range(seq_length)}
        ret.update({labels[t]: batch_outp[t] for t in range(seq_length)})
        ret[keep_prob] = keep_prob_val if data_type == 'train' else 1.0

        return ret

    step = 0
    try:
        while training_context['epoch'] < num_epochs:

            # Training on train set
            for i in tqdm(range(num_train_batches)):
                input_dict = get_feed_from_bath('train')

                _, loss_val, summary = sess.run([train, loss, merged], feed_dict=input_dict)
                train_losses[i] = loss_val

                summary_writer.add_summary(summary, step)
                step += 1

            print('Epoch: %6i ; Train Loss: %.3f' % (training_context['epoch'], np.mean(train_losses)))

            if (training_context['epoch'] == num_epochs - 1) or (training_context['epoch'] + 1) % verbose == 0:
                # Testing on valid set
                pt('validate')
                for i in range(num_valid_batches):
                    input_dict = get_feed_from_bath('valid')

                    loss_val = sess.run(loss, feed_dict=input_dict)
                    valid_losses[i] = loss_val

                log_txt = "Epoch: " + str(training_context['epoch']) +\
                        " ; Steps: " + str(step) +\
                        " ; train_loss: " + str(round(np.mean(train_losses),4)) + '+' + str(round(np.std(train_losses),2)) +\
                        " ; valid_loss: " + str(round(np.mean(valid_losses),4)) + '+' + str(round(np.std(valid_losses),2)) 
                print(log_txt)

                f = open('log.txt', 'a')
                f.write(log_txt + '\n')
                f.close()

                # Saving takes ~20s
                if save_model:
                    pt('saving...')
                    saver.save(sess, model_path)
                    pt('saved')
                    save_dict(training_context_path, training_context)
                            
            training_context['epoch'] += 1

    except KeyboardInterrupt:
        print("Stopped at epoch: " + str(training_context['epoch']) + ' and step: ' + str(step))

    print("Training completed")

if __name__ == '__main__':
    train()
