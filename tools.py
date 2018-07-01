#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

path_src = 'data8k/original'
path_rec = 'data8k/recons'
path_data = 'data'
data_names = 'dcs_data_output_train_sent.txt dcs_data_input_train_sent.txt dcs_data_input_test_sent.txt dcs_data_output_test_sent.txt'

def show_help():
    print('''
tools.py ACTION [ARG1] [ARG2] ...

ACTION:
  recons   : reconstruct the original plain text dataset
    from %s into %s
  build-voc: build a vocabulary and sentencepiece model
    ARG1 = vocabulary size e.g. '6k'
  tokenise : tokenise the dataset using a vocabulary 
    ARG1 = vocabulary size e.g. '6k'
  select   : select a different, pre-built vocabulary
    ARG1 = '4k' or '6k' or '8k'
  rm-model : remove the model
  rm-npy   : remove the numpy data files (can be regenerated)
''' % (path_src, path_rec))

import sentencepiece as spm
import sys, os

action_is_valid = False
action = None
args = []
if len(sys.argv) > 1:
    action = sys.argv[1]

    if len(sys.argv) > 2:
        args = sys.argv[2:]

if action == 'train':
    action_is_valid = True
    import train
    train.train()

if action == 'rm-model':
    action_is_valid = True
    os.system('rm models/*')

if action == 'rm-npy':
    action_is_valid = True
    os.system('rm %s/*npy' % path_data)

if action == 'select':
    action_is_valid = True
    if args:
        dir_name = 'data%s' % args[0]
        if os.path.exists(dir_name):
            if os.path.exists('data'):
                os.system('unlink data')
            os.system('ln -s data%s data' % args[0])
        else:
            print('%s not found' % dir_name)
    else:
        print('Missing argument. Please pass only the the following vocabulary sizes:')
        os.system('''ls | grep 'data.*k' | sed s/data//''')

if action == 'recons':
    action_is_valid = True
    for fn in data_names.split():
        with open(path_src+'/'+fn, 'rt') as f:
            content = f.read()
            content = content.replace(' ', '').replace('▁', ' ')

            with open(path_rec+'/'+fn, 'wt') as f2:
                f2.write(content)
    
    print('Files created in %s' % path_rec)

    # https://github.com/cltk/sanskrit_text_dcs.git

if action == 'build-voc':
    tmp_input_path = 'voc-input.tmp'
    os.system('cat %s/%s %s/%s > %s' % (path_rec, data_names.split()[0], path_rec, data_names.split()[1], tmp_input_path))

    action_is_valid = True
    if args and args[0] != '8k':
        size = int(re.sub('\D', '', args[0])) * 1000
        path_out = 'data' + args[0]

        if not os.path.exists(path_out):
            os.makedirs(path_out)

        spm.SentencePieceTrainer.Train('--input={} --model_prefix={}/m --vocab_size={}'.format(tmp_input_path, path_out, size))
    else:
        print('ERROR: please pass a vocabulary size. E.g. 10k, 4k, 6k. 8k not allowed!')

if action == 'tokenise':
    action_is_valid = True

    dir_name = 'data'
    if args and args[0] != '8k':
        dir_name += args[0]

    s = spm.SentencePieceProcessor()
    s.Load('{}/m.model'.format(dir_name))  

    for fn in data_names.split():
        input_path = path_rec+'/'+fn
        print(input_path)
        with open(input_path, 'rt') as f:
            with open(dir_name+'/'+fn, 'wt') as f2:
                for line in f:
                    f2.write(' '.join([str(st) for st in s.SampleEncodeAsPieces(line, -1, 0.1)]) + '\n')

if not action_is_valid:
   show_help() 

if action_is_valid:
    print('done')


