# skt
Sanskrit compound segmentation using seq2seq model 
Code for the paper titled 'Building a Word Segmenter for Sanskrit Overnight'

Instructions
============

Pre-requisites
--------------
The following python packages are required to be installed:
* Tensorflow: https://www.tensorflow.org/

File organization
-------------------------------------------
* Data is located in data/.
* Logs generated by tensorflow summarywriter is stored in logs/.
* Models which are trained are stored in models/.

Training
--------
The file train.py can be used to train the model.
The file test.py can be used to test the model.