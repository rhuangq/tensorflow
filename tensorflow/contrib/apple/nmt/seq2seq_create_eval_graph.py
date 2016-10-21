# Copyright (c) 2016 Apple Inc. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
from tensorflow.contrib.apple.nmt import seq2seq_attention

logging = tf.logging

#models
tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Batch size to use during evaluation.")
tf.app.flags.DEFINE_integer("embed_size", 128, "Size of embedding vector.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of RNN layers in the encoder-decoder model.")
tf.app.flags.DEFINE_integer("num_gen_layers", 3, "Number of RNN layers in the target generation model.")
tf.app.flags.DEFINE_integer("source_vocab_size", 100, "source vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 100, "target vocabulary size.")
tf.app.flags.DEFINE_string("rnn_type", 'lstm', "RNN types, one of 'gru', 'lstm', 'lstm_ph' (peep hole), 'lstm_ln' (layer normalized).")
tf.app.flags.DEFINE_boolean("use_birnn", True, "use BiRNN in the encoder")
tf.app.flags.DEFINE_float("keep_rate", 1.0, "value less than 1 will turn on dropouts")
tf.app.flags.DEFINE_integer("attention_type", 1, "attention type to use. 0: basic encoder-decoder; 1: global attention; 2: recurrent global attention")
tf.app.flags.DEFINE_integer("attention_dim", 256, "attention vector dimension when using attention.")
tf.app.flags.DEFINE_integer("readout_dim", 256, "the output dimension for target generation")

#data
tf.app.flags.DEFINE_string("model_ckpt", None, "the ckpt model file, must set")
tf.app.flags.DEFINE_string("output_file", None, "the output graph file, must set")

FLAGS = tf.app.flags.FLAGS

def main(_):
  config = seq2seq_attention.create_model_config(FLAGS)
  seq2seq_attention.create_eval_graph(FLAGS.model_ckpt, config, FLAGS.output_file)

if __name__ == "__main__":
  tf.app.run()



