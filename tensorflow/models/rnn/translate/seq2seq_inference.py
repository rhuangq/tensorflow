"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import sys
import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

#TODO: make it into regular TF lib?
import seq2seq_attention
import data_utils_qnn
_buckets = data_utils_qnn._buckets
_EOS_ID = data_utils_qnn._EOS_ID
_BOS_ID = data_utils_qnn._BOS_ID
_UNK_ID = data_utils_qnn._UNK_ID


logging = tf.logging


tf.app.flags.DEFINE_string("graph_file", None, "the model file, must set")
tf.app.flags.DEFINE_string("source_vocab_file", None, "the source vocab file (from QNN), must set")
tf.app.flags.DEFINE_string("target_vocab_file", None, "the target vocab file (from QNN), must set")
tf.app.flags.DEFINE_string("input_data", None, "the source data file, one sequence per line, pre-numericized, must set")
tf.app.flags.DEFINE_string("output_file", None, "the generated data file, one sequence per line, must set")
tf.app.flags.DEFINE_integer("max_length", 60, "the maximum sequence length, longer sequence will be discarded")
tf.app.flags.DEFINE_boolean("reverse", True, "reverse the source sequence")
tf.app.flags.DEFINE_integer("embed_size", 128, "Size of embedding vector.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_vocab_size", 100, "source vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 100, "target vocabulary size.")
tf.app.flags.DEFINE_boolean("use_lstm", True, "Use LSTM or GRU as the RNN layers")
tf.app.flags.DEFINE_boolean("use_birnn", True, "use BiRNN in the encoder")
tf.app.flags.DEFINE_float("keep_rate", 1.0, "value less than 1 will turn on dropouts")
tf.app.flags.DEFINE_integer("num_samples", 512, "number of samples used in importance sampling, use 0 to turn it off.")
tf.app.flags.DEFINE_integer("attention_type", 1, "attention type to use. 0: basic encoder-decoder; 1: global attention; 2: recurrent global attention")

FLAGS = tf.app.flags.FLAGS


def main(_):
  seq2seq_attention.inference(FLAGS)

if __name__ == "__main__":
  tf.app.run()










