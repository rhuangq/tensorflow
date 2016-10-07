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

from tensorflow.models.rnn.translate import seq2seq_attention
from tensorflow.models.rnn.translate import data_utils_qnn
_buckets = data_utils_qnn._buckets
_EOS_ID = data_utils_qnn._EOS_ID
_BOS_ID = data_utils_qnn._BOS_ID
_UNK_ID = data_utils_qnn._UNK_ID


logging = tf.logging


tf.app.flags.DEFINE_string("graph_file", None, "the model file, must set")
tf.app.flags.DEFINE_boolean("has_attention", True, "the graph has attention model")
tf.app.flags.DEFINE_boolean("has_gen_layers", True, "the graph has RNN in the target generation (not the decoder RNN)")
tf.app.flags.DEFINE_string("source_vocab_file", None, "the source vocab file (from QNN), must set")
tf.app.flags.DEFINE_string("target_vocab_file", None, "the target vocab file (from QNN), must set")
tf.app.flags.DEFINE_string("input_data", None, "the source data file, one sequence per line, pre-numericized, must set")
tf.app.flags.DEFINE_string("output_file", None, "the generated data file, one sequence per line, must set")
tf.app.flags.DEFINE_integer("max_length", 100, "the maximum sequence length, longer sequence will be discarded")
tf.app.flags.DEFINE_boolean("reverse", True, "reverse the source sequence")

FLAGS = tf.app.flags.FLAGS


def main(_):
  seq2seq_attention.inference(FLAGS)

if __name__ == "__main__":
  tf.app.run()










