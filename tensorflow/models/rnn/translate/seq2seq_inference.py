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
from tensorflow.models.rnn.translate import seq2seq_attention
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
tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Batch size to use during inference.")
tf.app.flags.DEFINE_integer("max_length", 60, "the maximum sequence length, longer sequence will be discarded")

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


def read_dev_data(source_path, target_path, max_len=1000):
  """read development data (usually pretty small)"""
  data_set = []
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        data_set.append([source_ids, target_ids])
        source, target = source_file.readline(), target_file.readline()

  #sort the data by length, by target and source
  data_set.sort(key=lambda x:len(x[1])*max_len+len(x[0]))
  return data_set


def load_eval_graph(graph_file, config):
  config.mode = 2
  graph_def = tf.GraphDef()
  with open(graph_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name="")
  with tf.variable_scope("TranslationModel"):
    model_graph = seq2seq_attention.create_nmt_graph(config)

  return model_graph

def main(_):
  
  graph = load_eval_graph(FLAGS.graph_file, seq2seq_attention.create_model_config())


if __name__ == "__main__":
  tf.app.run()










