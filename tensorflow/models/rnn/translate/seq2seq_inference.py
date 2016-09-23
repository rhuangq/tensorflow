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
  source_vocab = seq2seq_attention.read_qnn_vocab(FLAGS.source_vocab_file)
  target_vocab = seq2seq_attention.read_qnn_vocab(FLAGS.target_vocab_file)
  data_set = seq2seq_attention.read_eval_data(FLAGS.input_data, source_vocab, FLAGS.reverse)
  graph = seq2seq_attention.load_eval_graph(FLAGS.graph_file, seq2seq_attention.create_model_config(FLAGS))
  [source_input,
   target_input,
   encoder_cached_value,
   encoder_attention,
   decoder_init_state,
   decoder_final_state,
   tgt_gen_init_state,
   tgt_gen_final_state,
   logits] = [graph['source_input'],
              graph['target_input'],
              graph['encoder_cached_value'],
              graph['encoder_attention'],
              graph['decoder_init_state'],
              graph['decoder_final_state'],
              graph['tgt_gen_init_state'],
              graph['tgt_gen_final_state'],
              graph['logits']]
  preds = tf.reshape(tf.cast(tf.argmax(logits, 1), tf.int32), [-1, 1])


  _, tgt_id2word = target_vocab
  fout = open(FLAGS.output_file, 'w')
  with tf.Session() as session:
    for seq in data_set:
      output = []
      src = [seq]
      tgt = [[_BOS_ID]]
      feed_dict = {source_input:src, target_input:tgt}
      if encoder_attention is None:
        [decoder_state,
         tgt_gen_state,
         tgt] = session.run([decoder_final_state,
                             tgt_gen_final_state,
                             preds], feed_dict)
      else:
        [cached_value,
         attention,
         decoder_state,
         tgt_gen_state,
         tgt] = session.run([encoder_cached_value,
                              encoder_attention,
                              decoder_final_state,
                              tgt_gen_final_state,
                              preds], feed_dict)
      output.append(tgt[0][0])
      while len(output) < FLAGS.max_length and output[-1] != _EOS_ID:
        if encoder_attention is None:
          feed_dict = {target_input:tgt, decoder_init_state:decoder_state, tgt_gen_init_state:tgt_gen_state}
          [decoder_state,
           tgt_gen_state,
           tgt] = session.run([decoder_final_state, tgt_gen_final_state, preds], feed_dict)
        else:
          feed_dict = {target_input: tgt, encoder_cached_value: cached_value, decoder_init_state: decoder_state,
                       tgt_gen_init_state: tgt_gen_state}
          [attention,
           decoder_state,
           tgt_gen_state,
           tgt] = session.run([encoder_attention, decoder_final_state, tgt_gen_final_state, preds], feed_dict)

        output.append(tgt[0][0])

      out_words = [tgt_id2word[idx] for idx in output[0:-1]]
      fout.write(" ".join(out_words)+"\n")

  fout.close()


if __name__ == "__main__":
  tf.app.run()










