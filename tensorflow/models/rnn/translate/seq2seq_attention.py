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

#TODO: bazel it so others can import it
import data_utils_qnn
_buckets = data_utils_qnn._buckets
_EOS_ID = data_utils_qnn._EOS_ID
_BOS_ID = data_utils_qnn._BOS_ID
_UNK_ID = data_utils_qnn._UNK_ID


def _create_rnn_multi_cell(use_lstm, num_cells, num_layers, keep_rate):
  """a helper function creates multi-layer RNNs"""

  if use_lstm:
    cell = tf.nn.rnn_cell.LSTMCell(num_cells)
  else:
    cell = tf.nn.rnn_cell.GRUCell(num_cells)
  if keep_rate != 1.0:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_rate)
  if num_layers > 1:
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
  if keep_rate != 1.0:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_rate)

  return cell

class SequenceLength(object):
  """a helper class computes the sequence length and assign weights
  `data` has shape [batch_size, ?]. This class is to create length
  tensor and weight tensor about the actual length and corresponding weight"""

  def __init__(self, batch_size, max_length, filler):
    self.batch_size = batch_size
    self.max_length = max_length
    self.mapper = tf.reshape(tf.convert_to_tensor([filler] * (batch_size*max_length), dtype=tf.int32), [batch_size, max_length])

  def get_length(self, data):
    """compute the sequence length."""

    shape = data.get_shape()
    if shape[0] != self.batch_size or shape[1] > self.max_length:
      raise ValueError("the input data has shape %s, does not match a shape (%d, <=%d)" %(str(shape), self.batch_size, self.max_length))

    return tf.reduce_sum(tf.cast(tf.not_equal(data, tf.slice(self.mapper, [0, 0], [-1, tf.shape(data)[1]])), dtype=tf.int32), 1)

  def get_weight(self, data):
    """assign weight to the sequence, assign zero weight to filler symbols"""

    shape = data.get_shape()
    if shape[0] != self.batch_size or shape[1]> self.max_length:
      raise ValueError("the input data has shape %s, does not match a shape (%d, <=%d)" % (str(shape), self.batch_size, self.max_length))

    return tf.cast(tf.not_equal(data, tf.slice(self.mapper, [0, 0], [-1, tf.shape(data)[1]])), dtype=tf.float32)


def Seq2SeqRnn(
               inputs,
               num_cells,
               num_layers,
               use_lstm=False,
               use_birnn=False,
               seq_lengths=None,
              init_state=None,
               keep_rate=0.5,
               initializer=None,
               scope=None,
               dtype=tf.float32):
  """
  this function create multi layer RNN (including bidirectional), and run the RNN computation.
  """

  batch_size = inputs.get_shape()[0]
  with vs.variable_scope(scope or "Seq2SeqRnn", initializer=initializer):
    init_state_forward = None
    init_state_backward = None
    if init_state:
      if use_birnn:
        init_state_forward, init_state_backward = init_state
      else:
        init_state_forward = init_state

    forward_cell = _create_rnn_multi_cell(use_lstm, num_cells, num_layers, keep_rate)
    if init_state_forward is None:
      init_state_forward = forward_cell.zero_state(batch_size, dtype)
    if use_birnn:
      backward_cell = _create_rnn_multi_cell(use_lstm, num_cells, num_layers, keep_rate)
      if init_state_backward is None:
        init_state_backward = backward_cell.zero_state(batch_size, dtype)
      outputs, final_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                             inputs, tf.cast(seq_lengths, tf.int64),
                                                             initial_state_fw=init_state_forward,
                                                             initial_state_bw=init_state_backward,
                                                             dtype=dtype, time_major=False)
      outputs = tf.concat(2, outputs)
    else:
      outputs, final_state = tf.nn.dynamic_rnn(forward_cell, inputs, seq_lengths,
                                               initial_state=init_state_forward, dtype=dtype, time_major=False)

    return outputs, final_state

def GlobalAttention(
    batch_size,
    src_dim,
    tgt_dim,
    source_inputs,
    target_inputs,
    out_dim,
    src_seq_lengths,
    tgt_seq_lengths,
    src_values,
    dot_dim,
    single_step=False,
    initializer=None,
    scope=None,
    dtype=tf.float32):
  """
  Compute the attention vector by looking at the whole source sequence and current step in target.
   This function can be used in batch or called per step.
  """

  with vs.variable_scope(scope or "GlobalAttention", initializer=initializer):
    bs, _, sd = source_inputs.get_shape()
    if bs != batch_size or sd != src_dim:
      raise ValueError("source data dimension mismatch, batch_size expected %d vs actual %d, dimention expected %d vs actual %d"
                       %(batch_size, bs, src_dim, sd))

    if not single_step:
      bs, _, sd = target_inputs.get_shape()
    else:
      bs, sd = target_inputs.get_shape()

    if bs != batch_size or sd != tgt_dim:
      raise ValueError("target data dimension mismatch, batch_size expected %d vs actual %d, dimention expected %d vs actual %d"
                       %(batch_size, bs, tgt_dim, sd))

    tgt_dot_trans_w = tf.get_variable("tgt_dot_trans_w", [tgt_dim, dot_dim], dtype=dtype)
    tgt_dot_trans_b = tf.get_variable("tgt_dot_trans_b", [dot_dim], dtype=dtype)
    tgt_add_trans_w = tf.get_variable("tgt_add_trans_w", [tgt_dim, out_dim], dtype=dtype)
    tgt_add_trans_b = tf.get_variable("tgt_add_trans_b", [out_dim], dtype=dtype)

    src_dot_values, src_add_values = src_values

    tgt_input_rs = tf.reshape(target_inputs, [-1, tgt_dim])
    tgt_dot_values = tf.reshape(tf.add(tf.matmul(tgt_input_rs, tgt_dot_trans_w), tgt_dot_trans_b),
                                [batch_size, -1, dot_dim])
    tgt_add_values = tf.reshape(tf.add(tf.matmul(tgt_input_rs, tgt_add_trans_w), tgt_add_trans_b),
                                [batch_size, -1, out_dim])

    if not single_step:
      out_padding = tf.zeros([tf.shape(target_inputs)[1], out_dim])
    outputs = None
    weights = []
    for batch_idx in xrange(batch_size):
      #extract one sequence
      src_dot_value = tf.squeeze(tf.slice(src_dot_values, [batch_idx, 0, 0], [1, src_seq_lengths[batch_idx], -1]), [0])
      src_add_value = tf.squeeze(tf.slice(src_add_values, [batch_idx, 0, 0], [1, src_seq_lengths[batch_idx], -1]), [0])

      if not single_step:
        tgt_dot_value = tf.squeeze(tf.slice(tgt_dot_values, [batch_idx, 0, 0], [1, tgt_seq_lengths[batch_idx], -1]), [0])
        tgt_add_value = tf.squeeze(tf.slice(tgt_add_values, [batch_idx, 0, 0], [1, tgt_seq_lengths[batch_idx], -1]), [0])
      else:
        tgt_dot_value = tf.squeeze(tf.slice(tgt_dot_values, [batch_idx, 0, 0], [1, 1, -1]), [0])
        tgt_add_value = tf.squeeze(tf.slice(tgt_add_values, [batch_idx, 0, 0], [1, 1, -1]), [0])

      context_weight = tf.nn.softmax(tf.matmul(tgt_dot_value, src_dot_value, transpose_b=True)) # dim (tgt_len, src_len)
      context = tf.tanh(tf.add(tf.matmul(context_weight, src_add_value), tgt_add_value)) #dim (tgt_len, out_dim)
      if not single_step:
        padding = tf.slice(out_padding, [tgt_seq_lengths[batch_idx], 0], [-1, -1])
        if outputs is None:
          outputs = tf.concat(0, [context, padding])
        else:
          outputs = tf.concat(0, [outputs, context, padding])
      else:
        if outputs is None:
          outputs = context
        else:
          outputs = tf.concat(0, [outputs, context])
      weights.append(context_weight)

  if single_step:
    return outputs, weights
  else:
    return tf.reshape(outputs, [batch_size, -1, out_dim]), weights


class ModelConfig(dict):
  #TODO: is there an easier way to sync this with FLAGS?
  def __init__(self):
    self.batch_size = 64
    self.embed_size = 128
    self.num_layers = 3
    self.use_lstm = True
    self.use_birnn = False
    self.source_vocab_size = 100
    self.target_vocab_size = 100
    self.initial_weight = 0.1
    self.initial_learning_rate = 0.5
    self.learning_rate_decay = 0.8
    self.keep_rate = 1.0
    self.dtype = tf.float32
    self.max_length = 32
    self.max_grad_norm = 5.0
    self.filler = _EOS_ID

    # enum is only available in python3, otherwise it should be used here.
    # 0: training; 1: evaluation; 2: inference
    self.mode = 0
    # 0: simple encoder-decoder; 1: global attention; 2: recurrent global attention
    self.attention_type = 0
    self.attention_dim = 256

    self['batch_size'] = self.batch_size
    self['embed_size'] = self.embed_size
    self['num_layers'] = self.num_layers
    self['use_lstm'] = self.use_lstm
    self['use_birnn'] = self.use_birnn
    self['source_vocab_size'] = self.source_vocab_size
    self['target_vocab_size'] = self.target_vocab_size
    self['initial_weight']  = self.initial_weight
    self['initial_learning_rate'] = self.initial_learning_rate
    self['learning_rate_decay'] = self.learning_rate_decay
    self['keep_rate'] = self.keep_rate
    self['dtype'] = self.dtype
    self['max_length'] = self.max_length
    self['max_grad_norm'] = self.max_grad_norm
    self['filler'] = self.filler
    self['mode'] = self.mode
    self['attention_type'] = self.attention_type
    self['attention_dim'] = self.attention_dim

  def __getattr__(self, item):
    if item in self:
      return self[item]
    else:
      raise AttributeError("no such attribute " + item)

  def __setattr__(self, key, value):
    self[key] = value


def create_nmt_graph(config):
  batch_size = config.batch_size
  filler = config.filler
  source_vocab_size = config.source_vocab_size
  target_vocab_size = config.target_vocab_size
  num_cells = config.embed_size
  num_layers = config.num_layers
  dtype = config.dtype
  max_length = config.max_length
  use_lstm = config.use_lstm
  use_birnn = config.use_birnn
  keep_rate = config.keep_rate
  mode = config.mode
  attention_type = config.attention_type
  attention_dim = config.attention_dim if attention_type != 0 else num_cells
  dot_dim = num_cells

  source_input = tf.placeholder(tf.int32, [batch_size, None], name="source_input")
  target_input = tf.placeholder(tf.int32, [batch_size, None], name="target_input")

  # TODO: move embedding to CPU?
  with vs.variable_scope("input"):
    src_embedding = tf.get_variable("source_embedding", [source_vocab_size, num_cells], dtype=dtype)
    tgt_embedding = tf.get_variable("target_embedding", [target_vocab_size, num_cells], dtype=dtype)
  s2s_src_input = tf.nn.embedding_lookup(src_embedding, source_input)
  s2s_tgt_input = tf.nn.embedding_lookup(tgt_embedding, target_input)

  seq_length_op = SequenceLength(batch_size, max_length, filler)
  src_length = seq_length_op.get_length(source_input)
  tgt_length = seq_length_op.get_length(target_input)

  encoder_init_state = None

  with vs.variable_scope("encoder") as varscope:
    encoder_outputs, encoder_final_state = Seq2SeqRnn(s2s_src_input,
                                                      num_cells,
                                                      num_layers,
                                                      use_lstm=use_lstm,
                                                      use_birnn=use_birnn,
                                                      seq_lengths=src_length,
                                                      init_state=encoder_init_state,
                                                      scope=varscope,
                                                      keep_rate=1.0 if mode != 0 else keep_rate)
    src_dot_values, src_add_values = None, None
    if attention_type == 1:
      src_dim = num_cells * 2 if use_birnn else num_cells
      src_dot_trans_w = tf.get_variable("src_dot_trans_w", [src_dim, dot_dim], dtype=dtype)
      src_dot_trans_b = tf.get_variable("src_dot_trans_b", [dot_dim], dtype=dtype)
      src_add_trans_w = tf.get_variable("src_add_trans_w", [src_dim, attention_dim], dtype=dtype)
      src_add_trans_b = tf.get_variable("src_add_trans_b", [attention_dim], dtype=dtype)
      src_input_rs = tf.reshape(encoder_outputs, [-1, src_dim])
      src_dot_values = tf.reshape(tf.add(tf.matmul(src_input_rs, src_dot_trans_w), src_dot_trans_b),
                                  [batch_size, -1, dot_dim])
      src_add_values = tf.reshape(tf.add(tf.matmul(src_input_rs, src_add_trans_w), src_add_trans_b),
                                  [batch_size, -1, attention_dim])

  if use_birnn:
    decoder_init_state, _ = encoder_final_state
  else:
    decoder_init_state = encoder_final_state

  with vs.variable_scope("decoder") as varscope:
    decoder_outputs, decoder_final_state = Seq2SeqRnn(s2s_tgt_input,
                                                      num_cells,
                                                      num_layers,
                                                      use_lstm=use_lstm,
                                                      use_birnn=False,
                                                      seq_lengths=tgt_length,
                                                      init_state=decoder_init_state,
                                                      scope=varscope,
                                                      keep_rate=1.0 if mode != 0 else keep_rate)

  if attention_type == 1:
    s2s_outputs, _ = GlobalAttention(batch_size,
                                     num_cells * 2 if use_birnn else num_cells,
                                     num_cells,
                                     encoder_outputs,
                                     decoder_outputs,
                                     attention_dim,
                                     src_length,
                                     tgt_length,
                                     src_values=(src_dot_values, src_add_values),
                                     dot_dim=dot_dim,
                                     single_step=False)
  elif attention_type == 0:
    s2s_outputs = decoder_outputs
  else:
    raise ValueError("attention type %d is not implemented yet" % (attention_type))

  with vs.variable_scope("output"):
    tgt_gen_cell = _create_rnn_multi_cell(use_lstm, attention_dim, num_layers,
                                          keep_rate=1.0 if mode != 0 else config.keep_rate)
    tgt_gen_init_state = None
    tgt_gen_outputs, tgt_gen_final_state = tf.nn.dynamic_rnn(tgt_gen_cell, s2s_outputs, tgt_length,
                                                             initial_state=tgt_gen_init_state, dtype=dtype,
                                                             time_major=False)
    output = tf.reshape(tgt_gen_outputs, [-1, attention_dim])
    softmax_w = tf.get_variable("softmax_w", [attention_dim, target_vocab_size], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [target_vocab_size], dtype=dtype)
    logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name="prediction")

  return dict(
    source_input=source_input,
    target_input=target_input,
    seq_length_op=seq_length_op,
    logits=logits,
  )

def create_recurrent_attention_graph(config):
  batch_size = config.batch_size
  filler = config.filler
  source_vocab_size = config.source_vocab_size
  target_vocab_size = config.target_vocab_size
  num_cells = config.embed_size
  num_layers = config.num_layers
  dtype = config.dtype
  max_length = config.max_length
  use_lstm = config.use_lstm
  use_birnn = config.use_birnn
  keep_rate = config.keep_rate
  mode = config.mode
  attention_type = config.attention_type
  attention_dim = config.attention_dim if attention_type != 0 else num_cells
  dot_dim = num_cells

  source_input = tf.placeholder(tf.int32, [batch_size, None], name="source_input")
  target_input = tf.placeholder(tf.int32, [batch_size, None], name="target_input")

  # TODO: move embedding to CPU?
  with vs.variable_scope("input"):
    src_embedding = tf.get_variable("source_embedding", [source_vocab_size, num_cells], dtype=dtype)
    tgt_embedding = tf.get_variable("target_embedding", [target_vocab_size, num_cells], dtype=dtype)
  s2s_src_input = tf.nn.embedding_lookup(src_embedding, source_input)
  s2s_tgt_input = tf.nn.embedding_lookup(tgt_embedding, target_input)

  seq_length_op = SequenceLength(batch_size, max_length, filler)
  src_length = seq_length_op.get_length(source_input)
  tgt_length = seq_length_op.get_length(target_input)

  encoder_init_state = None

  with vs.variable_scope("encoder") as varscope:
    encoder_outputs, encoder_final_state = Seq2SeqRnn(s2s_src_input,
                                                      num_cells,
                                                      num_layers,
                                                      use_lstm=use_lstm,
                                                      use_birnn=use_birnn,
                                                      seq_lengths=src_length,
                                                      init_state=encoder_init_state,
                                                      scope=varscope,
                                                      keep_rate=1.0 if mode != 0 else keep_rate)
    src_dot_values, src_add_values = None, None
    if attention_type == 1:
      src_dim = num_cells * 2 if use_birnn else num_cells
      src_dot_trans_w = tf.get_variable("src_dot_trans_w", [src_dim, dot_dim], dtype=dtype)
      src_dot_trans_b = tf.get_variable("src_dot_trans_b", [dot_dim], dtype=dtype)
      src_add_trans_w = tf.get_variable("src_add_trans_w", [src_dim, attention_dim], dtype=dtype)
      src_add_trans_b = tf.get_variable("src_add_trans_b", [attention_dim], dtype=dtype)
      src_input_rs = tf.reshape(encoder_outputs, [-1, src_dim])
      src_dot_values = tf.reshape(tf.add(tf.matmul(src_input_rs, src_dot_trans_w), src_dot_trans_b),
                                  [batch_size, -1, dot_dim])
      src_add_values = tf.reshape(tf.add(tf.matmul(src_input_rs, src_add_trans_w), src_add_trans_b),
                                  [batch_size, -1, attention_dim])

  if use_birnn:
    decoder_init_state, _ = encoder_final_state
  else:
    decoder_init_state = encoder_final_state

  with vs.variable_scope("decoder") as varscope:
    decoder_outputs, decoder_final_state = Seq2SeqRnn(s2s_tgt_input,
                                                      num_cells,
                                                      num_layers,
                                                      use_lstm=use_lstm,
                                                      use_birnn=False,
                                                      seq_lengths=tgt_length,
                                                      init_state=decoder_init_state,
                                                      scope=varscope,
                                                      keep_rate=1.0 if mode != 0 else keep_rate)

  if attention_type == 1:
    s2s_outputs, _ = GlobalAttention(batch_size,
                                     num_cells * 2 if use_birnn else num_cells,
                                     num_cells,
                                     encoder_outputs,
                                     decoder_outputs,
                                     attention_dim,
                                     src_length,
                                     tgt_length,
                                     src_values=(src_dot_values, src_add_values),
                                     dot_dim=dot_dim,
                                     single_step=False)
  elif attention_type == 0:
    s2s_outputs = decoder_outputs
  else:
    raise ValueError("attention type %d is not implemented yet" % (attention_type))

  with vs.variable_scope("output"):
    tgt_gen_cell = _create_rnn_multi_cell(use_lstm, attention_dim, num_layers,
                                          keep_rate=1.0 if mode != 0 else config.keep_rate)
    tgt_gen_init_state = None
    tgt_gen_outputs, tgt_gen_final_state = tf.nn.dynamic_rnn(tgt_gen_cell, s2s_outputs, tgt_length,
                                                             initial_state=tgt_gen_init_state, dtype=dtype,
                                                             time_major=False)
    output = tf.reshape(tgt_gen_outputs, [-1, attention_dim])
    softmax_w = tf.get_variable("softmax_w", [attention_dim, target_vocab_size], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [target_vocab_size], dtype=dtype)
    logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name="prediction")

  return dict(
    source_input=source_input,
    target_input=target_input,
    seq_length_op=seq_length_op,
    logits=logits,
  )

def _create_encoder_eval_graph(config):
  """no padding"""
  batch_size = config.batch_size
  source_vocab_size = config.source_vocab_size
  num_cells = config.embed_size
  num_layers = config.num_layers
  dtype = config.dtype
  use_lstm = config.use_lstm
  use_birnn = config.use_birnn
  keep_rate = 1.0
  attention_type = config.attention_type
  attention_dim = config.attention_dim if attention_type != 0 else num_cells
  dot_dim = num_cells

  source_input = tf.placeholder(tf.int32, [batch_size, None], name="__QNNI__source_input")

  with vs.variable_scope("input"):
    src_embedding = tf.get_variable("source_embedding", [source_vocab_size, num_cells], dtype=dtype)

  s2s_src_input = tf.nn.embedding_lookup(src_embedding, source_input)
  seq_lengths = [tf.shape(source_input)[1]] * batch_size

  encoder_init_state = None
  with vs.variable_scope("encoder") as varscope:
    encoder_outputs, encoder_final_state = Seq2SeqRnn(s2s_src_input,
                                                      num_cells,
                                                      num_layers,
                                                      use_lstm=use_lstm,
                                                      use_birnn=use_birnn,
                                                      seq_lengths=seq_lengths,
                                                      init_state=encoder_init_state,
                                                      scope=varscope,
                                                      keep_rate=keep_rate)
    encoder_outputs = tf.identity(encoder_outputs, name="__QNNO__encoder_output")
    if use_birnn:
      final_state, _ = encoder_final_state
    else:
      final_state = encoder_final_state
    encoder_final_state = tf.identity(final_state, name="__QNNO__encoder_final_state")
    if attention_type == 1:
      src_dim = num_cells * 2 if use_birnn else num_cells
      src_dot_trans_w = tf.get_variable("src_dot_trans_w", [src_dim, dot_dim], dtype=dtype)
      src_dot_trans_b = tf.get_variable("src_dot_trans_b", [dot_dim], dtype=dtype)
      src_add_trans_w = tf.get_variable("src_add_trans_w", [src_dim, attention_dim], dtype=dtype)
      src_add_trans_b = tf.get_variable("src_add_trans_b", [attention_dim], dtype=dtype)
      src_input_rs = tf.reshape(encoder_outputs, [-1, src_dim])
      src_dot_values = tf.reshape(tf.add(tf.matmul(src_input_rs, src_dot_trans_w), src_dot_trans_b),
                                  [batch_size, -1, dot_dim], name="__QNNO__encoder_dot_value")
      src_add_values = tf.reshape(tf.add(tf.matmul(src_input_rs, src_add_trans_w), src_add_trans_b),
                                  [batch_size, -1, attention_dim], name="__QNNO__encoder_add_value")

def _create_decoder_eval_graph(config):
  batch_size = config.batch_size
  target_vocab_size = config.target_vocab_size
  num_cells = config.embed_size
  num_layers = config.num_layers
  dtype = config.dtype
  use_lstm = config.use_lstm
  use_birnn = config.use_birnn
  keep_rate = 1.0
  attention_type = config.attention_type
  attention_dim = config.attention_dim if attention_type != 0 else num_cells
  dot_dim = num_cells

  target_input = tf.placeholder(tf.int32, [batch_size], name="__QNNI__target_input")

  with vs.variable_scope("input"):
    tgt_embedding = tf.get_variable("target_embedding", [target_vocab_size, num_cells], dtype=dtype)
  s2s_tgt_input = tf.nn.embedding_lookup(tgt_embedding, target_input)

  with vs.variable_scope("decoder"):
    with vs.variable_scope("RNN"):
      decoder_cell = _create_rnn_multi_cell(use_lstm, num_cells, num_layers, keep_rate)
      if use_lstm:
        state_size = [num_layers, 2, batch_size, num_cells]
      else:
        state_size = [num_layers, batch_size, num_cells]
      decoder_init_state_ph = tf.placeholder_with_default(tf.zeros(state_size), state_size, name="__QNNI__decoder_state")
      if use_lstm:
        decoder_init_state = tuple(tuple(tf.unpack(i, axis=0)) for i in tuple(tf.unpack(decoder_init_state_ph, axis=0)))
      else:
        decoder_init_state = tuple(tf.unpack(decoder_init_state_ph, axis=0))
      decoder_output, decoder_final_state = decoder_cell(s2s_tgt_input, decoder_init_state)
      decoder_final_state = tf.identity(decoder_final_state, name="__QNNO__decoder_state")

  attentions = None
  if attention_type == 1:
    src_dim = num_cells * 2 if use_birnn else num_cells
    encoder_outputs = tf.placeholder(dtype, [batch_size, None, src_dim], name="__QNNI__encoder_output")
    src_dot_values = tf.placeholder(dtype, [batch_size, None, dot_dim], name="__QNNI__encoder_dot_value")
    src_add_values = tf.placeholder(dtype, [batch_size, None, attention_dim], name="__QNNI__encoder_add_value")
    src_length = [tf.shape(encoder_outputs)[1]] * batch_size
    tgt_length = None
    s2s_outputs, attentions = GlobalAttention(batch_size,
                                              src_dim,
                                              num_cells,
                                              encoder_outputs,
                                              decoder_output,
                                              attention_dim,
                                              src_length,
                                              tgt_length,
                                              src_values=(src_dot_values, src_add_values),
                                              dot_dim=dot_dim,
                                              single_step=True)
    attentions = tf.identity(attentions, name="__QNNO__attention")
  elif attention_type == 0:
    s2s_outputs = decoder_output
  else:
    raise ValueError("attention type %d is not implemented yet" % (attention_type))

  with vs.variable_scope("output"):
    with vs.variable_scope("RNN"):
      tgt_gen_cell = _create_rnn_multi_cell(use_lstm, attention_dim, num_layers, keep_rate=keep_rate)
      if use_lstm:
        state_size = [num_layers, 2, batch_size, attention_dim]
      else:
        state_size = [num_layers, batch_size, attention_dim]
      tgt_gen_init_state_ph = tf.placeholder_with_default(tf.zeros(state_size), state_size, name="__QNNI__target_gen_state")
      if use_lstm:
        tgt_gen_init_state = tuple(tuple(tf.unpack(i, axis=0)) for i in tuple(tf.unpack(tgt_gen_init_state_ph, axis=0)))
      else:
        tgt_gen_init_state = tuple(tf.unpack(tgt_gen_init_state_ph, axis=0))
      tgt_gen_output, tgt_gen_final_state = tgt_gen_cell(s2s_outputs, tgt_gen_init_state)

    output = tf.reshape(tgt_gen_output, [-1, attention_dim])
    softmax_w = tf.get_variable("softmax_w", [attention_dim, target_vocab_size], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [target_vocab_size], dtype=dtype)
    logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name="__QNNO__prediction")
    tgt_gen_final_state = tf.identity(tgt_gen_final_state, name="__QNNO__target_gen_state")


class TranslationModel(object):
  """The NMT model"""

  def __init__(self, config):
    self.batch_size = config.batch_size
    self.target_vocab_size = config.target_vocab_size
    dtype = config.dtype
    self.graph = create_nmt_graph(config)
    is_training = True if config.mode == 0 else False
    self.source_input = self.graph['source_input']
    self.target_input = self.graph['target_input']
    seq_length_op = self.graph['seq_length_op']
    logits = self.graph['logits']

    pad_batch = tf.constant(config.filler, tf.int32, [self.batch_size, 1])
    target_output = tf.concat(1, [tf.slice(self.target_input, [0, 1], [-1, -1]), pad_batch])

    tgt_weight = tf.reshape(seq_length_op.get_weight(self.target_input), [-1])
    tgt_output = tf.reshape(target_output, [-1])
    cross_entropy = nn_ops.sparse_softmax_cross_entropy_with_logits(logits, tgt_output) * tgt_weight
    self._batch_weight = tf.reduce_sum(tgt_weight)
    self._cost = tf.reduce_sum(cross_entropy)
    self.total_cost = 0.0
    self.total_weight = 0
    self.total_accuracy = 0
    self.global_step = tf.Variable(0, trainable=False)
    self.is_training = is_training
    self._accuracy = tf.no_op()

    if not is_training:
      self._accuracy = tf.reduce_sum(tf.cast(tf.equal(tgt_output, tf.cast(tf.argmax(logits, 1), tf.int32)), tf.float32) * tgt_weight)
      self._eval_op = self.global_step.assign(self.global_step + 1)
      return

    self.learning_rate = tf.Variable(
      config.initial_learning_rate, trainable=False, dtype=dtype)
    self._learning_rate_decay_op = self.learning_rate.assign(
      self.learning_rate * config.learning_rate_decay)

    max_grad_norm = config.max_grad_norm
    all_params = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost/self._batch_weight, all_params), max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self._train_op = optimizer.apply_gradients(zip(grads, all_params), global_step=self.global_step)

  def run_minibatch(self, session, source_input, target_input):
    """run one minibatch through the graph"""
    if len(source_input) != self.batch_size or len(target_input) != self.batch_size:
      raise ValueError("the input source/target data has batch size %d, %d, which is not the expected %d batch size" %(len(source_input), len(target_input), self.batch_size))

    feed_dict = {}
    feed_dict[self.source_input] = source_input
    feed_dict[self.target_input] = target_input
    cost, weight, accuracy, _ = session.run([self._cost,
                                             self._batch_weight,
                                             self._accuracy,
                                             self._train_op if self.is_training else self._eval_op],
                                            feed_dict)
    self.total_cost += cost
    self.total_weight += weight
    if not self.is_training: self.total_accuracy += accuracy


  def report_progress(self, session):
    iters = self.global_step.eval(session)
    if self.total_weight > 0:
      ppl = np.exp(self.total_cost * 1.0 / self.total_weight)
      error = 100.0 - self.total_accuracy * 100.0 / self.total_weight
    else:
      ppl = self.target_vocab_size
      error = 100.0

    return iters, ppl, error

  def reset_stats(self):
    self.total_cost = 0
    self.total_accuracy = 0
    self.total_weight = 0

  def reduce_learning_rate(self, session):
    if not self.is_training:
      print("Warning: cannot change learning rate in evaluation")
    return session.run(self._learning_rate_decay_op)

def examine_model(params):
  total = 0
  for v in params:
    print (v.name)
    print (v.get_shape().as_list())
    res = 1
    for x in v.get_shape().as_list():
      res *= x
    total += res

  print("Number of parameters: %d" %total)

def restore_model(saver, session, outdir):
  ckpt = tf.train.get_checkpoint_state(outdir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())


def create_model_config(FLAGS):
  orig = ModelConfig()
  for k, v in FLAGS.__dict__['__flags'].items():
    if k in orig:
      orig[k] = v

  return orig


def train(FLAGS):
  """run model training"""

  trn_config = create_model_config(FLAGS)
  if trn_config.max_length < _buckets[-1][0] or trn_config.max_length < _buckets[-1][1]:
    raise ValueError("the maximum sequence must no less than the possible bucket size")

  warm_start = FLAGS.warm_start
  dev_config = create_model_config(FLAGS)
  dev_config.mode = 1
  np.random.seed(FLAGS.random_seed)
  src_train = os.path.join(FLAGS.data_dir, "source.train")
  tgt_train = os.path.join(FLAGS.data_dir, "target.train")
  src_dev = os.path.join(FLAGS.data_dir, "source.valid")
  tgt_dev = os.path.join(FLAGS.data_dir, "target.valid")
  random.seed(FLAGS.random_seed)


  with tf.Graph().as_default(), \
       tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.soft_placement,
                                        log_device_placement=FLAGS.log_device)) as session:
    initializer = tf.random_uniform_initializer(-trn_config.initial_weight, trn_config.initial_weight, seed=FLAGS.random_seed)
    with tf.variable_scope("TranslationModel", reuse=None, initializer=initializer):
      trn_model = TranslationModel(trn_config)
      examine_model(tf.trainable_variables())
    with tf.variable_scope("TranslationModel", reuse=True, initializer=initializer):
      dev_model = TranslationModel(dev_config)

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
    restore_model(saver, session, FLAGS.output_dir)
    tf.train.write_graph(session.graph.as_graph_def(), FLAGS.output_dir, "nmt.pb")
    graph_def_file = os.path.join(FLAGS.output_dir, "nmt.pb")

    train_set = data_utils_qnn.read_train_data(src_train, tgt_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = sum(train_bucket_sizes)
    print("Total %d sequences in the training data" % train_total_size)
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print(train_buckets_scale)

    if warm_start:
      train_bucket_sizes_ws = [train_bucket_sizes[i]*1.0/(2**i) for i in xrange(len(train_bucket_sizes))]
      train_total_size_ws = sum(train_bucket_sizes_ws)
      train_buckets_scale_ws = [sum(train_bucket_sizes_ws[:i + 1]) / train_total_size_ws for i in xrange(len(train_bucket_sizes_ws))]
      print(train_buckets_scale_ws)

    dev_set = data_utils_qnn.read_dev_data(src_dev, tgt_dev)
    print("Total %d sequences in the development data" %(len(dev_set)))

    #TODO: write training stats for tensorboard
    if FLAGS.log_stats:
      tf.merge_all_summaries()
      summary_writer = tf.train.SummaryWriter(FLAGS.output_dir, session.graph)
    else:
      summary_writer = None

    dev_losses = []
    best_ckpt = None
    best_loss = None
    while True:
      iters = trn_model.global_step.eval(session)
      if iters % FLAGS.spot_check_iters == 0:
        _, trn_ppl, _ = trn_model.report_progress(session)
        print("Has run %d minibatches, and PPL (training) is %.3f" %(iters, trn_ppl))
        trn_model.reset_stats()
      if iters % FLAGS.check_iters == 0:
        for dev_src_input, dev_tgt_input in data_utils_qnn.get_minibatch_dev(dev_config.batch_size, dev_set):
          dev_model.run_minibatch(session, dev_src_input, dev_tgt_input)
        _, dev_ppl, dev_loss = dev_model.report_progress(session)
        if summary_writer:
          summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Error", simple_value=dev_loss)]), iters)
        print("Has run training on %d minibatches, the PPL/Loss on the dev set is %.3f , %.3f" %(iters, dev_ppl, dev_loss))
        dev_model.reset_stats()
        if len(dev_losses) > 1 and (dev_losses[-1] - dev_loss) / dev_losses[-1] < 0.01:
          print("The learning rate is reduced to %f" %(trn_model.reduce_learning_rate(session)))

        dev_losses.append(dev_loss)
        checkpoint_path = os.path.join(FLAGS.output_dir, "nmt.ckpt")
        curr_ckpt = saver.save(session, checkpoint_path, global_step=iters)
        if not best_loss or best_loss > dev_loss:
          best_loss = dev_loss
          best_ckpt = curr_ckpt

      if iters > FLAGS.max_iters: break

      src_input, tgt_input = data_utils_qnn.get_minibatch(trn_config.batch_size,
                                                          train_set,
                                                          train_buckets_scale_ws if warm_start and iters < FLAGS.check_iters else train_buckets_scale)
      trn_model.run_minibatch(session, src_input, tgt_input)

  print("the best performing model is "+best_ckpt)
  return graph_def_file, best_loss, best_ckpt

def create_eval_graph(ckpt, config, out_graph_file):
  from tensorflow.python.framework import graph_util
  tf.reset_default_graph()
  with tf.Graph().as_default() as graph, tf.Session() as session:
    with tf.variable_scope("TranslationModel"):
      _create_encoder_eval_graph(config)
      _create_decoder_eval_graph(config)

    output_node_names = []
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(session, ckpt)
    graph_def = session.graph.as_graph_def()
    for node in graph_def.node: node.device = ""
    for op in session.graph.get_operations():
      idx = op.name.find("__QNNO__")
      if idx > 0 and op.name[idx:].find("/") < 0:
        print("find output node "+op.name)
        output_node_names.append(op.name)
      else:
        idx = op.name.find("__QNNI__")
        if idx > 0 and op.name[idx:].find("/") < 0:
          print("find input node " + op.name)


    output_graph_def = graph_util.convert_variables_to_constants(
      session, graph_def, output_node_names)

  with tf.gfile.GFile(out_graph_file, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  print("%d ops in the final graph." % len(output_graph_def.node))




def inference(FLAGS):
  source_vocab = data_utils_qnn.read_qnn_vocab(FLAGS.source_vocab_file)
  target_vocab =  data_utils_qnn.read_qnn_vocab(FLAGS.target_vocab_file)
  data_set = data_utils_qnn.read_eval_data(FLAGS.input_data, source_vocab, reverse = FLAGS.reverse)
  _, tgt_id2word = target_vocab

  with tf.Graph().as_default(), tf.Session() as session:
    ops_io = load_eval_graph(FLAGS.graph_file)
    source_input = ops_io['source_input']
    target_input = ops_io['target_input']
    encoder_output = ops_io['encoder_output']
    encoder_final_state = ops_io['encoder_final_state']
    encoder_dot_value_out = ops_io['encoder_dot_value_out']
    encoder_add_value_out = ops_io['encoder_add_value_out']
    encoder_dot_value_in = ops_io['encoder_dot_value_in']
    encoder_add_value_in = ops_io['encoder_add_value_in']
    attention = ops_io['attention']
    encoder_output_in = ops_io['encoder_output_in']
    decoder_init_state = ops_io['decoder_init_state']
    decoder_final_state = ops_io['decoder_final_state']
    tgt_gen_init_state = ops_io['tgt_gen_init_state']
    tgt_gen_final_state = ops_io['tgt_gen_final_state']
    logits = ops_io['logits']
    preds = tf.reshape(tf.cast(tf.argmax(logits, 1), tf.int32), [-1])

    fout = open(FLAGS.output_file, 'w')
    for seq in data_set:
      src = [seq]
      [enc_output,
       enc_final_state,
       enc_dot_value_out,
       enc_add_value_out,
       ] = session.run([encoder_output,
                        encoder_final_state,
                        encoder_dot_value_out,
                        encoder_add_value_out], {source_input:src})

      feed_dict = {encoder_output_in: enc_output,
                   encoder_add_value_in: enc_add_value_out,
                   encoder_dot_value_in: enc_dot_value_out}
      output = [_BOS_ID]
      while len(output) < FLAGS.max_length and output[-1] != _EOS_ID:
        tgt = output[-1]
        feed_dict[target_input] = [tgt]
        if tgt == _BOS_ID:
          feed_dict[decoder_init_state] = enc_final_state
        else:
          feed_dict[decoder_init_state] = decoder_state
          feed_dict[tgt_gen_init_state] = tgt_gen_state

        [enc_attention,
         pred,
         decoder_state,
         tgt_gen_state] = session.run([attention,
                         preds,
                         decoder_final_state,
                         tgt_gen_final_state], feed_dict)
        output.append(pred[0])

      out_words = [tgt_id2word[idx] for idx in output[1:-1]]
      fout.write(" ".join(out_words)+"\n")

    fout.close()

def load_eval_graph(graph_file):
  graph_def = tf.GraphDef()
  with open(graph_file, "rb") as f:
    graph_def.ParseFromString(f.read())

  ops_io = {}
  [ops_io['source_input'],
   ops_io['target_input'],
   ops_io['encoder_output'],
   ops_io['encoder_final_state'],
   ops_io['encoder_dot_value_out'],
   ops_io['encoder_add_value_out'],
   ops_io['encoder_dot_value_in'],
   ops_io['encoder_add_value_in'],
   ops_io['attention'],
   ops_io['encoder_output_in'],
   ops_io['decoder_init_state'],
   ops_io['decoder_final_state'],
   ops_io['tgt_gen_init_state'],
   ops_io['tgt_gen_final_state'],
   ops_io['logits']
   ] = tf.import_graph_def(graph_def, {}, ['TranslationModel/__QNNI__source_input:0',
                                           'TranslationModel/__QNNI__target_input:0',
                                           'TranslationModel/encoder/__QNNO__encoder_output:0',
                                           'TranslationModel/encoder/__QNNO__encoder_final_state:0',
                                           'TranslationModel/encoder/__QNNO__encoder_dot_value:0',
                                           'TranslationModel/encoder/__QNNO__encoder_add_value:0',
                                           'TranslationModel/__QNNI__encoder_dot_value:0',
                                           'TranslationModel/__QNNI__encoder_add_value:0',
                                           'TranslationModel/__QNNO__attention:0',
                                           'TranslationModel/__QNNI__encoder_output:0',
                                           'TranslationModel/decoder/RNN/__QNNI__decoder_state:0',
                                           'TranslationModel/decoder/RNN/__QNNO__decoder_state:0',
                                           'TranslationModel/output/RNN/__QNNI__target_gen_state:0',
                                           'TranslationModel/output/__QNNO__target_gen_state:0',
                                           'TranslationModel/output/__QNNO__prediction:0'], name="")

  return ops_io








