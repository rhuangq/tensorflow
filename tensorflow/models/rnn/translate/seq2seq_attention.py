"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

#from tensorflow.models.rnn.translate import data_utils_qnn
#_buckets = data_utils_qnn._buckets
#_EOS_ID = data_utils_qnn._EOS_ID
#_BOS_ID = data_utils_qnn._BOS_ID
#_UNK_ID = data_utils_qnn._UNK_ID

_buckets = [(5, 5), (7, 7), (9, 9), (11, 11), (15, 15), (20, 20), (32, 32)]
_UNK_ID = 0
_BOS_ID = 1
_EOS_ID = 2

logging = tf.logging

#scheduling
tf.app.flags.DEFINE_float("learning_rate", 0.5, "initial learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.8,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learn_rate", 0.05, "when the learning rate is reduced to this, stop training")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("spot_check_iters", 200,
                            "How many training iters/steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("check_iters", 10000, "do the validation for every this amount of model update")
tf.app.flags.DEFINE_integer("max_iters", 300000, "maximum number of model updates")
tf.app.flags.DEFINE_integer("random_seed", 5789, "the random seed for repeatable experiments")

#models
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("src_vocab_size", 30000, "source vocabulary size.")
tf.app.flags.DEFINE_integer("tgt_vocab_size", 30000, "target vocabulary size.")
tf.app.flags.DEFINE_boolean("use_lstm", True, "Use LSTM or GRU as the RNN layers")
tf.app.flags.DEFINE_boolean("use_birnn", True, "use BiRNN in the encoder")
tf.app.flags.DEFINE_integer("num_samples", 512, "number of samples used in importance sampling, use 0 to turn it off.")
tf.app.flags.DEFINE_integer("attention_type", 1, "attention type to use. 0: basic encoder-decoder; 1: global attention; 2: recurrent global attention")

#data
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory, assume {source|target}.{train|valid|vocab} files (generated in the QNN setup)")
tf.app.flags.DEFINE_string("output_dir", "output", "Training directory.")

tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

#multi-device computation, logging
tf.app.flags.DEFINE_boolean("soft_placement", True, "Allow soft placement of computation on different devices")
tf.app.flags.DEFINE_boolean("log_device", False, "Set to True to log computation device placement")
tf.app.flags.DEFINE_boolean("log_stats", False, "log stats for tensorboard")

FLAGS = tf.app.flags.FLAGS


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

    with vs.variable_scope("forward"):
      forward_cell = _create_rnn_multi_cell(use_lstm, num_cells, num_layers, keep_rate)
      if init_state_forward is None:
        init_state_forward = forward_cell.zero_state(batch_size, dtype)
    if use_birnn:
      with vs.variable_scope("backward"):
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
    src_values=None,
    dot_dim=None,
               initializer=None,
               scope=None,
               dtype=tf.float32):
  """
  Compute the attention vector by looking at the whole source sequence and current step in target.
   This function can be used in batch or called per step.
  """

  if dot_dim is None: dot_dim = out_dim
  with vs.variable_scope(scope or "GlobalAttention", initializer=initializer):
    bs, _, sd = source_inputs.get_shape()
    if bs != batch_size or sd != src_dim:
      raise ValueError("source data dimension mismatch, batch_size expected %d vs actual %d, dimention expected %d vs actual %d"
                       %(batch_size, bs, src_dim, sd))

    bs, _, sd = target_inputs.get_shape()
    if bs != batch_size or sd != tgt_dim:
      raise ValueError("target data dimension mismatch, batch_size expected %d vs actual %d, dimention expected %d vs actual %d"
                       %(batch_size, bs, tgt_dim, sd))

    src_dot_trans_w = tf.get_variable("src_dot_trans_w", [src_dim, dot_dim], dtype=dtype)
    src_dot_trans_b = tf.get_variable("src_dot_trans_b", [dot_dim], dtype=dtype)
    src_add_trans_w = tf.get_variable("src_add_trans_w", [src_dim, out_dim], dtype=dtype)
    src_add_trans_b = tf.get_variable("src_add_trans_b", [out_dim], dtype=dtype)
    tgt_dot_trans_w = tf.get_variable("tgt_dot_trans_w", [tgt_dim, dot_dim], dtype=dtype)
    tgt_dot_trans_b = tf.get_variable("tgt_dot_trans_b", [dot_dim], dtype=dtype)
    tgt_add_trans_w = tf.get_variable("tgt_add_trans_w", [tgt_dim, out_dim], dtype=dtype)
    tgt_add_trans_b = tf.get_variable("tgt_add_trans_b", [out_dim], dtype=dtype)

    if src_values is None:
      src_input_rs = tf.reshape(source_inputs, [-1, src_dim])
      src_dot_values = tf.reshape(tf.add(tf.matmul(src_input_rs, src_dot_trans_w), src_dot_trans_b), [batch_size, -1, dot_dim])
      src_add_values = tf.reshape(tf.add(tf.matmul(src_input_rs, src_add_trans_w), src_add_trans_b), [batch_size, -1, out_dim])
      src_values = (src_dot_values, src_add_values)
    else:
      src_dot_values, src_add_values = src_values

    tgt_input_rs = tf.reshape(target_inputs, [-1, tgt_dim])
    tgt_dot_values = tf.reshape(tf.add(tf.matmul(tgt_input_rs, tgt_dot_trans_w), tgt_dot_trans_b),
                                [batch_size, -1, dot_dim])
    tgt_add_values = tf.reshape(tf.add(tf.matmul(tgt_input_rs, tgt_add_trans_w), tgt_add_trans_b),
                                [batch_size, -1, out_dim])

    context_weights = []
    out_padding = tf.zeros([tf.shape(target_inputs)[1], out_dim])
    outputs = None
    weights = []
    for batch_idx in xrange(batch_size):
      #extract one sequence
      src_dot_value = tf.squeeze(tf.slice(src_dot_values, [batch_idx, 0, 0], [1, src_seq_lengths[batch_idx], -1]), [0])
      src_add_value = tf.squeeze(tf.slice(src_add_values, [batch_idx, 0, 0], [1, src_seq_lengths[batch_idx], -1]), [0])
      tgt_dot_value = tf.squeeze(tf.slice(tgt_dot_values, [batch_idx, 0, 0], [1, tgt_seq_lengths[batch_idx], -1]), [0])
      tgt_add_value = tf.squeeze(tf.slice(tgt_add_values, [batch_idx, 0, 0], [1, tgt_seq_lengths[batch_idx], -1]), [0])
      context_weight = tf.nn.softmax(tf.matmul(tgt_dot_value, src_dot_value, transpose_b=True)) # dim (tgt_len, src_len)
      context = tf.add(tf.matmul(context_weight, src_add_value), tgt_add_value) #dim (tgt_len, out_dim)
      padding = tf.slice(out_padding, [tgt_seq_lengths[batch_idx], 0], [-1, -1])
      if outputs is None:
        outputs = tf.concat(0, [context, padding])
      else:
        outputs = tf.concat(0, [outputs, context, padding])
      weights.append(context_weights)

    return src_values, tf.reshape(outputs, [batch_size, -1, out_dim]), context_weights


class MTconfig(object):
  #TODO: generate this from the command line configs
  batch_size = 64
  embed_size = 128
  num_layers = 2
  use_lstm = True
  use_birnn = True
  source_voc_size = 74
  target_voc_size = 54
  initial_weight = 0.1
  initial_learning_rate = 0.5
  learning_rate_decay = 0.8
  keep_rate = 1.0
  dtype = tf.float32
  max_length = 32
  max_grad_norm = 5.0
  filler = _EOS_ID




class TranslationModel(object):
  """The NMT model"""

  def __init__(self, is_training, config):
    self.batch_size = config.batch_size
    self.num_cells = config.embed_size
    self.num_layers = config.num_layers
    self.source_voc_size = config.source_voc_size
    self.target_voc_size = config.target_voc_size
    self.dtype = config.dtype
    self.dict = {}

    if FLAGS.attention_type == 0: config.use_birnn = False

    self._source_input = tf.placeholder(tf.int32, [self.batch_size, None], name="source_input")
    self._target_input = tf.placeholder(tf.int32, [self.batch_size, None], name="target_input")
    pad_batch = tf.constant(_EOS_ID, tf.int32, [self.batch_size, 1])
    target_output = tf.concat(1, [tf.slice(self._target_input, [0, 1], [-1, -1]), pad_batch])

    #TODO: move embedding to CPU?
    with vs.variable_scope("input"):
      src_embedding = tf.get_variable("source_embedding", [self.source_voc_size, self.num_cells], dtype=self.dtype)
      tgt_embedding = tf.get_variable("target_embedding", [self.target_voc_size, self.num_cells], dtype=self.dtype)
    s2s_src_input = tf.nn.embedding_lookup(src_embedding, self._source_input)
    s2s_tgt_input = tf.nn.embedding_lookup(tgt_embedding, self._target_input)

    seq_length_op = SequenceLength(self.batch_size, config.max_length, config.filler)
    src_length = seq_length_op.get_length(self._source_input)
    tgt_length = seq_length_op.get_length(self._target_input)

    encoder_init_state = None

    with vs.variable_scope("encoder") as varscope:
      encoder_outputs, encoder_final_state = Seq2SeqRnn(s2s_src_input,
                                                       self.num_cells,
                                                       self.num_layers,
                                                       use_lstm=config.use_lstm,
                                                       use_birnn=config.use_birnn,
                                                       seq_lengths=src_length,
                                                       init_state=encoder_init_state,
                                                         scope=varscope,
                                                       keep_rate=1.0 if not is_training else config.keep_rate)

    if config.use_birnn:
      decoder_init_state, _ = encoder_final_state
    else:
      decoder_init_state = encoder_final_state

    with vs.variable_scope("decoder") as varscope:
      decoder_outputs, decoder_final_state = Seq2SeqRnn(s2s_tgt_input,
                                                         self.num_cells,
                                                         self.num_layers,
                                                         use_lstm=config.use_lstm,
                                                         use_birnn=False,
                                                         seq_lengths=tgt_length,
                                                         init_state=decoder_init_state,
                                                         scope=varscope,
                                                         keep_rate=1.0 if not is_training else config.keep_rate)

    src_values = None
    attentions = None
    if FLAGS.attention_type == 1:
      src_values, s2s_outputs, attentions = GlobalAttention(self.batch_size,
                                                          self.num_cells * 2 if config.use_birnn else self.num_cells,
                                                          self.num_cells,
                                                          encoder_outputs,
                                              decoder_outputs,
                                              self.num_cells,
                                              src_length,
                                              tgt_length,
                                              src_values=None,
                                              dot_dim=self.num_cells)
    elif FLAGS.attention_type == 0:
      s2s_outputs = decoder_outputs
    else:
      raise ValueError("attention type %d is not implemented yet" %(FLAGS.attention_type))


    with vs.variable_scope("output"):
      tgt_gen_cell = _create_rnn_multi_cell(config.use_lstm, self.num_cells, self.num_layers,
                                          keep_rate=1.0 if not is_training else config.keep_rate)
      tgt_gen_init_state = None
      tgt_gen_outputs, tgt_gen_final_state = tf.nn.dynamic_rnn(tgt_gen_cell, s2s_outputs, tgt_length,
                                           initial_state=tgt_gen_init_state, dtype=self.dtype, time_major=False)
      output = tf.reshape(tgt_gen_outputs, [-1, self.num_cells])
      softmax_w = tf.get_variable("softmax_w", [self.num_cells, self.target_voc_size], dtype=self.dtype)
      softmax_b = tf.get_variable("softmax_b", [self.target_voc_size], dtype=self.dtype)
      logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name="prediction")

    tgt_weight = tf.reshape(seq_length_op.get_weight(self._target_input), [-1])
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
    self.dict = dict(
      source_input = self._source_input,
      target_input =self._target_input,
      encoder_init_state = encoder_init_state,
      encoder_final_state = encoder_final_state,
      encoder_cached_value = src_values,
      encoder_attention = attentions,
      decoder_init_state = decoder_init_state,
      decoder_final_state = decoder_final_state,
      tgt_gen_init_state = tgt_gen_init_state,
      tgt_gen_final_state = tgt_gen_final_state,
      target_output = logits,
    )

    if not is_training:
      self._accuracy = tf.reduce_sum(tf.cast(tf.equal(tgt_output, tf.cast(tf.argmax(logits, 1), tf.int32)), tf.float32) * tgt_weight)
      self._eval_op = self.global_step.assign(self.global_step + 1)
      return

    self.learning_rate = tf.Variable(
      config.initial_learning_rate, trainable=False, dtype=self.dtype)
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
    feed_dict[self._source_input] = source_input
    feed_dict[self._target_input] = target_input
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
      ppl = self.target_voc_size
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



def read_train_data(source_path, target_path, max_size=None):
  """Read training data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    All the data pre-processing must have been done before-hand (including BOS/EOS, reverse etc)
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()

        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]

        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) <= source_size and len(target_ids) <= target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

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

def restore_model(saver, session):
  ckpt = tf.train.get_checkpoint_state(FLAGS.output_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())

def get_minibatch(batch_size, data, scale):
  random_thresh = np.random.random_sample()
  bucket_id = -1
  for i in xrange(len(scale)):
    if scale[i] >= random_thresh:
      bucket_id = i
      break
  assert bucket_id >= 0

  src_size, tgt_size = _buckets[bucket_id]
  src_inputs, tgt_inputs = [], []
  for i in xrange(batch_size):
    curr_src, curr_tgt = random.choice(data[bucket_id])
    src_pad = [_EOS_ID] * (src_size - len(curr_src))
    src_inputs.append(curr_src + src_pad)
    tgt_pad = [_EOS_ID] * (tgt_size - len(curr_tgt))
    tgt_inputs.append(curr_tgt + tgt_pad)

  return src_inputs, tgt_inputs

def get_minibatch_dev(batch_size, data):
  """No need for random minibatch, we read the dev data sequentially,
  for efficiency reason, the `data` need be sorted by length (target, source)"""
  tot_batches = int(len(data)/batch_size)
  for i in xrange(tot_batches):
    begin = i * batch_size
    end = begin + batch_size
    src_len = len(data[begin][0])
    tgt_len = len(data[begin][1])
    for src, tgt in data[begin+1:end]:
      if len(src) > src_len: src_len = len(src)
      if len(tgt) > tgt_len: tgt_len = len(tgt)

    src_inputs, tgt_inputs = [], []
    for src,tgt in data[begin:end]:
      src_pad = [_EOS_ID] * (src_len - len(src))
      src_inputs.append(src+src_pad)
      tgt_pad = [_EOS_ID] * (tgt_len - len(tgt))
      tgt_inputs.append(tgt+tgt_pad)

    yield src_inputs,tgt_inputs


def decode():
  pass

def self_test():
  pass



def train():
  """run model training"""

  trn_config = MTconfig()
  if trn_config.max_length < _buckets[-1][0] or trn_config.max_length < _buckets[-1][1]:
    raise ValueError("the maximum sequence must no less than the possible bucket size")

  dev_config = MTconfig()
  np.random.seed(FLAGS.random_seed)
  import os
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
      trn_model = TranslationModel(True, trn_config)
      examine_model(tf.trainable_variables())
    with tf.variable_scope("TranslationModel", reuse=True, initializer=initializer):
      dev_model = TranslationModel(False, dev_config)

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
    restore_model(saver, session)
    tf.train.write_graph(session.graph.as_graph_def(), FLAGS.output_dir, "nmt.pb")

    train_set = read_train_data(src_train, tgt_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = sum(train_bucket_sizes)
    print("Total %d sequences in the training data" % train_total_size)
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print(train_buckets_scale)

    dev_set = read_dev_data(src_dev, tgt_dev)
    print("Total %d sequences in the development data" %(len(dev_set)))

    #TODO: write training stats for tensorboard
    if FLAGS.log_stats:
      tf.merge_all_summaries()
      summary_writer = tf.train.SummaryWriter(FLAGS.output_dir, session.graph)
    else:
      summary_writer = None

    dev_losses = []
    while True:
      iters = trn_model.global_step.eval(session)
      if iters % FLAGS.spot_check_iters == 0:
        _, trn_ppl, _ = trn_model.report_progress(session)
        print("Has run %d minibatches, and PPL (training) is %.3f" %(iters, trn_ppl))
        trn_model.reset_stats()
      if iters % FLAGS.check_iters == 0:
        for dev_src_input, dev_tgt_input in get_minibatch_dev(dev_config.batch_size, dev_set):
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
        saver.save(session, checkpoint_path, global_step=iters)


      if iters > FLAGS.max_iters: break

      src_input, tgt_input = get_minibatch(trn_config.batch_size, train_set, train_buckets_scale)
      trn_model.run_minibatch(session, src_input, tgt_input)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()










