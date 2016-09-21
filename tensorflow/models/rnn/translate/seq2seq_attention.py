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

  src_values = None
  attentions = None
  if attention_type == 1:
    src_values, s2s_outputs, attentions = GlobalAttention(batch_size,
                                                          num_cells * 2 if use_birnn else num_cells,
                                                          num_cells,
                                                          encoder_outputs,
                                                          decoder_outputs,
                                                          num_cells,
                                                          src_length,
                                                          tgt_length,
                                                          src_values=src_values,
                                                          dot_dim=num_cells)
  elif attention_type == 0:
    s2s_outputs = decoder_outputs
  else:
    raise ValueError("attention type %d is not implemented yet" % (attention_type))

  with vs.variable_scope("output"):
    tgt_gen_cell = _create_rnn_multi_cell(use_lstm, num_cells, num_layers,
                                          keep_rate=1.0 if mode != 0 else config.keep_rate)
    tgt_gen_init_state = None
    tgt_gen_outputs, tgt_gen_final_state = tf.nn.dynamic_rnn(tgt_gen_cell, s2s_outputs, tgt_length,
                                                             initial_state=tgt_gen_init_state, dtype=dtype,
                                                             time_major=False)
    output = tf.reshape(tgt_gen_outputs, [-1, num_cells])
    softmax_w = tf.get_variable("softmax_w", [num_cells, target_vocab_size], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [target_vocab_size], dtype=dtype)
    logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name="prediction")

  return dict(
    source_input=source_input,
    target_input=target_input,
    seq_length_op=seq_length_op,
    encoder_init_state=encoder_init_state,
    encoder_final_state=encoder_final_state,
    encoder_cached_value=src_values,
    encoder_attention=attentions,
    decoder_init_state=decoder_init_state,
    decoder_final_state=decoder_final_state,
    tgt_gen_init_state=tgt_gen_init_state,
    tgt_gen_final_state=tgt_gen_final_state,
    logits=logits,
  )


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
  max_src_len, max_tgt_len = _buckets[-1]
  max_len = max(max_src_len, max_tgt_len, max_len)
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        if len(source_ids) <= max_src_len and len(target_ids) <= max_tgt_len:
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

def restore_model(saver, session, outdir):
  ckpt = tf.train.get_checkpoint_state(outdir)
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
    best_ckpt = None
    best_loss = None
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
        curr_ckpt = saver.save(session, checkpoint_path, global_step=iters)
        if not best_loss or best_loss > dev_loss:
          best_loss = dev_loss
          best_ckpt = curr_ckpt

      if iters > FLAGS.max_iters: break

      src_input, tgt_input = get_minibatch(trn_config.batch_size, train_set, train_buckets_scale)
      trn_model.run_minibatch(session, src_input, tgt_input)

  print("the best performing model is "+best_ckpt)
  return graph_def_file, best_loss, best_ckpt

def create_eval_graph(ckpt, config, out_graph_file):
  from tensorflow.python.framework import graph_util
  with tf.Graph().as_default() as graph, tf.Session() as session:
    with tf.variable_scope("TranslationModel"):
      model_graph = create_nmt_graph(config)

    output_node_names = ['TranslationModel/output/prediction']
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(session, ckpt)
    graph_def = session.graph.as_graph_def()
    for node in graph_def.node: node.device = ""
    output_graph_def = graph_util.convert_variables_to_constants(
      session, graph_def, output_node_names)

  with tf.gfile.GFile(out_graph_file, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  print("%d ops in the final graph." % len(output_graph_def.node))



def read_eval_data(source_path, vocab, reverse = True):
  """read eval data, numericized it, reverse it, ..."""
  data_set = []
  word2id, _ = vocab
  for line in open(source_path):
    ids = [word2id[w] for w in line.split()]
    if reverse: ids.reverse()
    ids.insert(0, _BOS_ID)
    data_set.append(ids)

  return data_set



def load_eval_graph(graph_file, config):
  config.mode = 2
  #TODO: change it for beam decoding?
  config.batch_size = 1
  graph_def = tf.GraphDef()
  with open(graph_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name="")
  with tf.variable_scope("TranslationModel"):
    model_graph = create_nmt_graph(config)

  return model_graph

def read_qnn_vocab(vocab_file):
  word2id = {}
  id2word = {}
  voc_size = None
  idx = -1
  unk, bos, eos, unk_id, bos_id, eos_id = None, None, None, None, None, None
  for line in open(vocab_file):
    if not voc_size:
      assert line.startswith("<VocabSize>")
      _, voc_size = line.split()
      voc_size = int(voc_size)
    else:
      word,prob = line.split()
      try:
        _ = float(prob)
        idx += 1
        word2id[word] = idx
        id2word[idx] = word
      except:
        if word == "<UnknownWord>": unk = prob
        elif word == "<BeginOfSentenceWord>": bos = prob
        elif word == "<EndOfSentenceWord>": eos = prob
        else:
          raise RuntimeError("cannot parse string %s in the vocab file %s" %(word, vocab_file))

  if idx+1 != voc_size:
    raise RuntimeError("vocab file %s says it has %d words, but only read %d" %(vocab_file, voc_size, idx+1))

  unk_id = word2id[unk]
  bos_id = word2id[bos]
  eos_id = word2id[eos]

  if unk_id != _UNK_ID: raise RuntimeError("unkown ID mismatch in the vocab file: %d vs %d" %(unk_id, _UNK_ID))
  if bos_id != _BOS_ID: raise RuntimeError("BOS ID mismatch in the vocab file: %d vs %d" % (bos_id, _BOS_ID))
  if eos_id != _EOS_ID: raise RuntimeError("EOS ID mismatch in the vocab file: %d vs %d" % (eos_id, _EOS_ID))

  return (word2id, id2word)







