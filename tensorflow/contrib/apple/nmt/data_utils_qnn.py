# Copyright (c) 2016 Apple Inc. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

# some commonly used bucket definitions
_preset_buckets = {
  'en-fr-mt': [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (52, 52)],
  'en-g2p': [(5, 5), (7, 7), (9, 9), (11, 11), (15, 15), (20, 20), (32, 32)]
}

# this is from the vocab file, it's ok since we never change these values
_UNK_ID = 0
_BOS_ID = 1
_EOS_ID = 2


def read_qnn_vocab(vocab_file):
  word2id = {}
  id2word = {}
  voc_size = None
  idx = 0
  unk, bos, eos, unk_id, bos_id, eos_id = None, None, None, None, None, None
  for line in open(vocab_file, 'rb'):
    word, prob = line.split()
    if not voc_size:
      assert word == b"<VocabSize>"
      voc_size = int(prob)
    else:
      try:
        _ = float(prob)
        word2id[word] = idx
        id2word[idx] = word
        idx += 1
      except:
        if word == b"<UnknownWord>": unk = prob
        elif word == b"<BeginOfSentenceWord>": bos = prob
        elif word == b"<EndOfSentenceWord>": eos = prob
        else:
          raise RuntimeError("cannot parse string %s in the vocab file %s" %(word, vocab_file))

  if idx != voc_size or len(word2id) != voc_size:
    raise RuntimeError("vocab file %s says it has %d words, but only read %d" %(vocab_file, voc_size, idx))

  unk_id = word2id[unk]
  bos_id = word2id[bos]
  eos_id = word2id[eos]

  if unk_id != _UNK_ID: raise RuntimeError("unkown ID mismatch in the vocab file: %d vs %d" %(unk_id, _UNK_ID))
  if bos_id != _BOS_ID: raise RuntimeError("BOS ID mismatch in the vocab file: %d vs %d" % (bos_id, _BOS_ID))
  if eos_id != _EOS_ID: raise RuntimeError("EOS ID mismatch in the vocab file: %d vs %d" % (eos_id, _EOS_ID))

  return (word2id, id2word)

def read_eval_data(source_path, vocab, unk_id = _UNK_ID, reverse = True):
  """read eval data, numericized it, reverse it, ..."""
  data_set = []
  word2id, _ = vocab
  for line in open(source_path, 'rb'):
    ids = [word2id[w] if w in word2id else unk_id for w in line.split()]
    if reverse: ids.reverse()
    ids.insert(0, _BOS_ID)
    data_set.append(ids)

  return data_set


def read_dev_data(buckets, source_path, target_path, max_len=1000):
  """read development data (usually pretty small)"""
  data_set = []
  max_src_len, max_tgt_len = buckets[-1]
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

def parse_bucket(in_str, preset):
  if in_str is None:
    if preset not in _preset_buckets:
      raise ValueError("%s does not match any preset buckets: %s" % (preset, " ".join(_preset_buckets.keys())))
    return _preset_buckets[preset]
  else:
    buckets = []
    try:
      for x in in_str.strip().split(','):
        sizes = [int(i) for i in x.split('-')]
        if len(sizes) != 2:
          raise ValueError("the bucket size def should be 'src-tgt'")
        if len(buckets) > 0:
          pre_src, pre_tgt = buckets[-1]
          if pre_src >= sizes[0] or pre_tgt >= sizes[1]:
            raise ValueError("the bucket size must be in strictly ascending order")
        buckets.append(tuple(sizes))
    except:
      raise ValueError("the bucket size def should be 'src-tgt', use comma to concatenate multiple buckets")

    return buckets

class TrainData(object):
  def __init__(self, buckets, source_path, target_path, batch_size, seed=8759, max_size=None):
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
      data_set: a list of length len(buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < buckets[n][0] and
        len(target) < buckets[n][1]; source and target are lists of token-ids.
    """
    self._buckets = buckets
    self._data_set = [[] for _ in buckets]
    self._random_number = random.Random(seed)
    self._batch_size = batch_size
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

          for bucket_id, (source_size, target_size) in enumerate(buckets):
            if len(source_ids) <= source_size and len(target_ids) <= target_size:
              self._data_set[bucket_id].append([source_ids, target_ids])
              break
          source, target = source_file.readline(), target_file.readline()


    train_bucket_sizes = [len(self._data_set[b]) for b in xrange(len(self._data_set))]
    train_total_size = sum(train_bucket_sizes)
    print("Total %d sequences in the training data" % train_total_size)
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size in i-th training bucket, as used later.

    train_bucket_sizes_ws = [train_bucket_sizes[i] * 1.0 / (2 ** i) for i in xrange(len(train_bucket_sizes))]
    train_total_size_ws = sum(train_bucket_sizes_ws)
    self._buckets_scale_ws = [sum(train_bucket_sizes_ws[:i + 1]) / train_total_size_ws for i in
                              xrange(len(train_bucket_sizes_ws))]
    print(self._buckets_scale_ws)
    self._cur_seq_idx = [0] * len(self._data_set)

    self._total_minibatch_indices = []
    for i, size in enumerate(train_bucket_sizes):
      self._total_minibatch_indices.extend([(i, j * self._batch_size) for j in xrange(int(size / self._batch_size))])

    self._minibatch_idx = 0

    self._shuffle()
    self._bucket_sizes = train_bucket_sizes
    self._epoch = 0
    self._warmedup = False

  def _shuffle(self):
    for i in range(len(self._data_set)):
      self._random_number.shuffle(self._data_set[i])
    self._random_number.shuffle(self._total_minibatch_indices)

  def get_minibatch(self, in_warmup = False):
    bucket_id = -1
    if in_warmup and not self._warmedup:
      max_draw = len(self._bucket_sizes)
      draw_idx = 0
      while draw_idx < max_draw and bucket_id == -1:
        random_thresh = np.random.random_sample()
        draw_idx += 1
        for i, scale in enumerate(self._buckets_scale_ws):
          if scale >= random_thresh and self._cur_seq_idx[i] + self._batch_size < self._bucket_sizes[i]:
            bucket_id = i
            break

      if bucket_id >= 0:
        src_inputs, tgt_inputs = self._get_minibatch_data(bucket_id, self._cur_seq_idx[bucket_id])
        self._cur_seq_idx[bucket_id] += self._batch_size
        return src_inputs, tgt_inputs
      else:
        self._warmedup = True

    # regular request
    bucket_id, minibatch_pos = self._total_minibatch_indices[self._minibatch_idx]
    src_inputs, tgt_inputs = self._get_minibatch_data(bucket_id, minibatch_pos)
    self._minibatch_idx += 1
    if self._minibatch_idx >= len(self._total_minibatch_indices):
      self._epoch += 1
      self._warmedup = True
      self._minibatch_idx = 0
      self._shuffle()

    return src_inputs, tgt_inputs

  def _get_minibatch_data(self, bucket_id, start):
    src_size, tgt_size = self._buckets[bucket_id]
    src_inputs, tgt_inputs = [], []
    for i in range(self._batch_size):
      idx = i + start
      curr_src, curr_tgt = self._data_set[bucket_id][idx]
      src_pad = [_EOS_ID] * (src_size - len(curr_src))
      src_inputs.append(curr_src + src_pad)
      tgt_pad = [_EOS_ID] * (tgt_size - len(curr_tgt))
      tgt_inputs.append(curr_tgt + tgt_pad)

    return src_inputs, tgt_inputs

  def epoch(self):
    return self._epoch + 1

