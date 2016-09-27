
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
import numpy as np
import tensorflow as tf

#TODO: set it through command line or config files?
# MT
#_buckets = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (52, 52)]

# G2P
_buckets = [(5, 5), (7, 7), (9, 9), (11, 11), (15, 15), (20, 20), (32, 32)]

# this is from the vocab file, it's ok since we never change these values
_UNK_ID = 0
_BOS_ID = 1
_EOS_ID = 2


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

