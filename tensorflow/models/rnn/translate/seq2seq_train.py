from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
import seq2seq_attention

logging = tf.logging

#scheduling
tf.app.flags.DEFINE_float("initial_learning_rate", 0.5, "initial learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.8,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learn_rate", 0.05, "when the learning rate is reduced to this, stop training")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("max_length", 32, "the maximum sequence length, longer sequence will be discarded")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("spot_check_iters", 200,
                            "How many training iters/steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("check_iters", 10000, "do the validation for every this amount of model update")
tf.app.flags.DEFINE_integer("max_iters", 300000, "maximum number of model updates")
tf.app.flags.DEFINE_integer("random_seed", 5789, "the random seed for repeatable experiments")

#models
tf.app.flags.DEFINE_integer("embed_size", 128, "Size of embedding vector.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_vocab_size", 100, "source vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 100, "target vocabulary size.")
tf.app.flags.DEFINE_boolean("use_lstm", True, "Use LSTM or GRU as the RNN layers")
tf.app.flags.DEFINE_boolean("use_birnn", True, "use BiRNN in the encoder")
tf.app.flags.DEFINE_float("keep_rate", 1.0, "value less than 1 will turn on dropouts")
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
tf.app.flags.DEFINE_string("out_eval_graph", None, "if set, write the inference graph to this file (pick the best on dev)")

FLAGS = tf.app.flags.FLAGS

def main(_):
  graph_def_file, _, best_ckpt = seq2seq_attention.train(FLAGS)
  if FLAGS.out_eval_graph:
    config = seq2seq_attention.create_model_config(FLAGS)
    seq2seq_attention.create_eval_graph(best_ckpt, config, FLAGS.out_eval_graph)

if __name__ == "__main__":
  tf.app.run()



