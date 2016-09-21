
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_attention.Seq2SeqAttention for details of how they work.

#TODO: set it through command line or config files?
# MT
#_buckets = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (52, 52)]

# G2P
_buckets = [(5, 5), (7, 7), (9, 9), (11, 11), (15, 15), (20, 20), (32, 32)]

# this is from the vocab file, it's ok since we never change these values
_UNK_ID = 0
_BOS_ID = 1
_EOS_ID = 2

