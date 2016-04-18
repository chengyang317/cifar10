# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import framwork
import cifar10_input


# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
BATCH_SIZE = 30


def loss(logits, labels, lambs):
    # put a sigfunction on logits and then transpose
    logits = tf.transpose(framwork.sig_func(logits))
    # according to the labels, erase rows which is not in labels
    labels_unique = tf.constant(range(NUM_CLASSES), dtype=tf.int32)
    labels_num = NUM_CLASSES
    # logits = tf.gather(logits, indices=labels_unique)
    # lambs = tf.gather(lambs, indices=labels_unique)
    # set the value of each row to True when it occurs in labels
    template = tf.tile(tf.expand_dims(labels_unique, dim=1), [1, BATCH_SIZE])
    labels_expand = tf.tile(tf.expand_dims(labels, dim=0), [labels_num, 1])
    indict_logic = tf.equal(labels_expand, template)
    # split the tensor along rows
    logit_list = tf.split(0, labels_num, logits)
    indict_logic_list = tf.split(0, labels_num, indict_logic)
    lambda_list = tf.split(0, NUM_CLASSES, lambs)
    # loss_list = list()
    # for i in range(self.image_classes):
    #     loss_list.append(framwork.loss_func(logit_list[i], indict_logic_list[i], lambda_list[i]))
    loss_list = map(framwork.loss_func, logit_list, indict_logic_list, lambda_list)
    losses = tf.add_n(loss_list)
    tf.add_to_collection('losses', losses)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')