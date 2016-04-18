# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10
import infor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 40000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train',
                           """Directory where to read model checkpoints.""")


def train(lambs):
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        # Get images and labels for CIFAR-10.
        images, labels = cifar10.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate loss.
        lambs = tf.constant(lambs, dtype=tf.float32)
        loss = infor.loss(logits, labels, lambs)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op, lr_op = cifar10.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            previous_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print ('Start training from previous')
        else:
            print('Strart training from step 0')
            previous_step = 0
            init = tf.initialize_all_variables()
            sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)

        step = previous_step
        while step <= FLAGS.max_steps:
            start_time = time.time()
            _, loss_value, lr_value = sess.run([train_op, loss, lr_op])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f, lr_value = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, lr_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            step += 1


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train([0.5] * 10)


if __name__ == '__main__':
    tf.app.run()
