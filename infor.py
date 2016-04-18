import tensorflow as tf
import framwork
import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES



def loss(logits, labels, lambs):
    """Add L2Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


    # put a sigfunction on logits and then transpose
    logits = tf.transpose(framwork.sig_func(logits))
    # according to the labels, erase rows which is not in labels
    labels_unique = tf.constant(range(NUM_CLASSES), dtype=tf.int32)
    labels_num = NUM_CLASSES
    logits = tf.gather(logits, indices=labels_unique)
    lambs = tf.gather(lambs, indices=labels_unique)
    # set the value of each row to True when it occurs in labels
    template = tf.tile(tf.expand_dims(labels_unique, dim=1), [1, self.batch_size])
    labels_expand = tf.tile(tf.expand_dims(labels, dim=0), [labels_num, 1])
    indict_logic = tf.equal(labels_expand, template)
    # split the tensor along rows
    logit_list = tf.split(0, labels_num, logits)
    indict_logic_list = tf.split(0, labels_num, indict_logic)
    lambda_list = tf.split(0, self.image_classes, lambs)
    # loss_list = list()
    # for i in range(self.image_classes):
    #     loss_list.append(framwork.loss_func(logit_list[i], indict_logic_list[i], lambda_list[i]))
    loss_list = map(framwork.loss_func, logit_list, indict_logic_list, lambda_list)
    loss = tf.add_n(loss_list)
    tensors_dict = {'labels_unique': labels_unique, 'template': template, 'logits_sig_trans': logits,
                    'loss': loss, 'indict_logic': indict_logic}
    self.tensors_names.extend(tensors_dict.keys())
    self.net_tensors.update(tensors_dict)