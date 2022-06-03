import tensorflow as tf


def carlini_wagner_loss_fn(target_label):
    def _carlini_wagner_loss_fn(target_output):
        max_probs = tf.reduce_max(target_output, axis=1)
        target_label_probs = target_output[:, target_label]

        return tf.reduce_sum(max_probs - target_label_probs)

    return _carlini_wagner_loss_fn


def l2_norm_soft_hinge_loss_fn(bound):
    def _l2_norm_soft_hinge_loss_fn(perturbations):
        reshaped = tf.reshape(perturbations, (perturbations.shape[0], -1))

        return tf.reduce_mean(tf.maximum(tf.norm(reshaped, axis=1) - bound, 0))

    return _l2_norm_soft_hinge_loss_fn
