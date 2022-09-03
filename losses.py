from token import MINUS
import tensorflow as tf
#import sklearn.metrics
from scipy.spatial.distance import directed_hausdorff
from tensorflow.python.ops.gen_array_ops import unique_with_counts
from tensorflow_graphics.nn.loss import hausdorff_distance
#from sklearn.metrics import jaccard_score


def diceLoss(y_true, y_pred, class_weights):
    y_true = tf.convert_to_tensor(y_true, 'float32')
    y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)
    
    Sum_nom = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0,1,2,3])
    nom = tf.math.reduce_sum(tf.math.multiply(class_weights, Sum_nom)) 
    
    Sum_den = tf.math.reduce_sum(tf.math.add(y_true, y_pred), axis=[0,1,2,3])
    den = tf.math.reduce_sum(tf.math.multiply(class_weights, Sum_den)) + 1e-8

    return 1-2*nom/den

def jaccard_distance(y_true, y_pred, class_weights):
    y_true = tf.convert_to_tensor(y_true, 'float32')
    y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)

    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=[0,1,2,3])
    inter = tf.math.reduce_sum(tf.math.multiply(class_weights, intersection)) 

    sum_ = tf.math.reduce_sum(y_true + y_pred, axis=[0,1,2,3])
    sum2 = tf.math.reduce_sum(tf.math.multiply(class_weights, sum_))

    jac = (inter) / ((sum2 - inter) + 1e-8)
    jd =  1 - jac
    return jd


def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_real_output) - disc_real_output, 2))
    fake_loss = tf.math.reduce_mean(tf.math.pow(tf.zeros_like(disc_fake_output) - disc_fake_output, 2))

    disc_loss = 0.5*(real_loss + fake_loss)

    return disc_loss


def generator_loss(target, gen_output, disc_fake_output, class_weights, alpha):
    
    # generalized dice loss
    dice_loss = diceLoss(target, gen_output, class_weights)

    #iou_loss
    iou_loss = jaccard_distance(target, gen_output, class_weights)
    
    # disc loss
    disc_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_fake_output) - disc_fake_output, 2))
       
    # total loss
    gen_loss = alpha*dice_loss + disc_loss

    return gen_loss, iou_loss, dice_loss, disc_loss
