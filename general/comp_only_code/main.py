import os
import time
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.backend import set_session
from comp_only_code.ai_analyze_compression import ai_analyze_compression
import cv2
import SimpleITK as sitk
import numpy as np

MONOCHROME_TAG = '0028|0004'
MONOCHROME_1_STR = 'MONOCHROME1'

"""
FUNCTIONS FOR LOADING CNN MODELS
"""


def set_gpus():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


"""
CUSTOM FUNCTIONS FOR LOADING CNN MODELS
"""


def recall(y_target, y_pred):
    """ Compare the predicted value and target value and find the true positive and true positive + false negative.
    Then Calculate the recall value using those values.

        Args:
            y_target (list<int>): list of the correct class of the image
            y_pred (list<float>): list of possibility of the class in percentage
        Returns:
            recall (float): recall value

        Example:
            > > > recall([0,1,1,0,1,1,0],[0,1,0,1,1,0,0])

    """
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))

    # True Positive is when target and prediciton is both positive
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = true is 1
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    recall_val = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall_val


def precision(y_target, y_pred):
    """ Compare the predicted value and target value and find the true positive and true positive + false positive.
        Then Calculate the precision value using those values.

        Args:
            y_target (list<int>): list of the correct class of the image
            y_pred (list<float>): list of possibility of the class in percentage
        Returns:
            precision (float): precision value

        Example:
            > > > precision([0,1,1,0,1,1,0],[0,1,0,1,1,0,0])

    """
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1))

    # True Positive is when target and prediciton is both positive
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = prediction is 1
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    precision_val = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision_val


def sensitivity(y_target, y_pred):
    """ Compare the predicted value and target value and find the true positive and true positive + false negative.
        Then Calculate the sensitivity value using those values.

        Args:
            y_target (list<int>): list of the correct class of the image
            y_pred (list<float>): list of possibility of the class in percentage
        Returns:
            sensitivity (float): sensitivity value

        Example:
            > > > sensitivity([0,1,1,0,1,1,0],[0,1,0,1,1,0,0])

    """
    true_positives = K.sum(K.round(K.clip(y_target * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_target, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_target, y_pred):
    """ Compare the predicted value and target value and find the true negative and true negative + false positive.
        Then Calculate the speicificity value using those values.

        Args:
            y_target (list<int>): list of the correct class of the image
            y_pred (list<float>): list of possibility of the class in percentage
        Returns:
            specificity (float): specificity value

        Example:
            > > > specificity([0,1,1,0,1,1,0],[0,1,0,1,1,0,0])

    """
    true_negatives = K.sum(K.round(K.clip((1 - y_target) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_target, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def f1score(y_target, y_pred):
    """ Use the y_target and y_pred to calculate the recall and precision values.
    Then calculate the f1-score using the recall and precision values.

        Args:
            y_target (list<int>): list of the correct class of the image
            y_pred (list<float>): list of possibility of the class in percentage
        Returns:
            f1score (float): f1score value

        Example:
            > > > f1score([0,1,1,0,1,1,0],[0,1,0,1,1,0,0])

    """
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score


def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')  # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(loss, axis=1)

    return categorical_focal_loss_fixed


custom_object_1 = {"dice_loss": dice_loss,
                   "sensitivity": sensitivity,
                   "specificity": specificity,
                   "f1score": f1score,
                   "recall": recall,
                   "mean_iou": mean_iou,
                   "bce_logdice_loss": bce_logdice_loss,
                   "dice_coef": dice_coef,
                   "precision": precision}

def get_first_item(tuple_item):
    return tuple_item[0]


def unwrap_bit_str(wrapped_str):
    return wrapped_str[2:-1]


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.

    data_wo_outlier = data.copy()
    data_wo_outlier[s>m] = np.min(data_wo_outlier)
    return data_wo_outlier


def minmax_img(img):
    img_minmax = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img_minmax


def window_img(img, window_center, window_width, intercept, slope):
    if (intercept != '') and (slope != ''):
        img = (img*slope + intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img


def window_img_total(img, sitk_image):
    if MONOCHROME_TAG in sitk_image.GetMetaDataKeys() and MONOCHROME_1_STR in sitk_image.GetMetaData(MONOCHROME_TAG):
        img = np.invert(img)
    return img


def read_image(dcm_path):
    sitk_image = sitk.ReadImage(dcm_path)
    numpy_image = window_img_total(sitk.GetArrayFromImage(sitk_image)[0], sitk_image)

    try:
        pixel_spacing = sitk_image.GetSpacing()[0]
    except Exception as e:
        pixel_spacing = 0.16

    numpy_image = cv2.normalize(src=numpy_image, dst=None, alpha=0, beta=255,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return sitk_image, numpy_image, pixel_spacing

if __name__ == '__main__':
    GPU_NUMS = '0'                                          # 사용할 GPU의 Index number
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUMS
    CF_SEG_PTH = '/home/hyoon/cf_model_1.h5'                # 옮기신 CF Segmentation 모델의 weight 경로로 변경해주세요.

    file_name = ''                                          # Test 하실 파일 이름을 넣어주세요
    dcm_path = '/data/hyoon/'                               # Test 하실 파일 경로를 넣어주세요
    sitk_object, numpy_image, pixel_spacing = read_image(os.path.join(dcm_path, file_name))

    set_gpus()
    cf_model = load_model(CF_SEG_PTH, custom_objects=custom_object_1)
    highest_comp_rate, result_img, exception = ai_analyze_compression(numpy_image, pixel_spacing, file_name, cf_model)

    print("highest compression ratio: " + str(highest_comp_rate))
    print("Exception: " + str(exception))

    cv2.imwrite("/data/result/" + file_name[:-3] + 'png', result_img)                # 파일이 저장될 경로로 변경해주세요.



