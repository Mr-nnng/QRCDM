import warnings
from keras.metrics import Accuracy, AUC, accuracy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
warnings.filterwarnings("ignore")

y_true, y_pred = tf.constant([1., 0, 1, 1]), tf.constant([0.5, 0.2, 0.7, 0.6])

a = Accuracy()(y_true, tf.round(y_pred))
b = accuracy(y_true, tf.round(y_pred))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(a)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
bbb = sess.run(b)

print(aaa, bbb)
