import tensorflow as tf
import numpy as np

x = np.linspace(-3.0, 3.0, 100)
x.shape
x.dtype

x = tf.linspace(-3.0, 3.0, 100)
x

# Computational graph
g = tf.get_default_graph()
g

[op.name for op in g.get_operations()]
g.get_tensor_by_name('LinSpace' + ':0')

# Session
sess = tf.Session()
computed_x = sess.run(x)
computed_x
computed_x = x.eval(session=sess)
computed_x
sess.close()

sess = tf.Session(graph = tf.get_default_graph())
g2 = tf.Graph()
sess = tf.Session(graph=g2)
sess = tf.InteractiveSession

computed_x = x.eval()
sess = tf.InteractiveSession()
computed_x = x.eval()
computed_x

ksize = z.get_shape().as_list()[0]

z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
        (2.0 * tf.pow(sigma, 2.0))))) * (1.0/(sigma*tf.sqrt(2.0 * 3.1415)))
ksize = z.get_shape().as_list()[0]
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1,ksize]))
plt.imshow(z_2d.eval())
plt.show()

