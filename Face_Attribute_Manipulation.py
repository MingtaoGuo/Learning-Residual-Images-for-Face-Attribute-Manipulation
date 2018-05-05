import tensorflow as tf
import numpy as np
import scipy.misc as misc
from PIL import Image
import os

epsilon = 1e-8
def mapping(img):
    return 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))

def conv(name, inputs, num_outs, ksize, strides, padding):
    kernel = tf.get_variable("w"+name, [ksize, ksize, int(inputs.shape[-1]), num_outs], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable("b"+name, [num_outs], initializer=tf.constant_initializer(0.02))
    return tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding) + b

def deconv(name, inputs, num_outs, ksize, strides, padding):
    B = int(inputs.shape[0])
    H = int(inputs.shape[1])
    W = int(inputs.shape[2])
    kernel = tf.get_variable("w"+name, [ksize, ksize, num_outs, int(inputs.shape[-1])], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable("b"+name, [num_outs], initializer=tf.constant_initializer(0.02))
    return tf.nn.conv2d_transpose(inputs, kernel, [B, int(H*strides), int(W*strides), num_outs], [1, strides, strides, 1], padding) + b

def fc(name, inputs, num_outs):
    inputs = tf.layers.flatten(inputs)
    C = int(inputs.shape[-1])
    W = tf.get_variable(name+"W", [C, num_outs], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(name+"FCb", [num_outs], initializer=tf.constant_initializer(0.02))
    return tf.matmul(inputs, W) + b

def instancenorm(x, scope_bn):
    with tf.variable_scope(scope_bn, reuse=tf.AUTO_REUSE):
        beta = tf.get_variable("beta", [x.shape[-1]], initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.get_variable("gamma", [x.shape[-1]], initializer=tf.constant_initializer(1.0), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments', keep_dims=True)
        return ((x - batch_mean) / tf.sqrt(batch_var + epsilon)) * gamma + beta

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(inputs, slope*inputs)

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            x = leaky_relu(instancenorm(conv("conv1", x, 64, 5, 1, "SAME"), "conv1"))
            x = leaky_relu(instancenorm(conv("conv2", x, 128, 4, 2, "SAME"),  "conv2"))
            x = leaky_relu(instancenorm(conv("conv3", x, 256, 4, 2, "SAME"), "conv3"))
            x = leaky_relu(instancenorm(deconv("deconv1", x, 128, 3, 2, "SAME"), "deconv1"))
            x = leaky_relu(instancenorm(deconv("deconv2", x, 64, 3, 2, "SAME"), "deconv2"))
            x = conv("conv4", x, 3, 4, 1, "SAME")
            return x

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            x = leaky_relu(instancenorm(conv("conv0", x, 32, 4, 2, "SAME"), "conv0"))
            x = leaky_relu(instancenorm(conv("conv1", x, 64, 4, 2, "SAME"), "conv1"))
            x = leaky_relu(instancenorm(conv("conv2", x, 128, 4, 2, "SAME"), "conv2"))
            temp = leaky_relu(instancenorm(conv("conv3", x, 256, 4, 2, "SAME"), "conv3"))
            x = temp
            x = leaky_relu(instancenorm(conv("conv4", x, 512, 4, 2, "SAME"), "conv4"))
            # x = batchnorm(leaky_relu(conv("conv5", x, 1024, 4, 2, "SAME")), train_phase, "conv5")
            x = conv("conv6", x, 512, 4, 1, "VALID")
            x = fc("fc", x, 3)
            return x, temp

class Face_Attribute_Manipulation:
    def __init__(self, batchsize=5):
        self.batchsize = batchsize
        self.x0 = tf.placeholder("float", [self.batchsize, 128, 128, 3])  # negative attribute
        self.x1 = tf.placeholder("float", [self.batchsize, 128, 128, 3])  # positive attribute
        G0 = Generator("G0")
        G1 = Generator("G1")
        D = Discriminator("D")
        #Transformation network
        self.r0 = G0(self.x0,  False)
        self.r1 = G1(self.x1, False)
        self.x0_tilde = self.x0 + self.r0
        self.x1_tilde = self.x1 + self.r1
        self.x0_tilde_logits, phi_x0_tilde = D(self.x0_tilde,  False)
        self.x1_tilde_logits, phi_x1_tilde = D(self.x1_tilde)
        self.x0_logits, phi_x0 = D(self.x0)
        self.x1_logits, phi_x1 = D(self.x1)
        self.dual_x0_logits, phi_dual_x0_tilde = D(G1(self.x0_tilde)+self.x0_tilde)
        self.dual_x1_logits, phi_dual_x1_tilde = D(G0(self.x1_tilde)+self.x1_tilde)
        #L1 norm regularization
        self.l_pix0 = tf.reduce_mean(tf.reduce_sum(tf.abs(self.r0), axis=[1, 2, 3]))
        self.l_pix1 = tf.reduce_mean(tf.reduce_sum(tf.abs(self.r1), axis=[1, 2, 3]))
        #Discriminator loss
        label_fake = np.ones([self.batchsize, 1])
        label_fake = np.float32(label_fake == np.array([1, 2, 3]))
        p_t = tf.nn.softmax(self.x0_tilde_logits)
        l_cls = tf.reduce_mean(-tf.log(tf.reduce_sum(p_t * label_fake, axis=1) + epsilon))
        p_t = tf.nn.softmax(self.x1_tilde_logits)
        l_cls += tf.reduce_mean(-tf.log(tf.reduce_sum(p_t * label_fake, axis=1) + epsilon))
        label_pos = np.ones([self.batchsize, 1]) * 2
        label_pos = np.float32(label_pos == np.array([1, 2, 3]))
        p_t = tf.nn.softmax(self.x1_logits)
        l_cls += tf.reduce_mean(-tf.log(tf.reduce_sum(p_t * label_pos, axis=1) + epsilon))
        label_neg = np.ones([self.batchsize, 1]) * 3
        label_neg = np.float32(label_neg == np.array([1, 2, 3]))
        p_t = tf.nn.softmax(self.x0_logits)
        l_cls += tf.reduce_mean(-tf.log(tf.reduce_sum(p_t * label_neg, axis=1) + epsilon))
        #perceptual loss
        self.l_per0 = tf.reduce_mean(tf.reduce_sum(tf.abs(phi_x0 - phi_x0_tilde), axis=[1, 2, 3]))
        self.l_per1 = tf.reduce_mean(tf.reduce_sum(tf.abs(phi_x1 - phi_x1_tilde), axis=[1, 2, 3]))
        #gan loss
        p_t = tf.nn.softmax(self.x0_tilde_logits)
        self.l_gan0 = tf.reduce_mean(-tf.log(tf.reduce_sum(p_t * label_pos, axis=1) + epsilon))
        p_t = tf.nn.softmax(self.x1_tilde_logits)
        self.l_gan1 = tf.reduce_mean(-tf.log(tf.reduce_sum(p_t * label_neg, axis=1) + epsilon))
        #dual loss
        p_t = tf.nn.softmax(self.dual_x0_logits)
        self.l_dual0 = tf.reduce_mean(-tf.log(tf.reduce_sum(p_t * label_neg, axis=1)+ epsilon))
        p_t = tf.nn.softmax(self.dual_x1_logits)
        self.l_dual1 = tf.reduce_mean(-tf.log(tf.reduce_sum(p_t * label_pos, axis=1) + epsilon))
        self.alpha = 5e-5
        self.beta = 0.1 * self.alpha
        self.l_G0 = self.l_gan0 + self.l_dual0 + self.alpha * self.l_pix0 + self.beta * self.l_per0
        self.l_G1 = self.l_gan1 + self.l_dual1 + self.alpha * self.l_pix1 + self.beta * self.l_per1
        self.l_D = l_cls
        self.Opt_G0 = tf.train.AdamOptimizer(2e-4).minimize(self.l_G0, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "G0"))
        self.Opt_G1 = tf.train.AdamOptimizer(2e-4).minimize(self.l_G1, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "G1"))
        self.Opt_D = tf.train.AdamOptimizer(2e-4).minimize(self.l_D, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "D"))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        positive_path = "./celeba_glasses_+1//"
        negative_path = "./celeba_glasses_-1//"
        positive = os.listdir(positive_path)[:1000]
        negative = os.listdir(negative_path)[:1000]
        nums = positive.__len__()
        saver = tf.train.Saver()
        for epoch in range(100):
            for i in range(int(nums / self.batchsize) - 1):
                x0 = np.zeros([self.batchsize, 128, 128, 3])  # negative
                x1 = np.zeros([self.batchsize, 128, 128, 3])  # positive
                for j in np.arange(i * self.batchsize, i * self.batchsize + self.batchsize, 1):
                    img = misc.imresize(np.array(Image.open(negative_path + negative[j])), [128, 128])
                    x0[j - i * self.batchsize, :, :, :] = img
                    img = misc.imresize(np.array(Image.open(positive_path + positive[j])), [128, 128])
                    x1[j - i * self.batchsize, :, :, :] = img
                self.sess.run(self.Opt_D, feed_dict={self.x0: x0, self.x1: x1})
                self.sess.run(self.Opt_G0, feed_dict={self.x0: x0, self.x1: x1})
                self.sess.run(self.Opt_G1, feed_dict={self.x0: x0, self.x1: x1})
                if i % 10 == 0:
                    [l_pix0, l_pix1, l_per0, l_per1, l_gan0, l_gan1, l_dual0, l_dual1, l_D, l_G0, l_G1] = \
                        self.sess.run([self.l_pix0, self.l_pix1, self.l_per0, self.l_per1, self.l_gan0, self.l_gan1, self.l_dual0, self.l_dual1, self.l_D, self.l_G0, self.l_G1],
                                      feed_dict={self.x0: x0, self.x1: x1})
                    print("epoch: %d step: %d l_pix0: %g l_pix1: %g l_per0: %g l_per1: %g l_gan0: %g l_gan1: %g l_dual0: %g l_dual1: %g l_G0: %g l_G1: %g l_D: %g"
                        % (epoch, i, l_pix0, l_pix1, l_per0, l_per1, l_gan0, l_gan1, l_dual0, l_dual1, l_G0, l_G1, l_D))
                    x0_tilde = self.sess.run(self.x0_tilde, feed_dict={self.x0: x0, self.x1: x1})
                    x0_tilde = mapping(x0_tilde)[0, :, :, :]
                    Image.fromarray(np.uint8(x0_tilde)).save("./result//x0_tilde.jpg")
                    x1_tilde = self.sess.run(self.x1_tilde,feed_dict={self.x0: x0, self.x1: x1})
                    x1_tilde = mapping(x1_tilde)[0, :, :, :]
                    Image.fromarray(np.uint8(x1_tilde)).save("./result//x1_tilde.jpg")
                    r0 = self.sess.run(self.r0, feed_dict={self.x0: x0, self.x1: x1})
                    r0 = mapping(r0)[0, :, :, :]
                    Image.fromarray(np.uint8(r0)).save("./result//r0.jpg")
                    r1 = self.sess.run(self.r1, feed_dict={self.x0: x0, self.x1: x1})
                    r1 = mapping(r1)[0, :, :, :]
                    Image.fromarray(np.uint8(r1)).save("./result//r1.jpg")
                    x0 = x0[0, :, :, :]
                    x1 = x1[0, :, :, :]
                    Image.fromarray(np.uint8(x0)).save("./result//x0.jpg")
                    Image.fromarray(np.uint8(x1)).save("./result//x1.jpg")
            np.random.shuffle(positive)
            np.random.shuffle(negative)
            saver.save(self.sess, "./saver//FAM.ckpt")


if __name__ == "__main__":
    fam = Face_Attribute_Manipulation()
    fam.train()