import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# 传入数据集，从数据集中随机选择四个元素，然后返回这四个元素
# def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# 生成验证码图片，返回图片转化后的numpy数组，以及验证码字符文本
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 将图片保存到硬盘
    captcha_image = Image.open(captcha)
    captcha_image = captcha_image.convert('L')
    captcha_image = captcha_image.point(lambda i: 255 - i)
    # 将图片取反，黑色变为白色，白色变为黑色，这样模型收敛更块
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector

def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = gen_captcha_text_and_image()
        batch_x[i, :] = image.flatten()  # 将二维数组拉平为一维
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=w_alpha))
    b_c1 = tf.Variable(tf.random_normal([32], stddev=b_alpha))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=w_alpha))
    b_c2 = tf.Variable(tf.random_normal([64], stddev=b_alpha))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=w_alpha))
    b_c3 = tf.Variable(tf.random_normal([64], stddev=b_alpha))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_d = tf.Variable(tf.random_normal([8 * 32 * 40, 1024], stddev=w_alpha))
    b_d = tf.Variable(tf.random_normal([1024], stddev=b_alpha))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN], stddev=w_alpha))
    b_out = tf.Variable(tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN], stddev=b_alpha))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out

def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid，可以自己选择
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
            # 每10 step计算一次准确率
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc, loss_ = sess.run([accuracy, loss], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 0.8})
                print(step, loss_, acc)
                saver.save(sess, "./model/crack_capcha1.model", global_step=step)
                # 如果准确率大于90%,保存模型,完成训练
                if acc > 0.9:
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    break
            step += 1

if __name__ == '__main__':

    text, image = gen_captcha_text_and_image()
    print("验证码图像channel:", image.shape)  # (60, 160)
    # 图像大小
    IMAGE_HEIGHT = 60
    IMAGE_WIDTH = 160
    MAX_CAPTCHA = len(text)
    print("验证码文本字符数", MAX_CAPTCHA)
    # char_set = number + alphabet + ALPHABET
    char_set = number
    CHAR_SET_LEN = len(char_set)

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)  # dropout

    train_crack_captcha_cnn()
