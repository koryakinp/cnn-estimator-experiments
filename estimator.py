import tensorflow as tf
import numpy as np
from utils import *
from network import get_model


class Estimator:

    def __init__(self, data, model_dir, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.best_model = model_dir + "/best/"
        self.last_model = model_dir + "/last/"
        self.checkpoint = "checkpoint.chk"

    def process_train(self):
        tf.reset_default_graph()
        X = tf.placeholder(shape=[None, 784], dtype=tf.float32, name="X")
        Y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name="Y")
        model = get_model(X, apply_dropout=True)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=Y, logits=model)
        opt = tf.train.AdamOptimizer(0.001).minimize(loss)

        with tf.Session() as sess:

            if tf.train.checkpoint_exists(self.last_model + self.checkpoint):
                tf.train.Saver().restore(
                    sess, self.last_model + self.checkpoint)
            else:
                tf.global_variables_initializer().run()

            batches = self.data.get_number_of_train_batches(self.batch_size)
            for batch in range(batches):

                x, y = self.data.get_next_train_batch(self.batch_size)
                res1, res2 = sess.run([loss, opt], feed_dict={X: x, Y: y})

                print("\rBatch: {}/{} | Loss: {:.5f}".format(
                    batch + 1, batches, res1.mean()), end="")

            save_path = tf.train.Saver().save(
                sess, self.last_model + self.checkpoint)
        print("\r")

    def process_test(self):
        results = []
        tf.reset_default_graph()
        X = tf.placeholder(shape=[None, 784], dtype=tf.float32, name="X")
        Y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name="Y")
        model = get_model(X, apply_dropout=False)
        res = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
        res = tf.cast(res, tf.float32)

        with tf.Session() as sess:
            tf.train.Saver().restore(sess, self.last_model + self.checkpoint)
            batches = self.data.get_number_of_test_batches(self.batch_size)
            for batch in range(batches):

                x, y = self.data.get_next_test_batch(self.batch_size)

                results.append(sess.run(res, feed_dict={X: x, Y: y}))

                print("\rBatch: {}/{}".format(batch + 1, batches), end="")

        return np.array(results).mean()

    def build_model(self):
        best_accuracy = 0
        clear_folder(self.best_model)
        clear_folder(self.last_model)

        epochs = 10

        for epoch in range(1, epochs):
            self.data.shuffle()
            print("Epoch: {}/{}".format(epoch, epochs))
            self.process_train()
            cur_accuracy = self.process_test()
            if(cur_accuracy > best_accuracy):
                best_accuracy = cur_accuracy
                clear_folder(self.best_model)
                copy_files(self.last_model, self.best_model)
                print(best_accuracy)
            else:
                break
