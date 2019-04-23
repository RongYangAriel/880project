from modules.multihead import *
from utils.model_helper import *
import time
from utils.prepare_data import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class AttentionClassifier(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int64, [None, self.max_len])
        self.label = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float64)

    def build_graph(self):
        print("building graph...")
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        # multi-head attention
        ma = multihead_attention(queries=batch_embedded, keys=batch_embedded)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        logits = tf.layers.dense(outputs, units=self.n_class)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
        self.prediction = tf.argmax(tf.nn.softmax(logits), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")


if __name__ == '__main__':
    # load data

    df = pd.read_csv('/home/g18rvb/nlpstocks/Project_Code/data/train.tsv', sep='\t')
    df_labels = df.ix[:, 1]
    # df.sort_values(by='Date')
    # df_labels = df.iloc[:, 2:3]
    # print(df_labels)

    input_ids_value = []
    for string_record in tf.python_io.tf_record_iterator("/home/g18rvb/nlpstocks/Project_Code/Organized_Code/bert_output/train.tf_record"):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        new_dict = dict(example.features.feature)
        input_ids_value.append(new_dict['input_ids'].int64_list.value)

    df = pd.DataFrame(input_ids_value)
    np_arr = np.asarray(df)
    # np_arr = np_arr[1:,]

    x_train = np_arr[1:24000]
    x_test = np_arr[24001:]
    y_train = df_labels[:23999]
    y_test = df_labels[24000:]
    
    # x_train = np_arr[:45000]
    # x_test = np_arr[45000:-1]
    # y_train = df_labels[:45000]
    # y_test = df_labels[45000:]
    # x_train, x_test, y_train, y_test = train_test_split(np_arr, df_labels, test_size=0.30, random_state=None) 

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    x_test, x_dev, y_test, y_dev, dev_size, test_size = \
        split_dataset(x_test, y_test, 0.1)
    print("Validation Size: ", dev_size)

    config = {
        "max_len": 256,
        "hidden_size": 128,
        "vocab_size": 30524,
        "embedding_size": 768,
        "n_class": 2,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "train_epoch": 30
    }

    classifier = AttentionClassifier(config)
    classifier.build_graph()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dev_batch = (x_dev, y_dev)
    start = time.time()
    for e in range(config["train_epoch"]):

        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))

        t1 = time.time()

        print("Train Epoch time:  %.3f s" % (t1 - t0))
        dev_acc = run_eval_step(classifier, sess, dev_batch)
        print("validation accuracy: %.3f " % dev_acc)

    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc = run_eval_step(classifier, sess, (x_batch, y_batch))
        test_acc += acc
        cnt += 1

    print("Test accuracy : %f %%" % (test_acc / cnt * 100))
