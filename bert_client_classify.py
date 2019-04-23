import pandas as pd
import json
import os
import random

import GPUtil
import tensorflow as tf
from bert_serving.client import BertClient
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])
tf.logging.set_verbosity(tf.logging.INFO)

data = pd.read_csv('../data/binary_twitter_nonan.txt')

# print(data.head())
# text = [data['text']]
# label = [data['signal'].astype(int)]

final_text = []
for i in data.text:
    final_text.append([i])

final_labels = []
for i in data.signal.astype(int):
    final_labels.append(str(i))

batch_size = 256
num_parallel_calls = 4
num_clients = num_parallel_calls*2
bc = BertClient()
# bc_clients = [BertClient(show_server_config=False) for _ in range(num_clients)]

def get_encodes(x):
    #bc_client = bc_clients.pop()
    # features = bc_client.encode(final_text)
    # bc_clients.append(bc_client)
    bc.encode(final_text)
    labels = final_labels
    return features, labels

train_fp = '../data/twitter_train.csv'
eval_fp = '../data/twitter_test.csv'

ds = (tf.data.TextLineDataset(train_fp).batch(batch_size)
                .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string]),  num_parallel_calls=num_parallel_calls)
                        .map(lambda x, y: {'feature': x, 'label': y})
                                .make_one_shot_iterator().get_next())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
run_config = RunConfig(model_dir='../data/',
                      session_config = config,
                      save_checkpoints_steps=1000)

estimator = DNNClassifier(
        hidden_units = [512],
        feature_columns = [tf.feature_column.numeric_column('feature')],
        n_classes=2,
        config=run_config,
        label_vocabulary=['30522'],
        dropout=0.1)

input_fn = lambda fp: (tf.data.TextLineDataset(fp)
                               .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
                                                      .batch(batch_size)
                                                                             .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string]), num_parallel_calls=num_parallel_calls)
                                                                                                    .map(lambda x, y: ({'feature': x}, y)).prefetch(20))
train_spec = TrainSpec(input_fn=lambda: input_fn(train_fp))
eval_spec = EvalSpec(input_fn=lambda: input_fn(eval_fp), throttle_secs=0)
train_and_evaluate(estimator, train_spec, eval_spec)


