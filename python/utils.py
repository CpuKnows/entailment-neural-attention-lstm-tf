import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import math
import json
from network import TensorFlowTrainable


def clean_sequence_to_words(sequence):
    sequence = sequence.lower()

    punctuations = [".", ",", ";", "!", "?", "/", '"', "'", "(", ")", "{", "}", "[", "]", "="]
    for punctuation in punctuations:
        sequence = sequence.replace(punctuation, " {} ".format(punctuation))
    sequence = sequence.replace("  ", " ")
    sequence = sequence.replace("   ", " ")
    sequence = sequence.split(" ")

    todelete = ["", " ", "  "]
    for i, elt in enumerate(sequence):
        if elt in todelete:
            sequence.pop(i)
    return sequence

def read_multinli_data(filename, size=-1):
    label_category = {
        'neutral': 0,
        'entailment': 1,
        'contradiction': 2
    }
    sentence1 = []
    sentence2 = []
    labels = []

    with open(filename, 'r', encoding="utf8") as f:
        i = 0
        not_found = 0
        for line in f:
            row = json.loads(line)
            if size == -1 or i < size:
                label = row['gold_label'].strip()
                if label in label_category:
                    sentence1.append(row['sentence1'].strip())
                    sentence2.append(row['sentence2'].strip())

                    labels.append(label_category[label])
                    i += 1
                else:
                    not_found += 1
            else:
                break;
        if not_found > 0:
            print('Label not recognized %d' % not_found)
                
    return (sentence1, sentence2, labels)

def load_data(data_files, embeddings_path):
    print("\nLoading embeddings:")
    embeddings = {}
    with open(embeddings_path, "r", encoding="utf8") as glove:
        for line in glove:
            name, vector = tuple(line.split(" ", 1))
            embeddings[name] = np.fromstring(vector, sep=" ")
    print("embeddings: done")

    dataset = {}
    print("\nLoading dataset:")
    for type_set, data_file in data_files.items():
        (p, h, t) = read_multinli_data(data_file)
        dataset[type_set] = {"premises": p, "hypothesis": h, "targets": t}

    tokenized_dataset = simple_preprocess(dataset=dataset, embeddings=embeddings)
    print("dataset: done\n")
    return embeddings, tokenized_dataset

def simple_preprocess(dataset, embeddings):
    tokenized_dataset = dict((type_set, {"premises": [], "hypothesis": [], "targets": []}) for type_set in dataset)
    print("tokenization:")
    for type_set in dataset:
        print("type_set:", type_set)
        map_targets = {"neutral": 0, "entailment": 1, "contradiction": 2}
        num_ids = len(dataset[type_set]["targets"])
        print("num_ids", num_ids)
        for i in range(num_ids):
            try:
                premises_tokens = [word for word in clean_sequence_to_words(dataset[type_set]["premises"][i])]
                hypothesis_tokens = [word for word in clean_sequence_to_words(dataset[type_set]["hypothesis"][i])]
                target = dataset[type_set]["targets"][i]
            except:
                pass
            else:
                tokenized_dataset[type_set]["premises"].append(premises_tokens)
                tokenized_dataset[type_set]["hypothesis"].append(hypothesis_tokens)
                tokenized_dataset[type_set]["targets"].append(target)
            sys.stdout.write("\rid: {}/{}".format(i + 1, num_ids))
            sys.stdout.flush()
        print("")
    print("tokenization: done")
    return tokenized_dataset



from network import RNN, LSTMCell, AttentionLSTMCell
from batcher import Batcher


def train(embeddings, dataset, parameters):
    modeldir = parameters["runs_dir"] + '/' + parameters["model_name"]
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    logdir = os.path.join(modeldir, "log")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logdir_train = os.path.join(logdir, "train")
    if not os.path.exists(logdir_train):
        os.mkdir(logdir_train)
    logdir_test = os.path.join(logdir, "test")
    if not os.path.exists(logdir_test):
        os.mkdir(logdir_test)
    logdir_dev = os.path.join(logdir, "dev")
    if not os.path.exists(logdir_dev):
        os.mkdir(logdir_dev)
    savepath = os.path.join(modeldir, "save")

    device_string = "/gpu:{}".format(parameters["gpu"]) if parameters["gpu"] else "/cpu:0"
    with tf.device(device_string):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config_proto = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        sess = tf.Session(config=config_proto)

        premises_ph = tf.placeholder(tf.float32, shape=[parameters["sequence_length"], None, parameters["embedding_dim"]], name="premises")
        hypothesis_ph = tf.placeholder(tf.float32, shape=[parameters["sequence_length"], None, parameters["embedding_dim"]], name="hypothesis")
        targets_ph = tf.placeholder(tf.int32, shape=[None], name="targets")
        keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob")

        _projecter = TensorFlowTrainable()
        projecter = _projecter.get_4Dweights(filter_height=1, filter_width=parameters["embedding_dim"], in_channels=1, out_channels=parameters["num_units"], name="projecter")

        optimizer = tf.train.AdamOptimizer(learning_rate=parameters["learning_rate"], name="ADAM", beta1=0.9, beta2=0.999)
        
        with tf.variable_scope(name_or_scope="premise"):
            premise = RNN(cell=LSTMCell, num_units=parameters["num_units"], embedding_dim=parameters["embedding_dim"], projecter=projecter, keep_prob=keep_prob_ph)
            premise.process(sequence=premises_ph)

        with tf.variable_scope(name_or_scope="hypothesis"):
            hypothesis = RNN(cell=AttentionLSTMCell, num_units=parameters["num_units"], embedding_dim=parameters["embedding_dim"], hiddens=premise.hiddens, states=premise.states, projecter=projecter, keep_prob=keep_prob_ph)
            hypothesis.process(sequence=hypothesis_ph)

        loss, loss_summary, accuracy, accuracy_summary = hypothesis.loss(targets=targets_ph)

        weight_decay = tf.reduce_sum([tf.reduce_sum(parameter) for parameter in premise.parameters + hypothesis.parameters])

        global_loss = loss + parameters["weight_decay"] * weight_decay

        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_writer = tf.summary.FileWriter(logdir_train, sess.graph)
        test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        test_summary_writer = tf.summary.FileWriter(logdir_test)
        
        saver = tf.train.Saver(max_to_keep=10)
        summary_writer = tf.summary.FileWriter(logdir)
        tf.train.write_graph(sess.graph_def, modeldir, "graph.pb", as_text=False)
        loader = tf.train.Saver(tf.global_variables())

        optimizer = tf.train.AdamOptimizer(learning_rate=parameters["learning_rate"], name="ADAM", beta1=0.9, beta2=0.999)
        train_op = optimizer.minimize(global_loss)

        sess.run(tf.global_variables_initializer())
        
        batcher = Batcher(embeddings=embeddings)
        train_batches = batcher.batch_generator(dataset=dataset["train"], num_epochs=parameters["num_epochs"], batch_size=parameters["batch_size"]["train"], sequence_length=parameters["sequence_length"])
        num_step_by_epoch = int(math.ceil(len(dataset["train"]["targets"]) / parameters["batch_size"]["train"]))
        for train_step, (train_batch, epoch) in enumerate(train_batches):
            feed_dict = {
                            premises_ph: np.transpose(train_batch["premises"], (1, 0, 2)),
                            hypothesis_ph: np.transpose(train_batch["hypothesis"], (1, 0, 2)),
                            targets_ph: train_batch["targets"],
                            keep_prob_ph: parameters["keep_prob"],
                        }

            _, summary_str, train_loss, train_accuracy = sess.run([train_op, train_summary_op, loss, accuracy], feed_dict=feed_dict)
            train_summary_writer.add_summary(summary_str, train_step)
            if train_step % 100 == 0:
                sys.stdout.write("\rTRAIN | epoch={0}/{1}, step={2}/{3} | loss={4:.2f}, accuracy={5:.2f}%   ".format(epoch + 1, parameters["num_epochs"], train_step % num_step_by_epoch, num_step_by_epoch, train_loss, 100. * train_accuracy))
                sys.stdout.flush()
            if train_step % 5000 == 0:
                test_batches = batcher.batch_generator(dataset=dataset["test"], num_epochs=1, batch_size=parameters["batch_size"]["test"], sequence_length=parameters["sequence_length"])
                for test_step, (test_batch, _) in enumerate(test_batches):
                    feed_dict = {
                                    premises_ph: np.transpose(test_batch["premises"], (1, 0, 2)),
                                    hypothesis_ph: np.transpose(test_batch["hypothesis"], (1, 0, 2)),
                                    targets_ph: test_batch["targets"],
                                    keep_prob_ph: 1.,
                                }

                    summary_str, test_loss, test_accuracy = sess.run([test_summary_op, loss, accuracy], feed_dict=feed_dict)
                    print("\nTEST | loss={0:.2f}, accuracy={1:.2f}%   ".format(test_loss, 100. * test_accuracy))
                    print("")
                    test_summary_writer.add_summary(summary_str, train_step)
                    break
            if train_step % 5000 == 0:
                saver.save(sess, save_path=savepath, global_step=train_step)
        print("")

