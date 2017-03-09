from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.") #CHAAAANGE HEREEEEEEEEEEEEEEEEE

#CHAAAANGE HEREEEEEEEEEEEEEEEEE
#CHAAAANGE HEREEEEEEEEEEEEEEEEE
#CHAAAANGE HEREEEEEEEEEEEEEEEEE
#CHAAAANGE HEREEEEEEEEEEEEEEEEE

tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def initialize_datasets(data_dir):

    # Train
    rev_train_ids_context = []
    rev_train_context = []
    rev_train_ids_question = []
    rev_train_question = []
    rev_train_span = []

    with tf.gfile.GFile(data_dir + "/train.ids.context", mode="rb") as f:
        rev_train_ids_context.extend(f.readlines())
    rev_train_ids_context = [line.strip('\n') for line in rev_train_ids_context]

    with tf.gfile.GFile(data_dir + "/train.context", mode="rb") as f:
        rev_train_context.extend(f.readlines())
    rev_train_context = [line.strip('\n') for line in rev_train_context]

    with tf.gfile.GFile(data_dir + "/train.ids.question", mode="rb") as f:
        rev_train_ids_question.extend(f.readlines())
    rev_train_ids_question = [line.strip('\n') for line in rev_train_ids_question]

    with tf.gfile.GFile(data_dir + "/train.question", mode="rb") as f:
        rev_train_question.extend(f.readlines())
    rev_train_question = [line.strip('\n') for line in rev_train_question]

    with tf.gfile.GFile(data_dir + "/train.span", mode="rb") as f:
        rev_train_span.extend(f.readlines())
    rev_train_span = [line.strip('\n') for line in rev_train_span]

    datasetTrain = []
    for i in range(len(rev_train_ids_context)):
        a_s = [0] * len(rev_train_ids_context[i].split())
        a_s[int(rev_train_span[i].split()[0])] = 1
        a_e = [0] * len(rev_train_ids_context[i].split())
        a_e[int(rev_train_span[i].split()[1])] = 1

        datasetTrain.append({"ids.paragraph": rev_train_ids_context[i].split(),
                             "paragraph": rev_train_context[i].split(),
                             "ids.question": rev_train_ids_question[i].split(),
                             "question": rev_train_question[i].split(),
                             "labels_start": a_s,
                             "labels_end": a_e,
                             "span": (int(rev_train_span[i].split()[0]), int(rev_train_span[i].split()[1]))})

    # Validation
    rev_val_ids_context = []
    rev_val_context = []
    rev_val_ids_question = []
    rev_val_question = []
    rev_val_span = []

    with tf.gfile.GFile(data_dir + "/val.ids.context", mode="rb") as f:
        rev_val_ids_context.extend(f.readlines())
    rev_val_ids_context = [line.strip('\n') for line in rev_val_ids_context]

    with tf.gfile.GFile(data_dir + "/val.context", mode="rb") as f:
        rev_val_context.extend(f.readlines())
    rev_val_context = [line.strip('\n') for line in rev_val_context]

    with tf.gfile.GFile(data_dir + "/val.ids.question", mode="rb") as f:
        rev_val_ids_question.extend(f.readlines())
    rev_val_ids_question = [line.strip('\n') for line in rev_val_ids_question]

    with tf.gfile.GFile(data_dir + "/val.question", mode="rb") as f:
        rev_val_question.extend(f.readlines())
    rev_val_question = [line.strip('\n') for line in rev_val_question]

    with tf.gfile.GFile(data_dir + "/val.span", mode="rb") as f:
        rev_val_span.extend(f.readlines())
    rev_val_span = [line.strip('\n') for line in rev_val_span]

    datasetVal = []
    for i in range(len(rev_val_ids_context)):
        a_s = [0] * len(rev_val_ids_context[i].split())
        a_s[int(rev_val_span[i].split()[0])] = 1
        a_e = [0] * len(rev_val_ids_context[i].split())
        a_e[int(rev_val_span[i].split()[1])] = 1

        datasetVal.append({"ids.paragraph": rev_val_ids_context[i].split(),
                            "paragraph": rev_val_context[i].split(),
                            "ids.question": rev_val_ids_question[i].split(),
                            "question": rev_val_question[i].split(),
                            "labels_start": a_s,
                            "labels_end": a_e,
                            "span": (int(rev_val_span[i].split()[0]), int(rev_val_span[i].split()[1]))})

    return datasetTrain, datasetVal


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    datasetTrain, datasetVal = initialize_datasets(FLAGS.data_dir)
    # THIS IS JUST TO DEBUG, CHANGE LATER!!!!!
    #datasetTrain = datasetTrain[0:200]


    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, datasetTrain, save_train_dir)

        qa.evaluate_answer(sess, datasetVal, log=True)#, vocab FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
