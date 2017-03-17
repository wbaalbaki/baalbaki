from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from datetime import datetime

import tensorflow as tf
import sys

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
from qa_data import PAD_ID

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.005, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 30, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 50, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 766, "The output size of your model.")
tf.app.flags.DEFINE_integer("question_size", 100, "The max question size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 50, "Size of the pretrained vocabulary.")
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

def initialize_datasets(data_dir, trainTest='train', debugMode=False):
    # Open files
    questions = open(data_dir + '/' + trainTest + 'ids.question', 'rt')
    contexts = open(data_dir + '/' + trainTest + 'ids.context', 'rt')
    spans = open(data_dir + '/' + trainTest + 'span', 'rt')

    output = []
    numExamples = 0
    for question in questions:
        context = contexts.next().strip()
        span = spans.next().strip()

        # Get question
        question = [int(wordId) for wordId in question.split()]
        questionLen = len(question)

        # Pad question
        if questionLen > FLAGS.question_size:
            question = question[:FLAGS.question_size]
            questionLen = FLAGS.question_size
            #questionMask = [True] * FLAGS.question_size
        else:
            question = question + [PAD_ID] * (FLAGS.question_size - questionLen)
            #questionMask = [True] * questionLen + [False] *  (FLAGS.question_size - questionLen)

        # Get context
        context = [int(wordId) for wordId in context.split()]
        contextLen = len(context)

        # Pad context
        if contextLen > FLAGS.output_size:
            context = context[:FLAGS.output_size]
            contextLen = FLAGS.output_size
            #contextMask = [True] * FLAGS.output_size
        else:
            context = context + [PAD_ID] * (FLAGS.output_size - contextLen)
            #contextMask = [True] * contextLen + [False] *  (FLAGS.output_size - contextLen)

        # Span
        span = [int(spanIdx) for spanIdx in span.split()]

        output.append({"question": question,
                       "questionMask": questionLen,#questionMask,
                       "context": context,
                       "contextMask": contextLen,#contextMask,
                       "span": span})

        numExamples += 1
        if debugMode and numExamples > 100:
            break

    # Close files
    questions.close()
    contexts.close()
    spans.close()

    return output

def main(_):
    # Do what you need to load datasets from FLAGS.data_dir
    datasetTrain = initialize_datasets(FLAGS.data_dir, 'train.', debugMode=False)
    datasetVal = initialize_datasets(FLAGS.data_dir, 'val.', debugMode=False)
    datasetTrain.extend(datasetVal)


    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    #This is taking a long time
    tic = datetime.now()
    qa = QASystem(encoder, decoder, embed_path, FLAGS, rev_vocab)
    print('Time to setup the model: ', datetime.now() - tic)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    #saver = tf.train.Saver()

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        # Get directory to save model
        #save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        results_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        save_train_dir = results_path + "model.weights/"
        if not os.path.exists(save_train_dir):
            os.makedirs(save_train_dir)

        qa.train(sess, datasetTrain, save_train_dir)#, saver)

        qa.evaluate_answer(sess, datasetVal, rev_vocab, sample=1000, log=True)

if __name__ == "__main__":
    tf.app.run()
