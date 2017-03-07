from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer()
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer()
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, paragraph, question, mask_paragraph, mask_question, droup_out, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        ones = tf.ones(shape=tf.shape(paragraph), dtype=tf.float32)
        a = tf.get_variable("test", initializer=2.0, dtype=tf.float32)
        return a*ones, 2*a*ones


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        return

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        #Load flags
        self.dropout = tf.app.flags.FLAGS.dropout
        self.batch_size = tf.app.flags.FLAGS.batch_size

        #Set up encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # ==== set up placeholder tokens ========
        self.max_length_paragraph = 800
        self.max_length_question = 100

        self.question = tf.placeholder(shape=(None, self.max_length_paragraph, encoder.vocab_dim), name="Question", dtype=tf.int32)
        self.paragraph = tf.placeholder(shape=(None, self.max_length_paragraph, encoder.vocab_dim), name="Paragraph", dtype=tf.int32)
        self.label_start = tf.placeholder(shape=(None, self.max_length_paragraph), name="LabelStart", dtype=tf.int32)
        self.label_end = tf.placeholder(shape=(None, self.max_length_paragraph), name="LabelEnd", dtype=tf.int32)
        self.mask_question= tf.placeholder(shape=(None, self.max_length_question), name="MaskQuestion", dtype=tf.bool)
        self.mask_paragraph = tf.placeholder(shape=(None, self.max_length_paragraph), name="MaskParagraph", dtype=tf.bool)
        self.dropout_placeholder = tf.placeholder(shape=(), name="Dropout", dtype=tf.float32)


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.setup_loss()
        self.optimizer = get_optimizer("adam")
        self.train_op = self.optimizer.minimize(self.loss)
        #pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        self.pred_start, self.pred_end = self.encoder.encode(self.paragraph, self.question, self.mask_paragraph, self.mask_question, self.dropout_placeholder, 1)
        #raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            lossStart = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_start, labels=self.label_start)
            lossEnd = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_end, labels=self.label_end)
            self.loss = lossStart + lossEnd
            self.loss = tf.boolean_mask(self.loss, self.mask_paragraph)
            self.loss = tf.reduce_mean(self.loss)
            #pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pass

    def set_feed_dict(self, x):
        # Create masks

        questions = []
        paragraphs = []
        masksQuestion = []
        masksParagraph = []
        labels_start = []
        labels_end = []

        for example in x:
            maskParagraph = [1] * len(self.max_size_paragraph)
            maskParagraph[len(example["ids.paragraph"]):] = 0
            masksParagraph.append(maskParagraph)

            maskQuestion = [1] * len(self.max_size_question)
            maskQuestion[len(example["ids.question"]):] = 0
            masksQuestion.append(maskQuestion)

            questions.append(example["ids.question"])
            paragraphs.append(example["ids.paragraph"])

        # Create feed_dict
        input_feed = {self.question: questions,
                      self.paragraph: paragraphs,
                      self.mask_question: maskParagraph,
                      self.mask_paragraph: maskQuestion,
                      self.dropout_placeholder: self.dropout}

        if "labels_start" in x:
            for example in x:
                labels_start = example["labels_start"]
                labels_end = example["labels_end"]

            input_feed[self.label_start] = labels_start
            input_feed[self.label_end] = labels_end

        return input_feed

    def optimize(self, session, train_x):#, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = self.set_feed_dict(train_x)
        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x):#, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = self.set_feed_dict(valid_x)
        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = self.set_feed_dict(test_x)
        output_feed = [self.pred_start, self.pred_end]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, in valid_dataset: #valid_y in valid_dataset:
          valid_cost += self.test(sess, valid_x)#, valid_y)

        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        totExamples = dataset.size[0]
        examplesToEvaluate = np.random.choice(totExamples, sample)

        for i in examplesToEvaluate:
            true_a_s = int(dataset[i]["span"][0])
            true_a_e = int(dataset[i]["span"][1])
            predicted_a_s, predicted_a_e = self.answer(session, dataset[i])
            ground_truth = dataset[i]["ids.paragraph"][true_a_s:true_a_e+1]
            prediction = dataset[i]["ids.paragraph"][predicted_a_s:predicted_a_e + 1]

            f1 += f1_score(prediction, ground_truth)
            em += exact_match_score(prediction, ground_truth)
        f1 /= sample
        em /= sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em


    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
        print("")


    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious approach can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """
        #loop through minibatches
        #set samples from minibatch
        #padding
        #populate placeholders
        #feed_dict(self.inputs: processedInouts[1:5])
        #tf.train / tf.run

        # Loop through epochs
        numExamples = len(dataset)
        firstExampleInBatch = 0

        for epoch in range(tf.app.flags.FLAGS.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, tf.app.flags.FLAGS.epochs)

            randomOrder = np.random.permutation(numExamples)

            # Loop until we ran through the whole dataset
            while firstExampleInBatch < numExamples:
                currBatchStart = firstExampleInBatch
                currBatchEnd = min(firstExampleInBatch + self.batch_size - 1, numExamples-1)
                print(currBatchStart, currBatchEnd)
                # Get ready for next batch
                firstExampleInBatch += self.batch_size

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
