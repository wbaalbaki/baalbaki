from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from qa_data import PAD_ID
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
        self.cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        self.cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.size*2)

    # Takes a batch, a sequence length (length of the paragraph), weights and biases, the max length of a sequence, the dimension of the hidden state, and return the
    # output of an LSTM
    def BiLSTM(self, paragraph, question, mask_paragraph, mask_question):#, encoder_state_input, scope="", reuse=False):

        with vs.variable_scope("ReadQuestion", reuse=None):
            questionLen = tf.reduce_sum(tf.cast(mask_question, tf.int32), axis=1)
            outputs_1, states_1 = tf.nn.bidirectional_dynamic_rnn(self.cell1, self.cell1, inputs=question,
                                                                        sequence_length=questionLen, dtype=tf.float64)
            lastStatesFw, lastStatesBw = states_1

            questionRepresentation = tf.nn.rnn_cell.LSTMStateTuple(tf.concat(1, lastStatesFw), tf.concat(1, lastStatesBw))

        with vs.variable_scope("ReadParagraphAfterQuestion", reuse=None):
            paragraphLen = tf.reduce_sum(tf.cast(mask_paragraph, tf.int32), axis=1)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(self.cell2, self.cell2, paragraph, sequence_length=paragraphLen,
                                                   initial_state_fw=questionRepresentation, initial_state_bw=questionRepresentation)
            questionContextRepresentation = outputs[1]


        #return questionContextRepresentation
        return self.encodeLinear(questionContextRepresentation, None)


    def encodeLinear(self, paragraph, max_length_paragraph):
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

        # Define weights
        W_s = tf.get_variable("W_s", shape=(self.size*2, 2), initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float64)
        b_s = tf.get_variable("b_s", shape=2, initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float64)

        W_e = tf.get_variable("W_e", shape=(self.size*2, 2), initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float64)
        b_e = tf.get_variable("b_e", shape=2, initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float64)

        pred_s = []
        pred_e = []

        with tf.variable_scope("LogisticRegression"):
            for time_step in range(paragraph.get_shape()[1]):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                pred_s.append(tf.matmul(paragraph[:,time_step,:], W_s) + b_s)
                pred_e.append(tf.matmul(paragraph[:,time_step,:], W_e) + b_e)

        # Stack and return
        pred_s = tf.stack(pred_s)
        pred_s = tf.transpose(pred_s, perm=[1, 0, 2])

        pred_e = tf.stack(pred_e)
        pred_e = tf.transpose(pred_e, perm=[1, 0, 2])

        return pred_s, pred_e

    def encode(self, paragraph, question, mask_paragraph, mask_question, dropout_rate, encoder_state_input, max_length_paragraph, n_hidden):

        return(self.BiLSTM(paragraph, question, mask_paragraph, mask_question))



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
        self.batch_size = tf.app.flags.FLAGS.batch_size#3
        self.numEpochs = tf.app.flags.FLAGS.epochs#2
        self.batchesToDisplay = 500
        self.embedPath = tf.app.flags.FLAGS.embed_path

        #Set up encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # ==== set up placeholder tokens ========
        self.max_length_paragraph = tf.app.flags.FLAGS.output_size
        self.max_length_question = 100
        self.setup_placeholders()

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.optimizer = get_optimizer("adam")
        self.train_op = self.optimizer.minimize(self.loss)
        #pass

    def setup_placeholders(self):
        self.question = tf.placeholder(shape=(None, self.max_length_question), name="Question", dtype=tf.int32)
        self.paragraph = tf.placeholder(shape=(None, self.max_length_paragraph), name="Paragraph", dtype=tf.int32)
        self.label_start = tf.placeholder(shape=(None, self.max_length_paragraph), name="LabelStart", dtype=tf.int32)
        self.label_end = tf.placeholder(shape=(None, self.max_length_paragraph), name="LabelEnd", dtype=tf.int32)
        self.mask_question = tf.placeholder(shape=(None, self.max_length_question), name="MaskQuestion", dtype=tf.bool)
        self.mask_paragraph = tf.placeholder(shape=(None, self.max_length_paragraph), name="MaskParagraph", dtype=tf.bool)
        self.dropout_placeholder = tf.placeholder(shape=(), name="Dropout", dtype=tf.float64)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        self.pred_start, self.pred_end = self.encoder.encode(self.embeddedParagraph, self.embeddedQuestion,
                                                             self.mask_paragraph, self.mask_question, self.dropout_placeholder,
                                                             1, self.max_length_paragraph, tf.app.flags.FLAGS.state_size)
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
            embedds = np.load("data/squad/glove.trimmed.100.npz")#self.embedPath)
            self.embeddings = tf.Variable(embedds["glove"], trainable=False)
            self.embeddedParagraph = tf.nn.embedding_lookup(self.embeddings, self.paragraph)
            self.embeddedQuestion = tf.nn.embedding_lookup(self.embeddings, self.question)

    def pad_sequence(self, sequence, max_length):
        currLen = len(sequence)
        if currLen > max_length:
            ret = sequence[0:max_length]
            mask = [1] * max_length
        else:
            ret = sequence + [PAD_ID] * (max_length - currLen)
            mask = [True] * currLen + [False] * (max_length - currLen)

        return [int(el) for el in ret], mask

    def set_feed_dict(self, x):
        # Create masks

        questions = []
        paragraphs = []
        masksQuestion = []
        masksParagraph = []
        labels_start = []
        labels_end = []

        # Make it a list if necessary
        if "ids.paragraph" in x:
            x = [x]

        for example in x:
            paragraph, maskParagraph = self.pad_sequence(example["ids.paragraph"], self.max_length_paragraph)
            paragraphs.append(paragraph)
            masksParagraph.append(maskParagraph)

            question, maskQuestion = self.pad_sequence(example["ids.question"], self.max_length_question)
            questions.append(question)
            masksQuestion.append(maskQuestion)

        # Create feed_dict
        input_feed = {self.question: questions,
                      self.paragraph: paragraphs,
                      self.mask_question: masksQuestion,
                      self.mask_paragraph: masksParagraph,
                      self.dropout_placeholder: self.dropout}

        # Add labels if that is the case
        if "labels_start" in x[0]:
            for example in x:
                label_start, _ = self.pad_sequence(example["labels_start"], self.max_length_paragraph)

                labels_start.append(label_start)

                label_end, _ = self.pad_sequence(example["labels_end"], self.max_length_paragraph)
                labels_end.append(label_end)

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

        # Might need to change deppending on the output format
        numWords = len(test_x["ids.paragraph"])
        yp = yp[0][:, 1][:numWords]
        yp2 = yp2[0][:, 1][:numWords]
        a_s = np.argmax(yp)
        a_e = np.argmax(yp2)

        #a_s = np.argmax(yp, axis=1)
        #a_e = np.argmax(yp2, axis=1)
        #a_s = 1
        #a_e = 2

        if a_s < a_e:
            return (a_s, a_e)
        else:
            return (a_e, a_s)

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

        totExamples = len(dataset)
        examplesToEvaluate = np.random.choice(totExamples, sample)

        for i in examplesToEvaluate:
            true_a_s = int(dataset[i]["span"][0])
            true_a_e = int(dataset[i]["span"][1])
            predicted_a_s, predicted_a_e = self.answer(session, dataset[i])
            ground_truth = dataset[i]["paragraph"][true_a_s:true_a_e+1]
            prediction = dataset[i]["paragraph"][predicted_a_s:predicted_a_e + 1]

            #Turn into a sentence
            ground_truth = ' '.join(ground_truth)
            prediction = ' '.join(prediction)

            em += float(exact_match_score(prediction, ground_truth))
            f1 += f1_score(prediction, ground_truth)
        f1 /= sample
        em /= sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

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

        # Loop through epochs
        numExamples = len(dataset)
        totBatches = numExamples/self.batch_size

        for epoch in range(self.numEpochs):
            tic = time.time()
            logging.info("\n\n\nEpoch %d out of %d", epoch + 1, tf.app.flags.FLAGS.epochs)

            randomOrder = np.random.permutation(numExamples)

            firstExampleInBatch = 0
            # Loop until we ran through the whole dataset
            batches = 0
            while firstExampleInBatch < numExamples:
                # Prepare minibatch
                currBatchStart = firstExampleInBatch
                currBatchEnd = min(firstExampleInBatch + self.batch_size - 1, numExamples - 1)
                currExamples = [dataset[randomOrder[i]] for i in range(currBatchStart, currBatchEnd+1)]

                # Set the max length paragraph and question according to the minibatch
                sizeParagraphs = []
                sizeQuestions = []
                for example in currExamples:
                    sizeParagraphs.append(len(example["ids.paragraph"]))
                    sizeQuestions.append(len(example["ids.question"]))

                # Train
                self.optimize(session, currExamples)

                if batches % self.batchesToDisplay == 0:
                    logging.info("%d batches out of %d", batches, totBatches)

                # Get ready for next batch
                firstExampleInBatch += self.batch_size
                batches += 1

            # Print current progress
            toc = time.time()
            logging.info("\n\nLast epoch took: %f secs" % (toc - tic))

            self.evaluate_answer(session, dataset, sample=100, log=True)

            # Save model after each epoch
            saver = tf.train.Saver()
            saver.save(session, train_dir)

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        # Set parameters back to default so can perform predictions
        self.max_length_paragraph = tf.app.flags.FLAGS.output_size
        self.max_length_question = 100

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
