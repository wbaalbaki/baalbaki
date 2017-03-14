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


def get_optimizer(opt, loss, max_grad_norm, learning_rate=0.01):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        assert (False)

    grads_and_vars = optfn.compute_gradients(loss)
    variables = [output[1] for output in grads_and_vars]
    gradients = [output[0] for output in grads_and_vars]

    gradients = tf.clip_by_global_norm(gradients, clip_norm=max_grad_norm)[0]
    #gradients = tmp_gradients

    grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]

    train_op = optfn.apply_gradients(grads_and_vars)

    return train_op
    #return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim
        self.LSTMcell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        self.CompresserLSTMcell = tf.nn.rnn_cell.BasicLSTMCell(1)

    def encode(self, inputs, masks, encoder_state_input=None):
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
        question, context = inputs
        questionLen, contextLen = masks

        #questionLen = tf.reduce_sum(tf.cast(questionMask, tf.int32), axis=1)
        #contextLen = tf.reduce_sum(tf.cast(contextMask, tf.int32), axis=1)
        # Question LSTM
        with vs.variable_scope("LSTMQuestionCOntext", reuse=None):
            # Biderectional
            _, (statesQuestion_fw, statesQuestion_bw)  = tf.nn.bidirectional_dynamic_rnn(self.LSTMcell, self.LSTMcell, inputs=question,
                                                                sequence_length=questionLen, dtype=tf.float32)

            # Uniderectional
            #_, statesQuestion = tf.nn.dynamic_rnn(cell=self.LSTMcell, inputs=question,
            #                                                    sequence_length=questionLen, dtype=tf.float32)

        with vs.variable_scope("LSTMQuestionCOntext", reuse=True):
            # Biderectional
            #(outputsFw, outputsBw)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.LSTMcell, self.LSTMcell, inputs=context,
                                                          sequence_length=contextLen, dtype=tf.float32,
                                                          initial_state_fw=statesQuestion_fw,
                                                          initial_state_bw=statesQuestion_bw)
            questionContext = tf.concat(2, outputs)

        return questionContext

            # Uniderectional
            #outputsQuestionContext, _ = tf.nn.dynamic_rnn(cell=self.LSTMcell, inputs=context,
            #                                                                  sequence_length=contextLen, dtype=tf.float32,
            #                                                                  initial_state=statesQuestion)


        #return(outputsQuestionContext)

        #with vs.variable_scope("Compresser", reuse=False):
            #(oneDimOutputs_fw, oneDimOutputs_bw)
        #    oneDimOutputs, _ = tf.nn.bidirectional_dynamic_rnn(self.CompresserLSTMcell,
        #                                                  self.CompresserLSTMcell, inputs=questionContext,
        #                                                  sequence_length=contextLen, dtype=tf.float32)


        #output = tf.reduce_sum(context, 1)
        #_, max_length, _ = oneDimOutputs_fw.get_shape().as_list()
        #oneDimOutputs_fw = tf.reshape(oneDimOutputs_fw, [-1, max_length])
        #oneDimOutputs_bw = tf.reshape(oneDimOutputs_bw, [-1, max_length])
        #output = (oneDimOutputs_fw, oneDimOutputs_bw)
        #return output  # Shape is [batch_size, max_length, encoded_size]        ## Must be a state of shape (None, some_enconder_number)

        oneDimOutputs = tf.concat(1, oneDimOutputs)
        _, max_length, _ = oneDimOutputs.get_shape().as_list()

        oneDimOutputs = tf.reshape(oneDimOutputs, [-1, max_length])

        return(oneDimOutputs)




class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep, dropout=None):
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

        # Approach 1
        #with vs.variable_scope("Decoder", reuse=False):
        #    output_s = tf.contrib.layers.fully_connected(inputs=knowledge_rep, num_outputs=self.output_size,
        #                                                 weights_initializer=tf.contrib.layers.xavier_initializer())
        #    output_e = tf.contrib.layers.fully_connected(inputs=knowledge_rep, num_outputs=self.output_size,
        #                                                 weights_initializer=tf.contrib.layers.xavier_initializer())
        #return (output_s, output_e)








        # Approach 2
        _, max_length, encoded_size = knowledge_rep.get_shape().as_list()

        with vs.variable_scope("Decoder", reuse=None):
            W_s = tf.get_variable("W_s", shape=(encoded_size, 1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_s = tf.get_variable("b_s", shape=(1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            W_e = tf.get_variable("W_e", shape=(encoded_size, 1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_e = tf.get_variable("b_e", shape=(1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        summaryRep_s = []
        summaryRep_e = []
        with vs.variable_scope("Decoder", reuse=True):
            for time_step in range(knowledge_rep.get_shape()[1]):
                x = knowledge_rep[:, time_step, :]

                if dropout is not None:
                    summaryRep_s.append(tf.matmul(tf.nn.dropout(x, dropout), W_s) + b_s)
                    summaryRep_e.append(tf.matmul(tf.nn.dropout(x, dropout), W_e) + b_e)
                else:
                    summaryRep_s.append(tf.matmul(x, W_s) + b_s)
                    summaryRep_e.append(tf.matmul(x, W_e) + b_e)

        output_s = tf.transpose(tf.stack(summaryRep_s), perm=[1, 0, 2])
        output_s = tf.reshape(output_s, [-1, max_length])

        output_e = tf.transpose(tf.stack(summaryRep_e), perm=[1, 0, 2])
        output_e = tf.reshape(output_e, [-1, max_length])

        return (output_s, output_e) #Must be of shape (None, Max_Context_Length)

class QASystem(object):
    def __init__(self, encoder, decoder, embed_path, FLAGS, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # Set to self
        self.encoder = encoder
        self.decoder = decoder
        self.embed_path = embed_path

        #Load from flags (later)
        self.max_grad_norm = FLAGS.max_gradient_norm
        self.questionMaxLen = 100
        self.output_size = FLAGS.output_size
        self.batch_size = FLAGS.batch_size
        self.numEpochs = FLAGS.epochs
        self.dropout = FLAGS.dropout
        self.starter_learning_rate = FLAGS.learning_rate


        # ==== set up placeholder tokens ========
        self.p_question = tf.placeholder(shape=(None, self.questionMaxLen), name="Question", dtype=tf.int32)
        #self.p_mask_question = tf.placeholder(shape=(None, self.questionMaxLen), name="MaskQuestion", dtype=tf.bool)
        self.p_mask_question = tf.placeholder(shape=(None), name="MaskQuestion", dtype=tf.int32)
        self.p_context = tf.placeholder(shape=(None, self.output_size), name="Context", dtype=tf.int32)
        #self.p_mask_context = tf.placeholder(shape=(None, self.output_size), name="MaskContext", dtype=tf.bool)
        self.p_mask_context = tf.placeholder(shape=(None), name="MaskContext", dtype=tf.int32)
        self.p_label_start = tf.placeholder(shape=(None), name="LabelEnd", dtype=tf.int32)
        self.p_label_end = tf.placeholder(shape=(None), name="LabelEnd", dtype=tf.int32)
        self.p_keep_prob_dropout_placeholder = tf.placeholder(shape=(), name="Dropout", dtype=tf.float32)


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        #self.optimizer = get_optimizer("adam", self.loss, self.max_grad_norm, self.starter_learning_rate)
        #self.train_op = self.optimizer.minimize(self.loss)  # , global_step=global_step)
        #learning_rate = self.starter_learning_rate

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step, 100000, 0.96, staircase=True)
        self.train_op = get_optimizer("adam", self.loss, self.max_grad_norm, learning_rate)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encodedState = self.encoder.encode(inputs=(self.embeddedQuestion, self.embeddedContext),
                                                 masks=(self.p_mask_question, self.p_mask_context))
        self.pred_s, self.pred_e = self.decoder.decode(encodedState, self.p_keep_prob_dropout_placeholder)
        #raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            lossStart = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_s, labels=self.p_label_start)
            #lossStart = tf.boolean_mask(lossStart, self.p_mask_context)

            lossEnd = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_e, labels=self.p_label_end)
            #lossEnd = tf.boolean_mask(lossEnd, self.p_mask_context)
            self.loss = tf.reduce_mean(lossStart + lossEnd)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedds = np.load(self.embed_path)
            self.embeddings = tf.Variable(embedds["glove"], trainable=False, dtype=tf.float32)
            self.embeddedQuestion = tf.nn.embedding_lookup(self.embeddings, self.p_question)
            self.embeddedContext = tf.nn.embedding_lookup(self.embeddings, self.p_context)

    def optimize(self, session, train_x):#, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        questions = [example["question"] for example in train_x]
        questionMasks = [example["questionMask"] for example in train_x]
        contexts = [example["context"] for example in train_x]
        contextsMasks = [example["contextMask"] for example in train_x]

        labels_s = [example["span"][0] for example in train_x]
        labels_e = [example["span"][1] for example in train_x]


        input_feed = {self.p_question: questions,
                      self.p_mask_question: questionMasks,
                      self.p_context: contexts,
                      self.p_mask_context: contextsMasks,
                      self.p_label_start: labels_s,
                      self.p_label_end: labels_e,
                      self.p_keep_prob_dropout_placeholder: 1.0-self.dropout}

        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {self.p_question: [test_x["question"]],
                      self.p_mask_question: [test_x["questionMask"]],
                      self.p_context: [test_x["context"]],
                      self.p_mask_context: [test_x["contextMask"]],
                      self.p_keep_prob_dropout_placeholder: 1.0}

        output_feed = [self.pred_s, self.pred_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)
        yp = yp[0]
        yp2 = yp2[0]


        # Masking answer
        yp = yp[0:test_x["contextMask"]]#"[yp[i] for i in range(len(yp)) if test_x["contextMask"]]
        yp2 = yp2[0:test_x["contextMask"]]#[yp2[i] for i in range(len(yp2)) if test_x["contextMask"]]

        a_s = np.argmax(yp)#, axis=1)
        a_e = np.argmax(yp2)#, axis=1)

        return (min(a_s, a_e), max(a_s, a_e))

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

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, vocab, sample=100, log=False):
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

            paragraphWords = [vocab[j] for j in dataset[i]["context"]]
            ground_truth = paragraphWords[true_a_s:true_a_e + 1]
            prediction = paragraphWords[predicted_a_s:predicted_a_e + 1]

            # Turn into a sentence
            ground_truth = ' '.join(ground_truth)
            prediction = ' '.join(prediction)

            # Evaluate
            em += float(exact_match_score(prediction, ground_truth))
            f1 += f1_score(prediction, ground_truth)
        f1 /= sample
        em /= sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):#, saver):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
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
        numExamples = len(dataset)
        totalBatches = numExamples/self.batch_size
        batchesToDisplay = int(totalBatches/5)

        saver = tf.train.Saver()
        maxLoss = 100000000
        #Loop through epochs
        for epoch in range(self.numEpochs):
            tic = time.time()
            logging.info("\n\nEpoch %d out of %d", epoch + 1, self.numEpochs)

            randomOrder = np.random.permutation(numExamples)
            firstExampleInBatch = 0
            batches = 0
            while firstExampleInBatch < numExamples:
                # Prepare minibatch
                currBatchStart = firstExampleInBatch
                currBatchEnd = min(firstExampleInBatch + self.batch_size - 1, numExamples - 1)
                currExamples = [dataset[randomOrder[i]] for i in range(currBatchStart, currBatchEnd + 1)]

                # Train
                _, currLoss = self.optimize(session, currExamples)

                # Display what is the current batch
                if batches % batchesToDisplay == 0: logging.info("%d batches out of %d, currentLoss is %f", batches, totalBatches, currLoss)

                # Get ready for next batch
                firstExampleInBatch += self.batch_size
                batches += 1

            # Print current progress
            toc = time.time()
            logging.info("Last epoch took: %f seconds" % (toc - tic))


            # Save model after each epoch
            if currLoss <= maxLoss:
                logging.info("Achieved best loss so far, saving the model")
                saver.save(session, train_dir+'my-model')
            #saver.save(session, train_dir)

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
