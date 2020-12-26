#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from model import *


class SeqUnit(object):
    def __init__(self, batch_size, hidden_size, emb_size, source_vocab, 
                 target_vocab, learning_rate, scope_name, name, use_coverage, coverage_penalty, 
                 copy_gate_penalty, use_copy_gate, gpt_hparams, decoding, beam_size, empty_token=28920, stop_token=50256, max_length=80):

        '''
        ###
        original full vocab ind
        empty_token=28920, stop_token=50256
        '''

        # data options
        self.empty_token = empty_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.start_token = empty_token

        self.decoding = decoding
        self.beam_size = beam_size

        # model hyperparams
        self.gpt_hparams = gpt_hparams
        self.hidden_size = self.gpt_hparams.n_embd

        # model architecture options
        # self.use_coverage = use_coverage
        # self.coverage_penalty = coverage_penalty
        # self.use_copy_gate = use_copy_gate
        # self.copy_gate_penalty = copy_gate_penalty
        self.scope_name = scope_name
        self.name = name

        # embedding sizes
        self.emb_size = self.gpt_hparams.n_embd # word embedding size

        # source and target vocabulary sizes, field and position vocabulary sizes
        self.source_vocab = self.gpt_hparams.n_vocab
        self.target_vocab = self.gpt_hparams.n_vocab

        # training options
        self.grad_clip = 5.0

        self.gpt_context = tf.placeholder(tf.int32, [None, None])
        self.encoder_input = tf.placeholder(tf.int32, [None, None])
        self.decoder_input = tf.placeholder(tf.int32, [None, None])
        self.encoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_output = tf.placeholder(tf.int32, [None, None])

        # define GPT decoder
        self.batch_size = tf.shape(self.encoder_input)[0]
        # initialize embedding
        gpt_emb_init_tune('model', self.gpt_hparams)
        # combine decoder contexts
        self.gpt_context_in = tf.concat([self.encoder_input, self.gpt_context], 1)
        context_outputs = self.step_gpt(self.gpt_hparams, self.gpt_context_in, self.batch_size)


        # get GPT embeddings
        with tf.variable_scope('model', reuse=True):
            ### use the one in gpt2
            self.embedding = tf.get_variable('wte_tune', [self.gpt_hparams.n_vocab, self.gpt_hparams.n_embd], trainable=False)

        # look up and combine embeddings
        # with tf.device("/gpu:1"):
        with tf.variable_scope(self.scope_name):
            
            self.encoder_embed = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
            self.decoder_embed = tf.nn.embedding_lookup(self.embedding, self.decoder_input)



        # decoder for training
        # get start values to start gpt generation
        logits0 = context_outputs['logits'][:, -1, :]
        dist0 = tf.nn.softmax(logits0) # start token
        x0 = tf.cast(tf.argmax(dist0, 1), tf.int32)
        past0 = context_outputs['presents']
        hidden0 = context_outputs['hidden'][:, -1, :]

        de_outputs, _ = self.decoder_t(self.decoder_input, self.decoder_len, x0, past0, hidden0)

        # # decoder for testing
        if self.decoding == "greedy":
            self.g_tokens = self.decoder_g(x0, past0, hidden0)

        # beam search
        elif self.decoding == "beam":
            x0_s = x0
            past0_s = past0
            hidden0_s = hidden0
            self.beam_seqs, self.beam_probs, self.cand_seqs, self.cand_probs = self.decoder_beam(x0_s, past0_s, hidden0_s, self.beam_size)

        ### enc-dec loss
        self.decoder_output_one_hot = tf.one_hot(indices=self.decoder_output, 
                                                depth=self.target_vocab,
                                                axis=-1)

        # mask for dec. plus eos
        dec_shape_len = tf.shape(self.decoder_output)[1]
        batch_nums = tf.range(0, dec_shape_len)
        batch_nums = tf.expand_dims(batch_nums, 0)
        batch_nums = tf.tile(batch_nums, [self.batch_size, 1])
        decoder_len_com = tf.expand_dims(self.decoder_len, 1)
        decoder_len_com = tf.tile(decoder_len_com, [1, dec_shape_len])
        mask = tf.cast(
                tf.less_equal(batch_nums, decoder_len_com), tf.float32)

        # total loss
        losses = -tf.reduce_sum(self.decoder_output_one_hot * tf.log(de_outputs + 1e-6), 2)
        losses = mask * losses
        # faster. original reduce mean
        self.mean_loss = tf.reduce_sum(losses)

        train_params = tf.trainable_variables()

        # train enc-dec
        with tf.variable_scope(scope_name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, train_params, colocate_gradients_with_ops=True), self.grad_clip)

            # accumulate gradient
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.acc_gradients = list(map(lambda param: tf.get_variable(param.name.split(":")[0],
                                                                    param.get_shape(), param.dtype,
                                                                    tf.constant_initializer(0.0), trainable=False),
                                                                    train_params))

            # initialize losses?
            self._loss = tf.get_variable("acc_loss", (), tf.float32, tf.constant_initializer(0.0), trainable=False)

            # We abuse the gradient descent optimizer for accumulating gradients and loss (summing)
            acc_opt = tf.train.GradientDescentOptimizer(-1.0)
            self.accumulate_gradients = acc_opt.apply_gradients(zip(self.grads, self.acc_gradients))
            self.acc_loss = acc_opt.apply_gradients([(self.mean_loss, self._loss)])

            # train update
            self.update = self.opt.apply_gradients(
                zip(list(map(lambda v: v.value(), self.acc_gradients)), train_params), global_step=self.global_step)

            # collect all values to reset after updating with accumulated gradient
            self.reset = list(map(lambda param: param.initializer, self.acc_gradients))
            self.reset.append(self._loss.initializer)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)


    def step_gpt(self, hparams, tokens, batch_size, past=None):
        """
        GPT2 model is imported here, as defined in model.py
        Args:
            hparams: Input parameters of the GPT architecture
            tokens: input tokens
            batch_size: batch size
            past: #TODO
        Returns: Output of transformer - logits in output sequence
        """

        lm_output = model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        hidden = lm_output['hidden']
        presents.set_shape(past_shape(hparams=hparams, batch_size=batch_size))
        return {'logits': logits, 'presents': presents, 'hidden': hidden}

    def decoder_t(self, inputs, inputs_len, x0, past0, hidden0):

        # gather p_gen and att_weights
        batch_size = tf.shape(self.decoder_input)[0]
        max_time = tf.shape(self.decoder_input)[1]
        encoder_len = tf.shape(self.encoder_embed)[1]

        time = tf.constant(0, dtype=tf.int32)
        f0 = tf.zeros([batch_size], dtype=tf.bool)

        inputs_ta = tf.TensorArray(dtype=tf.int32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, past, hidden, emit_ta, finished):

            # gpt generate
            temperature = 1.0  # hard coded temperature or noise in GPT logit output
            next_outputs = self.step_gpt(self.gpt_hparams, x_t[:, tf.newaxis], self.batch_size, past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
            final_dists = tf.nn.softmax(logits)
            past_nt = tf.concat([past, next_outputs['presents']], axis=-2)
            hidden_nt = next_outputs['hidden'][:, -1, :]

            # write to tensor array
            emit_ta = emit_ta.write(t, final_dists)

            # stop condition
            finished = tf.greater_equal(t, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.fill([batch_size], self.stop_token),
                                     lambda: inputs_ta.read(t))

            return t+1, x_nt, past_nt, hidden_nt, emit_ta, finished

        _, _, past_final, hidden_final, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, past0, hidden0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])

        return outputs, past_final



    def decoder_g(self, x0, past0, hidden0):

        batch_size = tf.shape(self.encoder_input)[0]
        encoder_len = tf.shape(self.encoder_embed)[1]

        time = tf.constant(0, dtype=tf.int32)
        f0 = tf.zeros([batch_size], dtype=tf.bool)

        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, past, hidden, emit_ta, finished):

            # gpt generate
            temperature = 1.0  # hard coded temperature or noise in GPT logit output
            next_outputs = self.step_gpt(self.gpt_hparams, x_t[:, tf.newaxis], self.batch_size, past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
            final_dists = tf.nn.softmax(logits)
            past_nt = tf.concat([past, next_outputs['presents']], axis=-2)
            hidden_nt = next_outputs['hidden'][:, -1, :]


            # write to tensor array
            emit_ta = emit_ta.write(t, final_dists)

            x_nt = tf.cast(tf.argmax(final_dists, 1), tf.int32)


            finished = tf.logical_or(finished, tf.equal(x_nt, self.stop_token))
            finished = tf.logical_or(finished, tf.greater_equal(t, self.max_length))
            return t+1, x_nt, past_nt, hidden_nt, emit_ta, finished

        _, _, past_final, hidden_final, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, past0, hidden0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        pred_tokens = tf.argmax(outputs, 2)

        return pred_tokens 


    def decoder_beam(self, x0, past0, hidden0, beam_size):
        '''
        to be used. not enough memory for deam search
        '''

        def beam_init():
            # return beam_seqs_1 beam_probs_1 cand_seqs_1 cand_prob_1 next_states time
            time_1 = tf.constant(1, dtype=tf.int32)
            # beam_seqs_0 = tf.constant([[x0]]*beam_size)
            beam_seqs_0 = tf.tile(x0[:, tf.newaxis], [beam_size, 1])
            beam_probs_0 = tf.constant([0.]*beam_size)

            # cand_seqs_0 = tf.constant([[x0]])
            cand_seqs_0 = x0[:, tf.newaxis]
            cand_probs_0 = tf.constant([-3e38])


            beam_seqs_0.set_shape((None, None))
            beam_probs_0.set_shape((None,))
            cand_seqs_0.set_shape((None, None))
            cand_probs_0.set_shape((None,))
            

            # gpt generate 
            ### fix batch size dimension issue
            temperature = 1.0  # hard coded temperature or noise in GPT logit output
            next_outputs = self.step_gpt(self.gpt_hparams, x0[:, tf.newaxis], 1, past=past0)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
            final_dists = tf.nn.softmax(logits)
            past_nt = tf.concat([past0, next_outputs['presents']], axis=-2)
            hidden_nt = next_outputs['hidden'][:, -1, :]

            # logprobs2d = tf.nn.log_softmax(o_t)
            logprobs2d = tf.log(final_dists)

            total_probs = logprobs2d + tf.reshape(beam_probs_0, [-1, 1])
            total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [1, self.stop_token]),
                               tf.tile([[-3e38]], [1, 1]),
                               tf.slice(total_probs, [0, self.stop_token + 1],
                                        [1, self.target_vocab - self.stop_token - 1])], 1)
            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
            print (flat_total_probs.get_shape().as_list())

            beam_k = tf.minimum(tf.size(flat_total_probs), beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

            next_bases = tf.floordiv(top_indices, self.target_vocab)
            next_mods = tf.mod(top_indices, self.target_vocab)

            next_beam_seqs = tf.concat([tf.gather(beam_seqs_0, next_bases),
                                        tf.reshape(next_mods, [-1, 1])], 1)

            cand_seqs_pad = tf.pad(cand_seqs_0, [[0, 0], [0, 1]], constant_values=self.stop_token)
            beam_seqs_EOS = tf.pad(beam_seqs_0, [[0, 0], [0, 1]], constant_values=self.stop_token)
            new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], 0)
            print (new_cand_seqs.get_shape().as_list())

            EOS_probs = tf.slice(total_probs, [0, self.stop_token], [beam_size, 1])
            new_cand_probs = tf.concat([cand_probs_0, tf.reshape(EOS_probs, [-1])], 0)
            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

            # part_state_0 = tf.reshape(tf.stack([s_nt[0]]*beam_size), [beam_size, self.hidden_size])
            # part_state_1 = tf.reshape(tf.stack([s_nt[1]]*beam_size), [beam_size, self.hidden_size])
            # part_state_0._shape = tf.TensorShape((None, None))
            # part_state_1._shape = tf.TensorShape((None, None))
            # next_states = (part_state_0, part_state_1)

            next_past = tf.stack([past_nt]*beam_size)
            next_past = tf.stack([tf.squeeze(past_nt, 0)]*beam_size)
            print (next_past.get_shape().as_list())

            return next_beam_seqs, next_beam_probs, next_cand_seqs, next_cand_probs, next_past, time_1

        beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, past_1, time_1 = beam_init()
        beam_seqs_1.set_shape((None, None))
        beam_probs_1.set_shape((None,))
        cand_seqs_1.set_shape((None, None))
        cand_probs_1.set_shape((None,))
        # states_1._shape = tf.TensorShape((2, None, self.hidden_size))
        def beam_step(beam_seqs, beam_probs, cand_seqs, cand_probs, past, time):
            '''
            beam_seqs : [beam_size, time]
            beam_probs: [beam_size, ]
            cand_seqs : [beam_size, time]
            cand_probs: [beam_size, ]
            '''
            inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [beam_size, 1]), [beam_size])

            # # print inputs.get_shape().as_list()
            # x_t = tf.nn.embedding_lookup(self.embedding, inputs)
            # # print(x_t.get_shape().as_list())
            # o_t, s_nt = self.dec_lstm(x_t, states)
            # o_t, w_t = self.att_layer(o_t)
            # o_t = self.dec_out(o_t)
            # logprobs2d = tf.nn.log_softmax(o_t)

            # gpt generate 
            ### fix batch size dimension issue
            temperature = 1.0  # hard coded temperature or noise in GPT logit output
            next_outputs = self.step_gpt(self.gpt_hparams, inputs[:, tf.newaxis], 1, past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
            final_dists = tf.nn.softmax(logits)
            past_nt = tf.concat([past, next_outputs['presents']], axis=-2)
            hidden_nt = next_outputs['hidden'][:, -1, :]

            logprobs2d = tf.log(final_dists)


            print (logprobs2d.get_shape().as_list())
            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])

            print (total_probs.get_shape().as_list())
            total_probs_noEOS = tf.concat([tf.slice(total_probs, [0, 0], [beam_size, self.stop_token]),
                                           tf.tile([[-3e38]], [beam_size, 1]),
                                           tf.slice(total_probs, [0, self.stop_token + 1],
                                                    [beam_size, self.target_vocab - self.stop_token - 1])], 1)
            print (total_probs_noEOS.get_shape().as_list())
            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
            print (flat_total_probs.get_shape().as_list())

            beam_k = tf.minimum(tf.size(flat_total_probs), beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)
            print (next_beam_probs.get_shape().as_list())

            next_bases = tf.floordiv(top_indices, self.target_vocab)
            next_mods = tf.mod(top_indices, self.target_vocab)
            print (next_mods.get_shape().as_list())

            next_beam_seqs = tf.concat([tf.gather(beam_seqs, next_bases),
                                        tf.reshape(next_mods, [-1, 1])], 1)
            # next_states = (tf.gather(s_nt[0], next_bases), tf.gather(s_nt[1], next_bases))
            next_past = tf.gather(past_nt, next_bases)
            print (next_beam_seqs.get_shape().as_list())

            cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]], constant_values=self.stop_token)
            beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]], constant_values=self.stop_token)
            new_cand_seqs = tf.concat([cand_seqs_pad, beam_seqs_EOS], 0) 
            print (new_cand_seqs.get_shape().as_list())

            EOS_probs = tf.slice(total_probs, [0, self.stop_token], [beam_size, 1])
            new_cand_probs = tf.concat([cand_probs, tf.reshape(EOS_probs, [-1])], 0)
            cand_k = tf.minimum(tf.size(new_cand_probs), beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

            return next_beam_seqs, next_beam_probs, next_cand_seqs, next_cand_probs, next_past, time+1
        
        def beam_cond(beam_probs, beam_seqs, cand_probs, cand_seqs, past, time):
            length =  (tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs))
            return tf.logical_and(length, tf.less(time, 60) )
            # return tf.less(time, 18)

        loop_vars = [beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, past_1, time_1]
        ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, back_prop=False,
                                    shape_invariants=[tf.TensorShape((beam_size, None)), tf.TensorShape((beam_size,)), 
                                    tf.TensorShape((beam_size, None)), tf.TensorShape((beam_size,)), 
                                    tf.TensorShape((beam_size, 12, 2, 12, None, 64)), tf.TensorShape((None))])
        beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all, _, time_all = ret_vars

        return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all



    def create_feed_dict(self, x, training=True):
        """
        Create feed dict with placeholder keys for feeding x input to model
        Args:
            x: dict, input
            training: bool, for training or inference
        Returns:
            feed_dict
        """
        feed_dict = {self.gpt_context: x['gpt_context'], self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len']}

        if training:
            feed_dict.update({self.decoder_input: x['dec_in'], self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out']})
        else:
            pass
        return feed_dict

    def __call__(self, x, sess, mode):
        """
        Calling this instance either accumulates gradients or runs optimizer update
        Args:
            x: data
            sess: TF Session
            mode: 0/1 accumulate gradient/run opt update
        Returns:
            total loss, copy gate loss, ?, ?
        """
        if mode == 0:
            feed_dict = self.create_feed_dict(x, training=True)
            loss, _, _ = sess.run([self.mean_loss,
                                   self.accumulate_gradients,
                                   self.acc_loss],
                                   feed_dict=feed_dict)
            return loss

        if mode == 1:
            acc_loss , summary, step = sess.run([self._loss, self.summary_op, self.global_step])
            sess.run(self.update)
            sess.run(self.reset)

            self.writer.add_summary(summary, step)

            return acc_loss 

    def generate(self, x, sess):
        """
        Generate predictions given input
        Args:
            x: input data
            sess: TF Session
        Returns:
            predictions and ? #TODO
        """
        feed_dict = self.create_feed_dict(x, training=False)
        predictions = sess.run(self.g_tokens, feed_dict=feed_dict)
        return predictions


    def generate_beam(self, x, sess):
        '''
        beam search generate. to be used
        not enough memory for beam search
        '''
        # beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all
        feed_dict = self.create_feed_dict(x, training=False)
        beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all = sess.run(
                         [self.beam_seqs,self.beam_probs, self.cand_seqs, self.cand_probs],
                         feed_dict=feed_dict)
        return beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all


    def build_summary(self, path):


        summaries = [
                tf.summary.scalar("loss/train", self._loss)
        ]
        self.summary_op = tf.summary.merge(summaries)
        summary_path = os.path.join(path, "summary")
        self.writer = tf.summary.FileWriter(summary_path)


    def save(self, path, sess):
        """
        Save model to file
        Args:
            path: path to save file
            sess: TF Session
        Returns:
            None
        """
        checkpoint_path = os.path.join(path, "model.ckpt")
        self.saver.save(sess, checkpoint_path, global_step=self.global_step.eval())
        print("Model saved on global step %d." % (self.global_step.eval()))
        return

    def load(self, path, sess):
        """
        Load saved model from checkpoint
        Args:
            path: checkpoint path
            sess: TF session
        Returns:
            None
        """
        ckpt = tf.train.get_checkpoint_state(path)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        return