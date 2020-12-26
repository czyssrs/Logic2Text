#!/usr/bin/env python
# -*- coding: utf-8 -*-

from SeqUnit import *
from DataLoader import DataLoader, Preprocessor
import model as model_gpt
from tqdm import tqdm
import encoder
import json
from utils import *
from datetime import datetime
import time
import pyrouge
import logging


# paths and mode
tf.app.flags.DEFINE_string("prog_name",'gpt_base','program name')

tf.app.flags.DEFINE_string("root_path", "../data_gpt/", "full path of data folder")
tf.app.flags.DEFINE_string("gpt_model_path",'../gpt_models','path of gpt2 model')
tf.app.flags.DEFINE_string("gpt_model_name",'117M','gpt2 model')
tf.app.flags.DEFINE_string("output_path", "../output_gpt", "full path of saved output")
tf.app.flags.DEFINE_string("model_save_name", "tmp", "full path of saved output")


tf.app.flags.DEFINE_string("mode",'test','train or test')

tf.app.flags.DEFINE_string("model_dir",'','specify model dir name')

tf.app.flags.DEFINE_boolean("use_table", True,'input table or not') # use table

# for resume training
tf.app.flags.DEFINE_string("resume_path",'','saved model path for use in resume mode')
tf.app.flags.DEFINE_string("resume_model_path",'','saved model path for use in resume model')

# for testing
tf.app.flags.DEFINE_string("saved_model_path",'','saved model path for use in test mode')

tf.app.flags.DEFINE_string("decoding",'beam','greedy or beam for decoding')
tf.app.flags.DEFINE_integer("beam_size", 2,'beam search for decoding')

# for table only
tf.app.flags.DEFINE_integer("max_input_len", 500, 'max input len')
tf.app.flags.DEFINE_integer("max_text_len", 50, 'max text len')
tf.app.flags.DEFINE_integer("max_table_len", 100, 'max table len')

# architecture choices
tf.app.flags.DEFINE_boolean("use_coverage", False,'use coverage or not')
tf.app.flags.DEFINE_float("coverage_penalty", 0.02,'coverage loss penalty')
tf.app.flags.DEFINE_boolean("use_copy_gate", True,'use copy gate or not')
tf.app.flags.DEFINE_float("copy_gate_penalty", 0.7, 'copy gate loss penalty')

# data options
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')
tf.app.flags.DEFINE_integer("source_vocab", 50257,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 50257,'vocabulary size')

# model hyperparams
tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 768, "Size of embedding.")

# training
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size of train set.")
tf.app.flags.DEFINE_integer("batch_size_test", 1, "Batch size of test set.")
tf.app.flags.DEFINE_integer("batch_update", 32, "apply gradients after steps")
tf.app.flags.DEFINE_integer("epoch", 5000, "Number of training epoch.")
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

# logging
tf.app.flags.DEFINE_integer("report", 50,'report valid results after some steps')
tf.app.flags.DEFINE_integer("report_loss", 10,'report loss results after some steps')

FLAGS = tf.app.flags.FLAGS

# create output paths
if FLAGS.mode == "train":
    model_dir_name =FLAGS.model_save_name + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(FLAGS.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=False)
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

elif FLAGS.mode == "resume":
    model_dir = os.path.join(FLAGS.output_path, FLAGS.resume_path)
    resume_model_dir = os.path.join(FLAGS.output_path, FLAGS.resume_model_path)
    results_path = os.path.join(model_dir, "results_1")
    saved_model_path = os.path.join(model_dir, "saved_model_1")
    os.makedirs(saved_model_path, exist_ok=False)
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log_1.txt')

else:
    saved_model_path = os.path.join(FLAGS.output_path, FLAGS.saved_model_path)
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(FLAGS.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, "results")
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')


# create data paths
root_path = FLAGS.root_path
print (root_path)
gold_path_valid = os.path.join(root_path, 'original_data', 'valid.text')
gold_path_test = os.path.join(root_path, 'original_data', 'test.text')
gold_json_valid = os.path.join(root_path, 'original_data', 'valid.json')
gold_json_test = os.path.join(root_path, 'original_data', 'test.json')
processed_data_dir = os.path.join(root_path, "processed_data")

# bpe vocab
last_best = 0.0
enc = encoder.get_encoder("117M", FLAGS.gpt_model_path)
eos = 50256 #TODO move to settings
empty = 2 #TODO move to settings


def train(sess, preprocessed_data, model):
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in FLAGS.flag_values_dict():
        value = FLAGS.flag_values_dict()[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    train_iterator = DataLoader(FLAGS.use_table, preprocessed_data.train_set, bpe_enc=FLAGS.gpt_model_path,
                                batch_size=FLAGS.batch_size, shuffle=True, eos=eos, empty=empty,
                                man_input_len=FLAGS.max_input_len, man_text_len=FLAGS.max_text_len, man_table_len=FLAGS.max_table_len)

    k = 0
    record_k = 0
    record_loss_k = 0 
    loss, start_time = 0.0, time.time()
    record_loss = 0.0

    for _ in range(FLAGS.epoch):
        train_iterator.reset()
        for x in train_iterator:

            model(x, sess, 0)
            k += 1

            #TODO also add to tensorboard
            if k % FLAGS.batch_update == 0:
                this_loss = model(x, sess, 1)
                record_loss += this_loss
                record_k += 1
                record_loss_k += 1

                if record_loss_k > 1 and record_loss_k % FLAGS.report_loss == 0:
                    write_log(log_file, "%d : loss = %.3f" % \
                        (record_k, record_loss / record_loss_k))
                    record_loss = 0.0
                    record_loss_k = 0

                if record_k > 1 and record_k % FLAGS.report == 0:
                    print("Round: ", record_k / FLAGS.report)
                    cost_time = time.time() - start_time
                    write_log(log_file, "%d : time = %.3f " % (record_k // FLAGS.report, cost_time))
                    start_time = time.time()
                    if record_k // FLAGS.report >= 1:
                        # save model
                        saved_model_path_cnt = os.path.join(saved_model_path, 'loads', str(record_k // FLAGS.report))
                        os.makedirs(saved_model_path_cnt, exist_ok=True)
                        model.save(saved_model_path_cnt, sess)

                        results_path_cnt = os.path.join(results_path, 'loads', str(record_k // FLAGS.report))
                        os.makedirs(results_path_cnt, exist_ok=True)
                        validation_result = evaluate(sess, preprocessed_data, model, results_path_cnt, 'valid')
                        write_log(log_file, validation_result)


def evaluate(sess, preprocessed_data, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        gold_path = gold_path_valid
        gold_json = gold_json_valid
        data_iterator = DataLoader(FLAGS.use_table, preprocessed_data.dev_set, 
                                    bpe_enc=FLAGS.gpt_model_path, batch_size=FLAGS.batch_size_test, shuffle=False,
                                    eos=eos, empty=empty, man_input_len=FLAGS.max_input_len, man_text_len=FLAGS.max_text_len, man_table_len=FLAGS.max_table_len)
    else:
        gold_path = gold_path_test
        gold_json = gold_json_test
        data_iterator = DataLoader(FLAGS.use_table, preprocessed_data.test_set, 
                                    bpe_enc=FLAGS.gpt_model_path, batch_size=FLAGS.batch_size_test, shuffle=False, eos=eos,
                                   empty=empty, man_input_len=FLAGS.max_input_len, man_text_len=FLAGS.max_text_len, man_table_len=FLAGS.max_table_len)

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    rouge_path = os.path.join(ksave_dir_mode, "rouge")
    os.makedirs(rouge_path, exist_ok=True)

    gold_path_rouge = os.path.join(root_path, "test_split_for_" + mode)
    out_real_path = os.path.join(ksave_dir_mode,  mode + "_text.clean.txt")
    out_json_path = os.path.join(ksave_dir_mode,  mode + "_text.res.txt")

    out_bpe = open(os.path.join(ksave_dir_mode, mode + "_text_bpe.txt"), "w")
    out_real = open(out_real_path, "w")
    pred_path = os.path.join(ksave_dir_mode,  mode + "_pred_text_")


    k = 0
    for x in tqdm(data_iterator):

        if FLAGS.decoding == "greedy":
            # greedy decoding
            predictions = model.generate(x, sess)
            for text in np.array(predictions):

                text = text.tolist()

                if eos in text:
                    text = text[:text.index(eos)] if text[0] != eos else [eos]
                real_sum = enc.decode(text)
                bpe_sum = " ".join([enc.decoder[tmp] for tmp in text])

                real_sum = real_sum.replace("\n", " ").strip()

                if check_res(real_sum) == False:
                    real_sum = "empty ."


                pred_list.append([real_sum])
                pred_unk.append(bpe_sum)

                out_real.write(real_sum + '\n')
                out_bpe.write(bpe_sum + '\n')


                with open(os.path.join(rouge_path, str(k) + "_decoded.txt"), "w") as f_out:
                    f_out.write(real_sum)

                k += 1


        elif FLAGS.decoding == "beam":

            ### beam search batch size = 1
            beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all = model.generate_beam(x, sess)
            seq = np.array(beam_seqs_all).tolist()
            score = np.array(beam_probs_all).tolist()
            cand = np.array(cand_seqs_all).tolist()
            cand_score = np.array(cand_probs_all).tolist()

            ind = cand_score.index(max(cand_score))
            ind_min = cand_score.index(min(cand_score))

            res_cand = cand[ind]

            if eos in res_cand:
                res_cand = res_cand[:res_cand.index(eos)] if res_cand[0] != eos else [eos]


            real_sum = enc.decode(res_cand[1:])
            bpe_sum = " ".join([enc.decoder[tmp] for tmp in res_cand])

            real_sum = real_sum.replace("\n", " ").strip()


            if check_res(real_sum) == False:
                real_sum = "empty ."
            # print (real_sum)

            pred_list.append([real_sum])
            pred_unk.append(bpe_sum)

            out_real.write(real_sum + '\n')
            out_bpe.write(bpe_sum + '\n')


            with open(os.path.join(rouge_path, str(k) + "_decoded.txt"), "w") as f_out:
                f_out.write(real_sum)

            k += 1



    out_bpe.close()
    out_real.close()


    ### write to results for compare
    get_res(gold_json, out_real_path, out_json_path)

    ### bleu
    bleu_res = bleu_score(gold_path, out_real_path)

    # print (bleu_result)
    bleu_result = "BLEU: %.4f\n" % bleu_res

    ### rouge
    r = pyrouge.Rouge155()

    r.model_filename_pattern = '#ID#_gold.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = gold_path_rouge
    r.system_dir = rouge_path

    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
    results_dict = r.convert_and_evaluate()

    rouge_result = "\n".join([results_dict.split("\n")[3], results_dict.split("\n")[7], results_dict.split("\n")[15], results_dict.split("\n")[19]])

    result = bleu_result + "\n" + rouge_result + "\n"

    return result


def main():

    # # keep track of the commit id
    # git_commit_id = get_current_git_version()
    # write_log(log_file, "GIT COMMIT ID: " + git_commit_id)

    gpt_model_name = os.path.join(FLAGS.gpt_model_path, FLAGS.gpt_model_name)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        hparams = model_gpt.default_hparams()
        with open(os.path.join(gpt_model_name, 'hparams.json')) as f:   
            hparams.override_from_dict(json.load(f))

        preprocessed_data = Preprocessor(processed_data_dir, FLAGS.limits, eos, empty, bpe_enc=FLAGS.gpt_model_path, 
                                            max_text_len=FLAGS.max_text_len, max_input_len=FLAGS.max_input_len)

        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size,
                        emb_size=FLAGS.emb_size, source_vocab=FLAGS.source_vocab, 
                        target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                        learning_rate=FLAGS.learning_rate, use_coverage = FLAGS.use_coverage,
                        coverage_penalty=FLAGS.coverage_penalty,
                        copy_gate_penalty=FLAGS.copy_gate_penalty, use_copy_gate=FLAGS.use_copy_gate,
                        gpt_hparams=hparams, decoding=FLAGS.decoding, beam_size=FLAGS.beam_size, empty_token=empty, stop_token=eos, 
                        max_length=FLAGS.max_text_len)

        model.build_summary(results_path)

        if FLAGS.mode == 'train':
            # collect all trainable variables, exclude embeddings
            gpt_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
            gpt_var_load = []
            for each_var in gpt_var:
                if "Adam" not in each_var.name:
                    gpt_var_load.append(each_var)
            gpt_var_load.remove(model.embedding)

            # load GPT checkpoint
            saver = tf.train.Saver(var_list=gpt_var_load)
            ckpt = tf.train.latest_checkpoint(gpt_model_name)
            saver.restore(sess, ckpt)

            # init other vars
            seq2seq_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq2seq')
            seq2seq_var.append(model.embedding)
            sess.run(tf.variables_initialier(var_list=seq2seq_var))

            train(sess, preprocessed_data, model)

        elif FLAGS.mode == 'resume':
            print (resume_model_dir)
            model.load(resume_model_dir, sess)
            train(sess, preprocessed_data, model)

        else:
            print (saved_model_path)
            model.load(saved_model_path, sess)
            test_result = evaluate(sess, preprocessed_data, model, results_path, 'test')
            write_log(log_file, test_result)


if __name__ == '__main__':
    main()