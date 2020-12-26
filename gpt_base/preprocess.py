import time
import os
import string
import queue
import encoder
from tqdm import tqdm
import sys
import json
import gzip
import struct
from tensorflow.core.example import example_pb2
import pyrouge
import logging
import codecs
import csv

# bpe vocab
field_empty = 28920
eos = 50256


def clean_space(sent_in):
    '''
    clean extra space in sentence
    '''

    sent_in = sent_in.strip()
    res_sent = []
    for token in sent_in.split(" "):
        if token != "":
            res_sent.append(token)

    return " ".join(res_sent)


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

#### for rouge error:
def check_res(res):
    if res.strip() == "":
        return False
    for token in res.strip():
        if token.isalpha():
            return True

    return False

def linear_table_in(table):
    '''
    get processed linear table for gpt
    '''
    res = ""
    for ind, row in enumerate(table):
        res += (" row " + str(ind) + " : ")
        res += " ; ".join(row)

    return res.strip()




def text2id(subdir):
    '''
    convert text to bpe encoding
    '''

    ### for now just topic and logic interpret
    for set in ["train", "valid", "test"]:
        original = os.path.join(subdir, "original_data", set + ".json")
        processed_text = os.path.join(subdir, "processed_data", set, set + "_text.id")
        processed_topic = os.path.join(subdir, "processed_data", set, set + "_topic.id")
        processed_logic_interpret = os.path.join(subdir, "processed_data", set, set + "_logic_interpret.id")
        processed_logic = os.path.join(subdir, "processed_data", set, set + "_logic.id")
        processed_header = os.path.join(subdir, "processed_data", set, set + "_header.id")
        processed_table = os.path.join(subdir, "processed_data", set, set + "_table.id")

        original_for_test_text = os.path.join(subdir, "original_data", set + ".text")

        text_out = open(processed_text, "w")
        topic_out = open(processed_topic, "w")
        logic_interpret_out = open(processed_logic_interpret, "w")
        logic_out = open(processed_logic, "w")
        header_out = open(processed_header, "w")
        table_out = open(processed_table, "w")

        original_out_text = open(original_for_test_text, "w")

        tmp_dir = os.path.join(subdir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        with open(original) as f:
            data = json.load(f)
            for tup in tqdm(data):
                text = tup["sent"]
                topic = tup["topic"]
                logic_interpret = tup["interpret"]
                logic = tup["logic_str"]
                header = " ; ".join(tup["table_header"])
                table = linear_table_in(tup["table_cont"])


                tokens_text, token_bpe = enc.encode(text)
                tokens_topic, _ = enc.encode(topic)
                tokens_interpret, _ = enc.encode(logic_interpret)
                tokens_logic, _ = enc.encode(logic)
                tokens_header, _ = enc.encode(header)
                tokens_table, _ = enc.encode(table)

                if len(tokens_text) == 0:
                    print ("error")
                if len(tokens_topic) == 0:
                    print ("error")
                if len(tokens_topic) == 0:
                    print ("error")
                if len(tokens_header) == 0:
                    print ("error")
                if len(tokens_table) == 0:
                    print ("error")


                # print ("###################")
                # print (topic + "\n\n")
                # print (header + "\n\n")
                # print (table + "\n\n")
                # print (logic_interpret + "\n\n")
                # print (text + "\n\n")

                    
                # print (tokens)

                text_out.write(" ".join([str(token) for token in tokens_text]) + "\n")
                topic_out.write(" ".join([str(token) for token in tokens_topic]) + "\n")
                logic_interpret_out.write(" ".join([str(token) for token in tokens_interpret]) + "\n")
                logic_out.write(" ".join([str(token) for token in tokens_logic]) + "\n")
                header_out.write(" ".join([str(token) for token in tokens_header]) + "\n")
                table_out.write(" ".join([str(token) for token in tokens_table]) + "\n")


                original_out_text.write(text + "\n")



        text_out.close()
        topic_out.close()
        logic_interpret_out.close()
        logic_out.close()
        header_out.close()

        original_out_text.close()



def test_split_for_rouge(subdir):
    '''
    split valid and test data for rouge eval
    '''

    for set in ["valid", "test"]:
        original = os.path.join(subdir, "original_data", set + ".json")
        os.mkdir(os.path.join(subdir, "test_split_for_" + set))

        k = 0
        with open(original) as f:
            data = json.load(f)
            for tup in data:
                text = tup["sent"]
                topic = tup["topic"]
                logic_interpret = tup["interpret"]


                this_name = str(k) + "_gold.txt"
                with open(os.path.join(subdir, "test_split_for_" + set, this_name), "w") as f_out:
                    f_out.write(text + "\n")

                k += 1



def preprocess(subdir):


    #### convert to id
    print("process data for input...")
    time_start = time.time()

    text2id(subdir)

    duration = time.time() - time_start
    print("finished in %.3f seconds" % float(duration))



    #### split for rouge evaluation
    print("process split for rouge ...")
    time_start = time.time()

    test_split_for_rouge(subdir)

    duration = time.time() - time_start
    print("finished in %.3f seconds" % float(duration))




def make_dirs(subdir):
    """
    Make directoies
    Args:
        subdir: Root directory

    Returns:
        None
    """
    # os.mkdir(os.path.join(subdir, "original_data"))
    os.mkdir(os.path.join(subdir, "processed_data"))
    os.mkdir(os.path.join(subdir, "processed_data", "train"))
    os.mkdir(os.path.join(subdir, "processed_data", "test"))
    os.mkdir(os.path.join(subdir, "processed_data", "valid"))



if __name__ == '__main__':


    root_path = sys.argv[1]
    gpt_path = sys.argv[2]

    enc = encoder.get_encoder("117M", gpt_path)

    subdir = root_path
    make_dirs(subdir)
    preprocess(subdir)
    print("check done")























