#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import encoder
import os


class Preprocessor:
    def __init__(self, data_dir, limits, eos, empty, bpe_enc, max_text_len, max_input_len):
        """
        Main dataloader
        Args:
            data_dir: str, path to data directory
            limits:
            eos: str, eos character
            empty:
        """
        self.data_dir = data_dir

        self.limits = limits
        self.man_text_len = max_text_len
        self.man_input_len = max_input_len
        self.eos = eos
        self.empty = empty
        start_time = time.time()
        self.enc = encoder.get_encoder("117M", bpe_enc)

        print('Reading datasets ...')
        self.train_set = self.load_data('train')
        self.test_set = self.load_data('test')
        self.dev_set = self.load_data('valid')
        print('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))
        print (self.data_dir)



    def load_file(self, file_path):
        """
        Load file, limit to self.limits lines, convert to list of lists
        Args:
            file_path: str, file path

        Returns:
            List of lists of tokens
        """
        data = open(file_path).read().strip().split('\n')
        if self.limits > 0:
            data = data[:self.limits]
        print (file_path)
        print(len(data))
        print(data[0].strip().split(' '))
        sample = list(map(int, data[0].strip().split(' ')))
        sample_res = self.enc.decode(sample)
        print (sample_res + "\n")
        for d in data:
            for h in d.strip().split(" "):
                if h == "":
                    print ("error")
                    print (d)
                    exit(0)

        d = [list(map(int, d.strip().split(' '))) for d in data]

        return d

    def load_data(self, split):
        """
        Load all data
        Args:
            split: str, one of 'train', 'test' or 'valid'

        Returns:
            Dict of data
        """
        subdir = os.path.join(self.data_dir, split)
        file_path_suffixes = {'text': '_text.id',
                              'topic': '_topic.id',
                              'logic_interpret': '_logic_interpret.id',
                              'logic': '_logic.id',
                              'header': '_header.id',
                              'table': '_table.id'}

        all_data = {}
        for fp in file_path_suffixes.keys():
            file_path = os.path.join(subdir, split + file_path_suffixes[fp])
            all_data[fp] = self.load_file(file_path)

        return all_data


class DataLoader:
    def __init__(self, use_table, data, bpe_enc, batch_size=64, shuffle=True, man_text_len=50,
                 man_input_len=300, man_table_len=200, eos=50256, empty=28920):
        """
        Main dataloader
        Args:
            data_dir: dict, all the data
            batch_size: int, batch size
            shuffle: bool, Whether to shuffle data
            domain: str, domain name
        """
        self.data = data
        self.batch_size = batch_size
        self.use_table = use_table
        self.man_text_len = man_text_len
        self.man_input_len = man_input_len
        self.man_table_len = man_table_len
        self.eos = eos
        self.empty = empty
        self.data_size = len(data['text'])
        self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
            else int(self.data_size / batch_size) + 1
        if shuffle:
            self.shuffle_all_data()
        self.count = 0

        self.enc = encoder.get_encoder("117M", bpe_enc)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.num_batches:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        """
        Shuffle all data
        Returns:
            None
        """
        data_size = len(self.data['text'])
        shuffle_indices = np.random.permutation(np.arange(data_size))
        for fp in self.data.keys():
            self.data[fp] = np.array(self.data[fp])[shuffle_indices]
        return

    def get_zipped_batch(self, data, start_index, end_index):
        """
        Get zipped batch of data given start and end index
        Args:
            data: Dict of data
            start_index: int, start index
            end_index: int, end index

        Returns:
            Iterable of batch data
        """
        return zip(data['text'][start_index:end_index],
                   data['topic'][start_index:end_index],
                   data['logic_interpret'][start_index:end_index],
                   data['logic'][start_index:end_index],
                   data['header'][start_index:end_index],
                   data['table'][start_index:end_index])

    def get_batch(self):
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        self.count += 1
        # print (self.count)

        max_topic_len = max([len(sample) for sample in self.data['topic'][start_index:end_index]])
        max_text_len = max([len(sample) for sample in self.data['text'][start_index:end_index]])

        max_logic_interpret_len = max([len(sample) for sample in self.data['logic_interpret'][start_index:end_index]])
        max_logic_len = max([len(sample) for sample in self.data['logic'][start_index:end_index]])

        max_header_len = max([len(sample) for sample in self.data['header'][start_index:end_index]])
        max_table_len = max([len(sample) for sample in self.data['table'][start_index:end_index]])

        max_input_len = max_topic_len + max_header_len + max_logic_len + 7

        if self.use_table:
            max_input_len += min(self.man_table_len, max_table_len)

        batch_data = {'enc_in': [], 'enc_len': [], 'dec_in': [], 'dec_len': [], 'dec_out': [], 'gpt_context': []}

        data_subset = self.get_zipped_batch(self.data, start_index, end_index)

        for text, topic, logic_interpret, logic, header, table in data_subset:

            # target text
            text_len = len(text)
            gold = text + [self.eos] * (max_text_len - text_len + 1)
            text = text + [self.eos] * (max_text_len - text_len)
            # OOM
            if max_text_len > self.man_text_len:
                gold = gold[:self.man_text_len + 1]
                text = text[:self.man_text_len]
                text_len = min(text_len, self.man_text_len)


            # inout
            period, _ = self.enc.encode(" . ")

            # full
            input = topic + period[:] + header + period[:]

            # # no title
            # input = header + period[:]

            # # no header
            # input = topic + period[:]

            # comment if test table only
            if self.use_table:

                table_len = len(table)
                table = table + [self.empty] * (max_table_len - table_len)

                if max_table_len > self.man_table_len:
                    table = table[:self.man_table_len]

                input += (table + period[:])


            # use raw
            # comment if test no logic
            input += logic



            input_len = len(input)
            input = [self.empty] * (max_input_len - input_len) + input


            ### for summarization. ref to gpt2 paper
            gpt_context = " description: "

            gpt_context, _ = self.enc.encode(gpt_context)


            batch_data['enc_in'].append(input)  
            batch_data['enc_len'].append(input_len)  
            batch_data['dec_in'].append(text)  # summary
            batch_data['dec_len'].append(text_len)  # summary len
            batch_data['dec_out'].append(gold)  # padded summary
            batch_data['gpt_context'].append(gpt_context)  

        return batch_data


