import time, os, sys, shutil, io, subprocess, re
import tensorflow as tf
import numpy as np
import zipfile
import json
import copy

# Progress bar

TOTAL_BAR_LENGTH = 100.
last_time = time.time()
begin_time = last_time
print(os.popen('stty size', 'r').read())
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


def bleu_score(labels_file, predictions_path):
    bleu_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [bleu_script, labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        print(bleu_score)
        return float(bleu_score)

    except subprocess.CalledProcessError as error:
      if error.output is not None:
        msg = error.output.strip()
        tf.logging.warning(
            "{} script returned non-zero exit code: {}".format(bleu_script, msg))
      return None

def read_word2vec_zip(word2vec_file):
    wordvec_map = {}
    num_words = 0
    dimension = 0
    zfile = zipfile.ZipFile(word2vec_file)
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        for line in ifile:
            line = line.strip()
            #print line
            entries = line.split(' ')
            if len(entries) == 2:
                continue
            word = entries[0].strip()
            vec = map(float, entries[1:])

            if word in wordvec_map:
                print ("Invalid word in embedding. Does not matter.")
                continue
            assert dimension == 0 or dimension == len(vec)

            wordvec_map[word] = np.array(vec)
            num_words += 1
            dimension = len(vec)

    return wordvec_map, num_words, dimension

def read_word2vec(word2vec_file):
    wordvec_map = {}
    num_words = 0
    dimension = 0
    with open(word2vec_file, "r") as f:
        for line in f:
            line = line.strip()
            #print line
            entries = line.split(' ')
            if len(entries) == 2:
                continue
            word = entries[0].strip()
            vec = map(float, entries[1:])

            if word in wordvec_map:
                print ("Invalid word in embedding. Does not matter.")
                continue
            # assert word not in wordvec_map
            assert dimension == 0 or dimension == len(vec)

            wordvec_map[word] = np.array(vec)
            num_words += 1
            dimension = len(vec)

    return wordvec_map, num_words, dimension

def load_vocab(vocab_file):
    vocab = {}

    vocab['<_PAD>'] = 0
    vocab['<_START_TOKEN>'] = 1
    vocab['<_END_TOKEN>'] = 2
    vocab['<_UNK_TOKEN>'] = 3

    cnt = 4
    with open(vocab_file, "r") as v:
        for line in v:
            if len(line.strip().split()) > 1:
                word = line.strip().split()[0]
                ori_id = int(line.strip().split()[1])
                if word not in vocab:
                    vocab[word] = (cnt + ori_id)

    return vocab

def create_init_embedding(vocab_file, extend_vocab_size, word2vec_file, emblen):
    '''
    create initial embedding for text relation words.
    words not in word2vec file initialized to random.

    key_map['PAD'] = 0
    key_map['START_TOKEN'] = 1
    key_map['END_TOKEN'] = 2
    key_map['UNK_TOKEN'] = 3
    '''

    vocab = load_vocab(vocab_file)
    print("vocab len: ", len(vocab))

    init_embedding = np.random.uniform(-np.sqrt(3), np.sqrt(3), size = (len(vocab) + extend_vocab_size, emblen))

    if word2vec_file.endswith('.gz'):
        word2vec_map = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    elif word2vec_file.endswith('.zip'):
        word2vec_map, num_words, dimension = read_word2vec_zip(word2vec_file)
    else:
        word2vec_map, num_words, dimension = read_word2vec(word2vec_file)

    num_covered = 0

    for word in vocab:
        if word in word2vec_map:
            vec = word2vec_map[word]
            if len(vec) != emblen:
                raise ValueError("word2vec dimension doesn't match.")
            init_embedding[vocab[word], :] = vec
            num_covered += 1

    unk_vec = init_embedding[3, :]
    for ind in range(len(vocab), len(init_embedding)):
        init_embedding[ind, :] = unk_vec

    ## embedding for pad
    # init_embedding[0][:] = np.zeros(emblen)

    print ("word2vec covered: %d" % num_covered)
    return init_embedding

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def write_word(pred_list, save_dir, name):
    ss = open(save_dir + name, "w+")
    for item in pred_list:
        ss.write(" ".join(item) + '\n')


def get_current_git_version():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')

#### for rouge error:
def check_res(res):
    if res.strip() == "":
        return False
    for token in res.strip():
        if token.isalpha():
            return True

    return False

def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def get_rouge(res_sum):

    sum_words = res_sum.split(" ")
    sum_sents = []
    while len(sum_words) > 0:
        try:
            fst_period_idx = sum_words.index(".")
        except ValueError: # there is text remaining that doesn't end in "."
            fst_period_idx = len(sum_words)
        sent = sum_words[:fst_period_idx+1] # sentence up to and including the period
        sum_words = sum_words[fst_period_idx+1:] # everything else
        sum_sents.append(' '.join(sent))

    sum_sents = [make_html_safe(w) for w in sum_sents]

    return sum_sents

def get_res(ori_file, gen_file, out_file):

    f_ori = open(ori_file)
    f_gen = open(gen_file)

    data = json.load(f_ori)
    generated = f_gen.readlines()

    assert len(data) == len(generated)

    res = []

    for tup_ori, tup_gen in zip(data, generated):

        to_write = copy.deepcopy(tup_ori)
        to_write["generated"] = tup_gen.strip()
        res.append(to_write)

    with open(out_file, "w") as f_out:
        f_out.write(json.dumps(res, indent=4))


















