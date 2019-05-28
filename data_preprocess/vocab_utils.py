# some parts/ideas taken from:
# • https://github.com/abisee/cs224n-win18-squad
# • Arumugam, R., Shanmugamani. R., Hands-On Natural Language Processing with Python, 2018, ISBN 9781789139495, Packt Publishing Limited.

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from .utils import Cleaners, Counter

_PAD, _UNK, _SOS, _EOS = '<pad>', '<unk>', '<sos>', '<eos>'
_SPECIAL_TOKENS = [_PAD, _UNK, _SOS, _EOS]
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
vocab_size = int(26831)

glove_path = 'embeddings/glove.bin' # todo: need to agree a file path and add comment for download location

def get_vectors(vector_path, vec_dim):
    #todo: explanation needed, possibly want to make this function more generic or
    # convert to class w/ configurable embedding type etc
    print("Loading vectors from file: {}s".format(vector_path))
    emb_matrix = np.zeros((vocab_size + len(_SPECIAL_TOKENS), vec_dim))
    tok2int = {}
    int2tok = {}

    random_init = True  # for special tokens
    if random_init:
        emb_matrix[:len(_SPECIAL_TOKENS), :] = np.random.randn(len(_SPECIAL_TOKENS), vec_dim)

    i = 0
    for token in _SPECIAL_TOKENS:
        tok2int[token] = i
        int2tok[i] = token
        i += 1

    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            emb_matrix[idx, :] = vector
            tok2int[tok] = idx
            int2tok[idx] = word
            idx += 1
    final_vocab_size = vocab_size + len(_SPECIAL_TOKENS)
    assert len(tok2int) == final_vocab_size
    assert len(int2tok) == final_vocab_size
    assert i == final_vocab_size
    # ------------------------------
    # tok2int, int2tok, emb_matrix - with
    # ------------------------------
    return emb_matrix, tok2int, int2tok




def count_words(words_dict, text):
    for sentence in text:
        for word in sentence.split():
            if word not in words_dict:
                words_dict[word] = 1
            else:
                words_dict[word] += 1


def get_tokens_maps(vector_path, vec_dim, text_input=None):
    #todo: explanation needed; can we make this function more general?
    # I.
    if text_input is None:
        print("Loading vectors from file: {}s".format(vector_path))
        emb_matrix = np.zeros((vocab_size + len(_SPECIAL_TOKENS), vec_dim))
        tok2int = {}
        int2tok = {}
        emb_index = {}

        random_init = True
        # randomly initialize the special tokens
        if random_init:
            emb_matrix[:len(_SPECIAL_TOKENS), :] = np.random.randn(len(_SPECIAL_TOKENS), vec_dim)

        # put start tokens in the dictionaries
        i = 0
        for token in _SPECIAL_TOKENS:
            tok2int[token] = i
            int2tok[i] = token
            i += 1

        with open(vector_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=vocab_size):
                line = line.lstrip().rstrip().split(" ")
                tok = line[0]
                vector = np.asarray(line[1:], dtype='float32')
                emb_index[tok] = vector
                tok2int[tok] = i
                int2tok[i] = tok
                i += 1

    # II.
    elif text_input is not None:
        print("Loading vectors from file: {} and a given corpus to extract vocabulary from".format(vector_path))
        word_counts_dict = {}
        count_words(word_counts_dict, text_input)

        emb_matrix = np.zeros((vocab_size + len(_SPECIAL_TOKENS), vec_dim))
        tok2int = {}
        int2tok = {}
        emb_index = {}

        random_init = True
        # randomly initialize the special tokens
        if random_init:
            emb_matrix[:len(_SPECIAL_TOKENS), :] = np.random.randn(len(_SPECIAL_TOKENS), vec_dim)

        # put start tokens in the dictionaries
        i = 0
        for token in _SPECIAL_TOKENS:
            tok2int[token] = i
            int2tok[i] = token
            i += 1

        with open(vector_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=vocab_size):
                line = line.lstrip().rstrip().split(" ")
                tok = line[0]
                vector = np.asarray(line[1:], dtype='float32')
                if tok in word_counts_dict:
                    emb_index[tok] = vector
                    tok2int[tok] = i
                    int2tok[i] = tok
                    i += 1
    # ------------------------------
    # tok2int, int2tok, emb_matrix
    # ------------------------------
    return word_counts_dict, tok2int, int2tok, emb_index



def get_tokens_maps_bkp(vector_path, vec_dim, vocab_size):
    #todo: still used?
    print("Loading vectors from file: {}s".format(vector_path))
    emb_matrix = np.zeros((vocab_size + len(_SPECIAL_TOKENS), vec_dim))
    tok2int = {}
    int2tok = {}
    emb_index = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_SPECIAL_TOKENS), :] = np.random.randn(len(_SPECIAL_TOKENS), vec_dim)

    # put start tokens in the dictionaries
    i = 0
    for token in _SPECIAL_TOKENS:
        tok2int[token] = i
        int2tok[i] = token
        i += 1

    with open(vector_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            tok = line[0]
            vector = np.asarray(line[1:], dtype='float32')
            emb_index[tok] = vector
            tok2int[tok] = i
            int2tok[i] = tok
            i += 1
    # ------------------------------
    # tok2int, int2tok, emb_matrix POS
    # ------------------------------
    return tok2int, int2tok, emb_index



def get_tokens_maps_pos(vector_path, vec_dim, vocab_size):
    # todo: still used? I notice that all of the outputs from this function are also provided by the get_tokens_maps() function
    print("Loading vectors from file: {}s".format(vector_path))
    emb_matrix = np.zeros((vocab_size + len(_SPECIAL_TOKENS), vec_dim))
    tok2int = {}
    int2tok = {}
    emb_index = {}

    # put start tokens in the dictionaries
    i = 0

    with open(vector_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            tok = line[0]
            vector = np.asarray(line[1:], dtype='float32')
            emb_index[tok] = vector
            tok2int[tok] = i
            int2tok[i] = tok
            i += 1

    return tok2int, int2tok, emb_index


def get_emb_matrix(tok2int, emb_index, embedding_dim):
    nwords = len(tok2int)
    emb_matrix = np.zeros((nwords, embedding_dim), dtype=np.float32)
    for word, i in tok2int.items():
        if word in emb_index:
            length_test = len(emb_index[word])
            if length_test != embedding_dim:
                continue
            emb_matrix[i] = emb_index[word]
        else:
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            emb_matrix[i] = new_embedding
    return emb_matrix


def get_embedding_index(vector_file):
    embedding_index = {}
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            line = line.split()
            word = line[0]
            embedding = np.asarray(line[1:], dtype='float32')
            embedding_index[word] = embedding
    return embedding_index


def process_encoding_input(target_data, word2int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    decoding_input = tf.concat([tf.fill([batch_size, 1], word2int[_SOS]), ending], 1)
    return decoding_input


# ----------------
#
# ------------------
def make_hashtables(list_in):
    int_to_token = {i: token for i, token in enumerate(list_in)}
    token_to_int = {token: i for i, token in int_to_token.items()}
    return int_to_token, token_to_int


def tokens_to_dictionaries(list_of_tokens, take_top=None):
    if take_top:
        count_tuples = Counter(list_of_tokens).most_common(take_top)
    else:
        count_tuples = Counter(list_of_tokens).most_common()
    count_dict = {token: count for token, count in count_tuples}

    int_to_token_map, token_to_int_map = make_hashtables(list(count_dict.keys()))
    vocab = [i for i in count_dict.keys()]
    print('| Vocabulary Size: {}'.format(len(vocab)))


    return vocab, count_dict, int_to_token_map, token_to_int_map