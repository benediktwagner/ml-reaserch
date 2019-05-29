# Text Preprocessing

# Inspiration for the cleaner as a class with regex has came from:
# Perkins, J., 2014, Python 3 Text Processing with NLTK 3 Cookbook, ISBN 9781782167853 Packt Publishing Limited.
# It is also a not bad book.

import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import unicodedata
import gensim
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, RegexpTokenizer, MWETokenizer
from nltk.tag import pos_tag
from .vocab_utils import vocab_size, PAD_ID, UNK_ID, SOS_ID, EOS_ID, _UNK, _PAD, _SOS, _EOS, make_hashtables


# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
# from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

# ------------------
# Class Cleaners()
# ------------------
# The Class takes a regex map (a list of tuples) for initialization.
# Methods have been created to conveniently clean content at three aggregatoion levels
class Cleaners(object):
    def __init__(self, regex_list, lowercase=False):
        self.regex_list = [(re.compile(regex), replacement) for (regex, replacement) in regex_list]
        self.lowercase = lowercase

    # Cleans string
    def text_cleaner(self, text):
        for (pattern, replacement) in self.regex_list:
            text = re.sub(pattern, replacement, text)
            text = replace_unicode(text)
            if self.lowercase == True:
                text = text.lower()
        return text

    # Cleans a list of strings
    def list_cleaner(self, list_of_texts):
        clean_list = []
        for i in list_of_texts:
            clean_text = self.text_cleaner(i)
            clean_list.append(clean_text)
        return clean_list

    # Cleans a list of lists of strings
    def paragraphs_cleaner(self, list_of_paragraphs):
        clean_paragraphs = []
        for paragraph in list_of_paragraphs:
            clean_sequences = []
            for sequence in paragraph:
                clean_sequence = self.text_cleaner(sequence)
                clean_sequences.append(clean_sequence)
            clean_paragraphs.append(clean_sequence)
        return clean_paragraphs


# --- End RegexCleaners() ---

# ---------------------
# Class Tokenizers()
# ---------------------
# Creates a tokenizer class.
class Tokenizers(object):
    tokenizer = RegexpTokenizer('\s+', gaps=True)  # split on whitespace

    def __init__(self, n_gram=1, lowercase=False, stopwords=None, lemma=False, pos_tagger=None):
        self.n_gram = n_gram
        self.lowercase = lowercase
        self.stopwords = stopwords
        self.lemma = lemma
        self.pos_tagger = pos_tagger

    # Tokenize with all settings. The n-grams loop could be done better but it was complex due to other parameters and for this application was good.
    def nltk_regex(self, text):
        if self.n_gram == 1:
            if self.lowercase == True:
                text = text.lower()  # else text
            tokens = Tokenizers.tokenizer.tokenize(text)
            if self.stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]  # else: tokens = tokens
            if self.lemma == True:
                tokens = [get_lemma(i) for i in tokens]
            n_gram_iterator = zip(*[tokens[i:] for i in range(int(self.n_gram))])

            tokens_list = list(n_gram_iterator)
            tokens_list = [list(i) for i in tokens_list]
            tokens_list_flat = flat_list(tokens_list)
            if self.pos_tagger != None:
                return self.pos_tagger(tokens_list_flat)
            else:
                return tokens_list_flat

        elif self.n_gram == 2:
            if self.lowercase == True:
                text = text.lower()  # else: text = text (Python works with IF statement only)
            tokens = Tokenizers.tokenizer.tokenize(text)
            if self.stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]  # else: tokens = tokens
            if self.lemma == True:
                tokens = [get_lemma(i) for i in tokens]
            n_gram_iterator = zip(*[tokens[i:] for i in range(int(self.n_gram))])
            tokens_list = list(n_gram_iterator)
            if self.pos_tagger != None:
                return [self.pos_tagger(list(i)) for i in tokens_list]
            else:
                return [' '.join(list(i)) for i in tokens_list]

        elif self.n_gram == 3:
            if self.lowercase == True:
                text = text.lower()  # else: text = text (Python works with IF statement only)
            tokens = Tokenizers.tokenizer.tokenize(text)
            if self.stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]  # else: tokens = tokens
            if self.lemma == True:
                tokens = [get_lemma(i) for i in tokens]
            n_gram_iterator = zip(*[tokens[i:] for i in range(int(self.n_gram))])
            tokens_list = list(n_gram_iterator)
            tokens_list = [' '.join(list(i)) for i in tokens_list]
            if self.pos_tagger != None:
                return [self.pos_tagger(list(i)) for i in tokens_list]
            else:
                return [' '.join(list(i)) for i in tokens_list]

        elif self.n_gram == 4:
            if self.lowercase == True:
                text = text.lower()  # else: text = text (Python works with IF statement only)
            tokens = Tokenizers.tokenizer.tokenize(text)
            if self.stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]  # else: tokens = tokens
            if self.lemma == True:
                tokens = [get_lemma(i) for i in tokens]
            n_gram_iterator = zip(*[tokens[i:] for i in range(int(self.n_gram))])
            tokens_list = list(n_gram_iterator)
            tokens_list = [' '.join(list(i)) for i in tokens_list]
            if self.pos_tagger != None:
                return [self.pos_tagger(list(i)) for i in tokens_list]
            else:
                return [' '.join(list(i)) for i in tokens_list]
        else:
            print('Not Implemented')

    def nltk_regex_list(self, list_of_texts):
        tokenized_list = []
        for i in list_of_texts:
            tokenized_text = self.nltk_regex(i)
            tokenized_list.append(tokenized_text)
        return tokenized_list

    def nltk_regex_sentence(self, list_of_paragraphs):
        paragraphs_out = []
        for paragraph in list_of_paragraphs:
            sentences_out = []
            for sentence in paragraph:
                sentence_out = self.nltk_regex(sentence)
                sentences_out.append(sentence_out)
            paragraphs_out.append(sentences_out)
        return paragraphs_out

    # def multi_words_tokenizer(self):


# ---------------------
# Class Vectorizers()
# ---------------------
class Vectorizers(object):
    def __init__(self, n_gram, vector_type='binary'):
        self.n_gram = n_gram
        self.vector_type = vector_type

    def vectorize(self, corpus):
        if self.vector_type == 'binary':
            vectorizer = CountVectorizer(binary=True, ngram_range=self.n_gram)
            corpus_vec = vectorizer.fit_transform(corpus)
            test = vectorizer.inverse_transform(corpus_vec)
        return (corpus_vec, vectorizer, test)


# --- End Tokenizers() ---


# ------------------
# Additional tools
# ------------------

# Converts list to a string.
def list_to_str(list_of_strings):
    return ' '.join(map(str, list_of_strings))


# Reduces a list of lists to a list.
def flat_list(list_of_lists):
    return [i for row in list_of_lists for i in row]


# Replace non-ASCII characters wit ASCII equivalent, otherwise drop.
def replace_unicode(text_in):
    text_out = unicodedata.normalize('NFD', text_in).encode('ascii', 'ignore')
    text_out = text_out.decode('utf-8')
    return text_out


# Retrieve vocabulary, count dictionary, integer-to-token and token-to-integer maps.
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


# Convert a sequence of tokens to sequence of integers.
def sequences_to_int(sequences, tok2int):
    sequences_int = []
    unk_counts = []
    for sequence in sequences:
        sequence_int = []
        unk_count = 0
        for token in sequence.split():
            if token in tok2int:
                sequence_int.append(tok2int[token])
            else:
                sequence_int.append(tok2int[_UNK])
                unk_count += 1
        unk_counts.append(unk_count)
        sequences_int.append(sequence_int)
    return sequences_int, unk_counts


# Lemmatize.
# The following part of speech tags can be enforced: ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'.
def get_lemma(word_in):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word_in)


# ---------------------------------------
# Make Context-Question-Answer triplets
# ---------------------------------------
# In effect the contects are multplied by the number of respective questions.
# Outputs are lists, all equal lengths.

# The same with POS tags.
def cqa_triplets_pos(topic):
    #todo: refactor these into the corpus handlers. Not sure what this code is doing? Are POS already available in the topic? Unsure where/how these are added
    contexts, questions, answers, cqa_ids = [], [], [], []
    contexts_pos, questions_pos, answers_pos = [], [], []
    for i in range(len(topic)):
        for j in range(len(topic[i][1])):
            # Decouple  (token, pos_tag)
            context_tokens, context_pos = [], []
            for k in range(len(topic[i][0])):
                context_tokens.append(topic[i][0][k][0])
                context_pos.append(topic[i][0][k][1])
            contexts.append(context_tokens)
            contexts_pos.append(context_pos)

            question_tokens, question_pos = [], []
            for k in range(len(topic[i][1][j])):
                question_tokens.append(topic[i][1][j][k][0])
                question_pos.append(topic[i][1][j][k][1])
            questions.append(question_tokens)
            questions_pos.append(question_pos)

            answer_tokens, answer_pos, cqa_id = [], [], []
            for k in range(len(topic[i][4][j])):
                answer_tokens.append(topic[i][4][j][k][0])
                answer_pos.append(topic[i][4][j][k][1])

            answers.append(answer_tokens)
            answers_pos.append(answer_pos)
    return contexts, questions, answers, contexts_pos, questions_pos, answers_pos


def write_to_file(out_file, line):
    out_file.write(str(line.encode('utf-8').decode('utf-8')) + '\n')


# ----------
# DRAWINGS
# ----------
# Common tools have been styled for this work.
# The drawing functions also have on-demand saving.

# Line graph.
def draw_line(x, y, label='items', save=False, name='barplot1.png'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, color='#0070c0', alpha=0.8)
    ax.spines['left'].set_color('#7f7f7f')
    ax.spines['bottom'].set_color('#7f7f7f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.label.set_color('#7f7f7f')
    ax.yaxis.label.set_color('#7f7f7f')
    ax.tick_params(axis='x', colors='#7f7f7f')
    ax.tick_params(axis='y', colors='#7f7f7f')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(label)
    if save:
        plt.savefig(name)
    plt.show()
    return None


# Scatterplot with annotations. This has been replaced with d3.js.
def draw_2dscatter(array1, array2, tokens):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(array1, array2, s=3)
    for i, token in enumerate(tokens):
        plt.annotate(token, xy=(array1[i], array2[i]), fontsize=8)
    plt.show()
    return None


# Histogram.
def draw_histogram(array_in, log_scale=False, bins=50, label='bins', save=False, name='histogram1.png'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(array_in, bins=bins, color='#0070c0', alpha=0.8)
    ax.set_xlabel(label)
    ax.spines['left'].set_color('#7f7f7f')
    ax.spines['bottom'].set_color('#7f7f7f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.label.set_color('#7f7f7f')
    ax.yaxis.label.set_color('#7f7f7f')
    ax.tick_params(axis='x', colors='#7f7f7f')
    ax.tick_params(axis='y', colors='#7f7f7f')
    if log_scale:
        plt.yscale('log')
    plt.grid(True)
    ax.set_axisbelow(True)
    if save:
        plt.savefig(name)
    plt.show()
    return None


# Barplot.
def draw_barplot(items, values, label='items', save=False, name='barplot1.png'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(items, values, color='#0070c0', alpha=0.8)
    ax.spines['left'].set_color('#7f7f7f')
    ax.spines['bottom'].set_color('#7f7f7f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.label.set_color('#7f7f7f')
    ax.yaxis.label.set_color('#7f7f7f')
    ax.set_xticks(items)
    ax.tick_params(axis='x', colors='#7f7f7f')
    ax.tick_params(axis='y', colors='#7f7f7f')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(label)
    if save:
        plt.savefig(name)
    plt.show()
    return None


# =======================
# In progress, not used
# =======================

# ---------------
# POS_Taggers()
# ---------------
class POS_Taggers(object):
    def __init__(self, tager_type='backoff'):
        self.tager_type = tager_type

    def backoff_tagger(sentences, tagger_classes, backoff=None):
        for cls in tagger_classes:
            backoff = cls(sentences, backoff=backoff)
        return backoff

# --- End of file ---