""" File defines handlers for various text-based corpuses """

import json
from os.path import join, abspath, dirname, isfile

from .utils import sent_tokenize

data_dir = abspath(join(dirname(__file__), '../data/'))

class BaseCorpusHandler:
    """
    Base class for corpus handlers: class should define base methods and properties
    which can be overridden for corpus-specific tweaks and actions
    """

    def __init__(self, data_path=None):
        self.data_path = data_path if data_path is not None else self._default_data_path

    @property
    def regex_list(self):
        return []

    @property
    def _default_data_path(self):
        return join(data_dir, 'example_data_dir')


class SQuADCorpusHandler(BaseCorpusHandler):
    """ Class defines methods for handling SQuAD corpus """
    def __init__(self, data_path=None):
        super().__init__() # run __init__() method of the base class
        self.data = None
        # self.titles = self.get_titles()

    @property
    def regex_list(self):
        return [
            (r',', " , "),
            (r"\.", " . "),
            (r"\(|\[|\{", " ( "),
            (r"\)|\]|\}", " ) "),
            (r'"', ' " '),
            (r"``|''|`", ""),  # recommended to remove by authors
            (r"'s|'d", ""),
            (r"'", ""),
            (r":|;|\?|\\|\/|\!", ""),
            (r"(R|r)\&(B|b)", "rnb"),
            (r"(U\.|u\.)(S\.|s\.)", "usa"),
            (r"-", "_"),
            (r"__+", "_"),
            (r"\s\s+", " ")]  # merge multiple white spaces

    @property
    def _default_data_path(self):
        return join(data_dir, 'train-v2.0.json')

    @_default_data_path.setter
    def _default_data_path(self, value):
        self._default_data_path = value

    def load_data(self, force_reload=True):
        """
        Method loads data to the .data attribute
        :param force_reload:
        :return:
        """
        if (self.data is None) or force_reload:
            self.data = self.get_data(force_reload=True)

    def get_data(self, force_reload=False):
        """
        Method returns all data in corpus
        :return: data (list of dicts)
        """
        assert isfile(self.data_path), "Attempting to load from file which does not exist: {} \n " \
                                       "Please either create this file or point the SQuADCorpusHandler.data_path " \
                                       "attribute to a valid SQuAD json file".format(self.data_path)
        if (self.data is None) or force_reload:
            with open(self.data_path, 'r') as fi:
                return json.loads(fi.read())['data']
        else:
            return self.data

    def get_data_item(self, index:int, force_reload=False):
        """
        Method returns one item of data from the corpus
        :param index: index requested
        :return: data item (dict)
        """
        return self.get_data(force_reload=force_reload)[index]

    def get_titles(self, force_reload=False):
        """ Method returns a list of paragraph titles"""
        data = self.get_data(force_reload=force_reload)
        return [p['title'] for p in data]

    def _get_index_for_title(self, title):
        if type(title) == int:
            return title
        elif type(title) == str:
            return self.titles.index(title)
        else:
            print('_get_index_for_title not implemented for title type {}'.format(type(title)))

    def get_all_paragraphs(self, force_reload=False):
        """
        Method returns a list of all paragraphs in corpus
        ie. all paragraphs for all topics in a single list
        :param force_reload:
        :return:
        """
        data = self.get_data(force_reload=force_reload)
        return [p['paragraphs'] for p in data]

    def get_single_paragraph(self, idx=None, force_reload=False):
        """
        Method returns paragraphs from corpus.
        :param idx: paragraph index. If None, return all.
        :return: single paragraph
        """
        return self.get_all_paragraphs(force_reload=force_reload)[idx]

    def get_all_contexts(self, sentence_tokenize=False, force_reload=False):
        """
        Method gets all contexts from all paragraphs
        :param sentence_tokenize: bool: if True, return nested list of tokenised sentences for each context. If false,
        return list of contexts as single strings.
        :param force_reload:
        :return:
        """
        contexts =  [c['context'] for p in self.get_all_paragraphs(force_reload=force_reload) for c in p]

        if sentence_tokenize:
            contexts = [sent_tokenize(c) for c in contexts]

        return contexts

    def get_single_topic(self, idx, sentence_tokenize=False, word_tokenizer=None, cleaner=None, force_reload=False):
        """
        Method to get a single SQuAD topic, ie many paragraphs on a single topic.
        Each paragraph contains a number of contexts
        Each context corresponds to a number of questions
        :param idx:
        :param sentence_tokenize: bool
        :param word_tokenizer: function or class implementing method tokenize()
        :param cleaner: object of Cleaners class, implementing method text_cleaner
        :param force_reload: bool: force reload of data
        :return:
        """

        paragraphs_list = []

        for p in self.get_single_paragraph(idx=idx, force_reload=force_reload):
            questions_text = []
            questions_id = []
            questions_isimp = []
            answers_text = []

            # Extract raw data
            for q in p['qas']:
                questions_text.append(q['question'])
                questions_id.append(q['id'])
                questions_isimp.append(q['is_impossible'])

                if q['answers']:
                    # if answer exists, append it: else add <no_answer> tag
                    answers_text.append(q['answers'][0]['text'])
                else:
                    answers_text.append(('<no_answer>', '<no_pos>'))

            context = p['context'] # start with raw text
            # If cleaner specified, apply cleaner to raw text strings: string -> string
            if cleaner:
                assert hasattr(cleaner, 'text_cleaner'), "Cleaner object must implement method text_cleaner, str -> str"
                questions_text = [cleaner.text_cleaner(q) for q in questions_text]
                answers_text = [cleaner.text_cleaner(q) for q in answers_text]
                context = cleaner.text_cleaner(context)

            # If sentence tokenization specified, apply to text strings: string -> list
            if sentence_tokenize:
                context = sent_tokenize(context)
            else:
                context = [context] # even if not tokenising sentences, wrap in a list to simplify downstream processing

            # If word tokenizer specified, apply to list of sentences: list -> nested list
            if word_tokenizer:
                # Allow a tokenizer which is either a function acting on an input string to return a list of strings,
                # or a class with a method tokenize() which does the same
                if not callable(word_tokenizer):
                    assert hasattr(word_tokenizer,'tokenize'), "word tokenizer {} must be callable, or have method .tokenize(), str -> list"
                    word_tokenize_fun = word_tokenizer.tokenize
                else:
                    word_tokenize_fun = word_tokenizer

                questions_text = [word_tokenize_fun(q) for q in questions_text]
                answers_text = [word_tokenize_fun(q) for q in answers_text]
                context = [word_tokenize_fun(cs) for cs in context] # expecting a sentence-tokenised representation here

            # Append postprocessed paragraph data to results list
            paragraphs_list.append([
                context, questions_text, questions_id, questions_isimp, answers_text
            ])

        return paragraphs_list

    def cqa_triplets_for_topic(self, topic_idx, **kwargs):
        topic = self.get_single_topic(idx=topic_idx, **kwargs)
        contexts, questions, answers = [], [], []
        for i in range(len(topic)):
            for j in range(len(topic[i][1])):
                contexts.append(topic[i][0])
                questions.append(topic[i][1][j])
                if not topic[i][4][j]:
                    answers.append('ERR')
                else:
                    ans = topic[i][4][j]
                    assert type(ans) in [str, list], \
                        "Unable to handle answer of data type {} (value {})".format(type(ans), ans)
                    if type(ans) == str:
                        answers.append(ans)
                    elif type(ans) == list:
                        answers.append(ans[0])

        return contexts, questions, answers



class RaceCorpusHandler(BaseCorpusHandler):
    """ Class defines methods for handling SQuAD corpus """
    def __init__(self):
        super().__init__() # run __init__() method of the base class

    @property
    def regex_list(self):
        return [
            (r',', " , "),
            (r"\.", " . "),
            (r"\(|\[|\{", " ( "),
            (r"\)|\]|\}", " ) "),
            (r'"', ' " '),
            (r"``|''|`", ""),  # recommended to remove by authors
            (r"'s|'d", ""),
            (r"'", ""),
            (r":|;|\?|\\|\/|\!|#", ""),
            (r"(R|r)\&(B|b)", "rnb"),
            (r"(U\.|u\.)(S\.|s\.)", "usa"),
            (r"-", "_"),
            (r"__+", "_"),
            (r"\s\s+", " ")]  # merge multiple white spaces

