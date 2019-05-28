""" File defines handlers for various text-based corpuses """

import json
from os.path import join, abspath, dirname

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
        self.data = self.get_data()
        self.titles = self.get_titles()

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

    def get_titles(self, data=None):
        """ Method returns a list of paragraph titles"""
        if data == None:
            data = self.data

        return [p['title'] for p in data]

    def _get_index_for_title(self, title):
        if type(title) == int:
            return title
        elif type(title) == str:
            return self.titles.index(title)
        else:
            print('_get_index_for_title not implemented for title type {}'.format(type(title)))

    def get_all_paragraphs(self):
        return [p['paragraphs'] for p in self.data]

    def get_paragraphs(self, idx=None):
        if idx is None:
            return self.get_all_paragraphs()
        else:
            return self.get_all_paragraphs()[idx]

    def get_context_for_paragraph(self, index):
        return [p['paragraphs'] for p in self.get_data()]

    def get_contexts_bkp(self, json_in, sentence_tokenize=False):
        """
        # ------------------------------------
        # Original Function used for the MVP
        # ------------------------------------
        # This was used for first inspections
        :param json_in:
        :param sentence_tokenize:
        :return:
        """
        with open(json_in) as file:
            contexts_list = []
            data = json.load(file)
            for i in range(len(data['data'])):
                data['data'][i]['title']

                for j in range(len(data['data'][i]['paragraphs'])):
                    if sentence_tokenize == True:
                        contexts_list.append(sent_tokenize(data['data'][i]['paragraphs'][j]['context']))
                    else:
                        contexts_list.append(data['data'][i]['paragraphs'][j]['context'])
            return contexts_list

    def get_squad_topic(self, json_in, i=0, sentence_tokenize=False, word_tokenizer=None, cleaner=None):
        """
        # ------------------
        # Read SQuAD Topic
        # ------------------
        # This function contains  conditional settings.
        # It is quite heavy to read. I am sorry for this experirnce.
        # It partially is because of the original naming convention. It works, but it could be a bit over-engineered. RACE data read has been simplified.
        # By Convention {i} is an element of the root, a distinct thematic part.
        # It reads a single SQuAD topic and applies relevanant transformations and returns the content organized in lists.
        :param json_in:
        :param i:
        :param sentence_tokenize:
        :param word_tokenizer:
        :param cleaner:
        :return:
        """
        with open(json_in) as file:
            data = json.load(file)
            topic = data['data'][i]
            paragraphs_list = []
            # print('| SELECTED TOPIC TITLE: {}'.format(topic['title']))
            # print('| SELECTED TOPIC IDX: {}'.format(i))
            # print('| ALL TOPICS: {}'.format(len(data['data'])))

            for j in range(len(data['data'][i]['paragraphs'])):
                questions_text = []
                questions_id = []
                questions_isimp = []
                answers_text = []

                # --------------------------------------
                # 1st IF. Tokenize into sentences only
                # --------------------------------------
                if (sentence_tokenize == True) & (word_tokenizer == None):
                    for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                        answers_list = []
                        if cleaner:
                            questions_text.append(
                                cleaner.text_cleaner(data['data'][i]['paragraphs'][j]['qas'][k]['question']))
                            questions_id.append(data['data'][i]['paragraphs'][j]['qas'][k]['id'])
                            questions_isimp.append(data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])
                        else:
                            questions_text.append(data['data'][i]['paragraphs'][j]['qas'][k]['question'])
                            questions_id.append(data['data'][i]['paragraphs'][j]['qas'][k]['id'])
                            questions_isimp.append(data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])

                        if data['data'][i]['paragraphs'][j]['qas'][k]['answers']:
                            if cleaner:
                                answers_text.append(
                                    cleaner.text_cleaner(
                                        data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text']))
                            else:
                                answers_text.append(data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])
                        else:
                            answers_text.append(('<no_answer>', '<no_pos>'))
                            # put everything together
                    if cleaner:
                        paragraphs_item = [
                            cleaner.list_cleaner(sent_tokenize(data['data'][i]['paragraphs'][j]['context'])),
                            questions_text, questions_id, questions_isimp, answers_text]
                    else:
                        paragraphs_item = [sent_tokenize(data['data'][i]['paragraphs'][j]['context']),
                                           questions_text, questions_id, questions_isimp, answers_text]

                # -----------------------------------------------------------------------------
                # 2nd IF. Tokenize into sentences into tokens using provided tokenizer object
                # -----------------------------------------------------------------------------
                elif (sentence_tokenize == True) & (word_tokenizer is not None):
                    for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                        answers_list = []
                        if cleaner:
                            questions_text.append(word_tokenizer.nltk_regex(
                                cleaner.text_cleaner(data['data'][i]['paragraphs'][j]['qas'][k]['question'])))
                            # print(questions_text)
                            questions_id.append(data['data'][i]['paragraphs'][j]['qas'][k]['id'])
                            questions_isimp.append(data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])
                        else:
                            questions_text.append(data['data'][i]['paragraphs'][j]['qas'][k]['question'])
                            questions_id.append(data['data'][i]['paragraphs'][j]['qas'][k]['id'])
                            questions_isimp.append(data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])

                        if data['data'][i]['paragraphs'][j]['qas'][k]['answers']:
                            if cleaner:
                                answers_text.append(word_tokenizer.nltk_regex(
                                    cleaner.text_cleaner(
                                        data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])))
                            else:
                                answers_text.append(data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])
                        else:
                            answers_text.append(('<no_answer>', '<no_pos>'))

                    if cleaner:
                        paragraphs_item = [word_tokenizer.nltk_regex_list(
                            cleaner.list_cleaner(sent_tokenize(data['data'][i]['paragraphs'][j]['context']))),
                            questions_text, questions_id, questions_isimp, answers_text]
                    else:
                        paragraphs_item = [
                            word_tokenizer.nltk_regex_list(sent_tokenize(data['data'][i]['paragraphs'][j]['context'])),
                            questions_text, questions_id, questions_isimp, answers_text]
                # -----------------------------------------------------------------------------------------
                # 3rd IF. Do not tokenize sentences. Tokenize into tokens using provided tokenizer object
                # -----------------------------------------------------------------------------------------
                elif (sentence_tokenize == False) & (word_tokenizer is not None):
                    for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                        answers_list = []
                        if cleaner:
                            questions_text.append(word_tokenizer.nltk_regex(
                                cleaner.text_cleaner(data['data'][i]['paragraphs'][j]['qas'][k]['question'])))
                            questions_id.append(data['data'][i]['paragraphs'][j]['qas'][k]['id'])
                            questions_isimp.append(data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])
                        else:
                            questions_text.append(data['data'][i]['paragraphs'][j]['qas'][k]['question'])
                            questions_id.append(data['data'][i]['paragraphs'][j]['qas'][k]['id'])
                            questions_isimp.append(data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])

                        if not data['data'][i]['paragraphs'][j]['qas'][k]['answers']:
                            answers_text.append([('<no_answer>', '<no_pos>')])
                        else:
                            if cleaner:
                                answers_text.append(word_tokenizer.nltk_regex(
                                    cleaner.text_cleaner(
                                        data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])))
                            else:
                                answers_text.append(data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'])

                    if cleaner:
                        paragraphs_item = [
                            word_tokenizer.nltk_regex(
                                cleaner.text_cleaner((data['data'][i]['paragraphs'][j]['context']))),
                            questions_text, questions_id, questions_isimp, answers_text]
                    else:
                        paragraphs_item = [word_tokenizer.nltk_regex((data['data'][i]['paragraphs'][j]['context'])),
                                           questions_text, questions_id, questions_isimp, answers_text]

                # -----------------------------------
                # 4th ELSE. Don't tokenize anything
                # -----------------------------------
                else:
                    for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                        answers_list = []
                        if cleaner:
                            questions_text.append(
                                cleaner.text_cleaner(data['data'][i]['paragraphs'][j]['qas'][k]['question']))
                            questions_id.append(data['data'][i]['paragraphs'][j]['qas'][k]['id'])
                            questions_isimp.append(data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])
                        else:
                            questions_text.append(data['data'][i]['paragraphs'][j]['qas'][k]['question'])
                            questions_id.append(data['data'][i]['paragraphs'][j]['qas'][k]['id'])
                            questions_isimp.append(data['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])

                        for l in range(len(data['data'][i]['paragraphs'][j]['qas'][k]['answers'])):
                            if cleaner:
                                answers_list.append(cleaner.text_cleaner(
                                    data['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['text'][0]))
                            else:
                                answers_list.append(data['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['text'][0])

                        answers_text.append(('<no_answer>', '<no_pos>'))

                    if cleaner:
                        paragraphs_item = [cleaner.text_cleaner(data['data'][i]['paragraphs'][j]['context']),
                                           questions_text, questions_id, question_isimp, answers_text]
                    else:
                        paragraphs_item = [data['data'][i]['paragraphs'][j]['context'],
                                           questions_text, questions_id, question_isimp, answers_text]
                paragraphs_list.append(paragraphs_item)
        return paragraphs_list, topic['title']

    def get_data(self):
        """
        Method returns all data in corpus
        :return: data (list of dicts)
        """
        with open(self.data_path, 'r') as fi:
            return json.loads(fi.read())['data']

    def get_data_item(self, index:int):
        """
        Method returns one item of data from the corpus
        :param index: index requested
        :return: data item (dict)
        """
        return self.get_data()[index]


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

