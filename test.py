import json
from nlp.corpus_handlers import SQuADCorpusHandler

if __name__ == '__main__':
    corpus = SQuADCorpusHandler()

    thing = corpus.get_paragraphs()
    thing2 = corpus.get_paragraphs(idx=0)
    thing2 = corpus.get_contexts_bkp()
    print(data.keys())

    print('done')