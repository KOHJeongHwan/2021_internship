import numpy as np 
from glove import Glove, Corpus 
from tqdm import tqdm 

corpus_fname = './step2/processed_wiki_ko_origin.txt' 
corpus_model = './wiki_ko_origin.model' 
glove_model = './glove.model' 
def make_corpus(load=False):
    if load: _corpus = Corpus.load(corpus_model) 
    else:
        corpus = [sent.strip().split(" ") for sent in tqdm(open(corpus_fname, 'r', encoding='utf-8').readlines())]
        _corpus = Corpus()
        _corpus.fit(corpus, window=10)
        _corpus.save(corpus_model)
        
    return _corpus
    
def train(corpus=None, train=True):
    if train:
        print('Dict size: %s' % len(corpus.dictionary))
        print('Collocations: %s' % corpus.matrix.nnz)
        
        _glove = Glove(no_components=100, learning_rate=0.05)
        _glove.fit(corpus.matrix, epochs=10, no_threads=4, verbose=True)
        _glove.add_dictionary(corpus.dictionary)
        _glove.save(glove_model)
    
    else:
        print('Loading pre-trained GloVe model')
        _glove = Glove.load(glove_model)
    
    return _glove

corpus = make_corpus(load=False)
glove = train(corpus, train=True)

# https://projector.tensorflow.org/ 에서 보기 위해 파일 생성
np.savetxt('./glove-vector.tsv', glove.word_vectors, delimiter='\t')
with open('./glove-metadata.tsv', 'w', encoding='utf-8') as f:
    for key in glove.dictionary.keys():
        f.write(f"{key}\n")

