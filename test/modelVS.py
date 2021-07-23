# from glove import Corpus, Glove
import glove

new_model = glove.load('./glove/models/origin.model')
new_model_result = new_model.most_similar('눈꽃')
print(new_model_result)