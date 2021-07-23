from glove import Corpus, Glove
import MeCab
import os
from nltk import word_tokenize


path = './step2/'
file_name = 'processed_korquad.txt'
result = []
with open(os.path.join(path, file_name), 'r', encoding='utf-8') as input:
    i = 0
    for input_line in input:

        # 진행률 확인
        i += 1
        if i % 1000 == 0:
            print("{}] {} finished".format(file_name, i))
        
        result.append(word_tokenize(input_line))
        # result.append(input_line.split(" "))

# 훈련 데이터로부터 Glove에서 사용할 행렬 생성
print(len(result))
print(result[0])
corpus = Corpus()
print("fit start\n")
corpus.fit(result, window=5)

print("fit end\n")

glove = Glove(no_components=100, learning_rate=0.05)
# 학습에 이용할 쓰레드의 개수는 4로 설정, 에폭은 20
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)

glove.add_dictionary(corpus.dictionary)
glove.save("glove.model")
