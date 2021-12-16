from gensim.models import Word2Vec

model = Word2Vec.load("C:\\Users\\delll\\PycharmProjects\\RenamePrediction\\w2v\\w2v.model")
print(model.wv['cat'])