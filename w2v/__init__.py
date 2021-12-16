from gensim.models import Word2Vec
import re
import pandas as pd
import os

base="C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\test_data_6x\\no_order\\"
sentences = []
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for root, ds, fs in os.walk(base):
    for f in fs:
        print(f)
        df = pd.read_csv(base+f, header=0)
        documents=[]
        print(df.head())
        for indexs in df.index:
            documents .append(df.loc[indexs].values[7])
            documents.append(df.loc[indexs].values[8])
            print(df.loc[indexs].values[7])

        #document包含了一个项目的所有statenment
        for doc in documents:
            doc = re.sub(stop, ' ', doc)
            print(doc)
            sentences.append(doc.split())





# size嵌入的维度，window窗口大小，workers训练线程数
# 忽略单词出现频率小于min_count的单词
# sg=1使用Skip-Gram，否则使用CBOW
model = Word2Vec(sentences, size=5, window=1, min_count=1, workers=4, sg=1)
model.save("C:\\Users\\delll\\PycharmProjects\\RenamePrediction\\w2v\\w2v.model")


