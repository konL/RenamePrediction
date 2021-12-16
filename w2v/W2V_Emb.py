import pandas as pd
import re
from gensim.models import Word2Vec
from imblearn import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
proj='cassandra'
issmall="_small"
"""
 读取训练集并构造训练样本
"""
def split_sentence(sentence):
    stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sentence = re.sub(stop, ' ', sentence)
    return sentence.split()


"""
 使用训练好的Word2Vec
"""
# 嵌入的维度
embedding_vector_size = 10
w2v_model = Word2Vec.load("C:\\Users\\delll\\PycharmProjects\\RenamePrediction\\w2v\\w2v.model")
# 取得所有单词
vocab_list = list(w2v_model.wv.vocab.keys())
# 每个词语对应的索引
word_index = {word: index for index, word in enumerate(vocab_list)}
# 序列化
def get_index(sentence):
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence

#得把数据的token变成数字序列~train_sentence

base="C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\train_data_6x\\no_order\\"+proj+"_train.csv"
train_sentences = []
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
df = pd.read_csv(base, header=0)
print(df['label_class'].value_counts())
documents=[]
for indexs in df.index:
    documents .append(df.loc[indexs].values[8]+"[SEP]"+df.loc[indexs].values[9])


for doc in documents:
    doc = re.sub(stop, ' ', doc)
    train_sentences.append(doc.split())
X_train = list(map(get_index, train_sentences))

# 截长补短
maxlen = 300
X_train = pad_sequences(X_train, maxlen=maxlen)
# 取得标签
Y_train = df['label_class'].values
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_train, Y_train = rus.fit_resample(X_train, Y_train)




test_base="C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\test_data_6x\\no_order\\"+proj+"_test_mask_change"+issmall+".csv"
test_sentences = []
test_df = pd.read_csv(test_base, header=0)
test_documents=[]
for indexs in test_df.index:
    test_documents .append(test_df.loc[indexs].values[7]+"[SEP]"+test_df.loc[indexs].values[8])


for doc in test_documents:
    doc = re.sub(stop, ' ', doc)
    test_sentences.append(doc.split())
X_test = list(map(get_index, test_sentences))

# 截长补短
X_test = pad_sequences(X_test, maxlen=maxlen)
# 取得标签
Y_test = test_df['label_class'].values

"""
 构建分类模型
"""
# 让 Keras 的 Embedding 层使用训练好的Word2Vec权重
embedding_matrix = w2v_model.wv.vectors

model = Sequential()
model.add(Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    input_length=maxlen,
    weights=[embedding_matrix],
    trainable=False))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))

from tensorflow.keras import backend as K
def recall(y_true,y_pred):

    true_positive = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    possible_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall=true_positive/(possible_positive+K.epsilon())
    return recall
def precision(y_true,y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision=true_positive/(predicted_positive+K.epsilon())
    return precision
def f1(y_true,y_pred):
    def recall(y_true,y_pred):

        true_positive = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        possible_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall=true_positive/(possible_positive+K.epsilon())
        return recall
    def precision(y_true,y_pred):
        true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision=true_positive/(predicted_positive+K.epsilon())
        return precision
    precision=precision(y_true,y_pred)
    recall=recall(y_true,y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from bert4keras.optimizers import Adam
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(5e-6),
    metrics=['accuracy', f1, precision, recall])
import keras
earlystop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    verbose=1,
    mode='min'
)
best_model_filepath = 'C:\\Users\\delll\\PycharmProjects\\RenamePrediction\\w2v\\w2v_best_model_'+proj+'.weights'
checkpoint = keras.callbacks.ModelCheckpoint(
    best_model_filepath,
    monitor='val_loss',
    verbose=2,
    save_best_only=True,
    mode='min'
)
history = model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_test, Y_test),
    batch_size=4,
    epochs=10,  callbacks=[earlystop,checkpoint])


model.load_weights('C:\\Users\\delll\\PycharmProjects\\RenamePrediction\\w2v\\w2v_best_model_'+proj+'.weights')
test_pred = []
test_true = []
test_pred = model.predict(X_test).argmax(axis=1)
test_true = Y_test
print(test_pred)
print(test_true)
from sklearn.metrics import classification_report, f1_score

print("项目：",proj)
fp = 0
tp = 0
fn = 0
tn = 0
index_i = 0
for i in range(len(test_true)):
    pred = int(test_pred[i])
    if ((test_true[i] == pred) & (test_true[i] == 0)):
        tn = tn + 1
    if ((test_true[i] == pred) & (test_true[i] == 1)):
        tp = tp + 1
    if ((test_true[i] != pred) & (test_true[i] == 0)):
        fp = fp + 1
    if ((test_true[i] != pred) & (test_true[i] == 1)):

        fn = fn + 1

print(tp, fp, tn, fn)

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("f1:", 2 * ((precision * recall) / (precision + recall)))
print("precision:", precision)
print("recall:", recall)
    # target_names = [line.strip() for line in open('label','r',encoding='utf8')]
    # print(classification_report(test_true, test_pred,target_names=target_names))
print(classification_report(test_true, test_pred))

changeNum=test_df['changeNum'].tolist()
print(len(changeNum),len(test_pred),len(test_true))
fp=0
tp=0
fn=0
tn=0
index_i=0
for i in range(len(test_true)):
    pred = int(test_pred[i])

    if((test_pred[i]==1)&(changeNum[i]==0)):

        test_pred[i]=0
        pred=0

    if((test_true[i]==pred) & (test_true[i]==0)):
        tn=tn+1
    if ((test_true[i] == pred) & (test_true[i]==1)):
        tp=tp+1
    if ((test_true[i] != pred) & (test_true[i] == 0)):
        fp=fp+1
    if ((test_true[i] != pred) & (test_true[i] == 1)):

        fn=fn+1

print(tp,fp,tn,fn)

precision=tp/(tp+fp)
recall=tp / (tp + fn)

print("f1:", 2*((precision*recall)/(precision+recall)))
print("precision:", precision)
print("recall:", recall)
