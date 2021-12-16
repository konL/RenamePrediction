import datasets as ds
import keras
#准备数值的数据
from keras import Input

print("[INFO] loading numeric attributes...")
inputPath = "C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\numericData\\jmeter_G1.csv"
df = ds.load_attributes(inputPath)
#获取credit_cards表中列的名称
columns=df.columns
#删除最后一列，即class列
features_columns=columns.delete(len(columns)-1)
#获取除class列以外的所有特征列
features=df[features_columns]
labels=df['20']
features = ds.process_attributes(df,features)
print(features)

print("[INFO] loading test_numeric attributes...")
inputPath = "C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\numericData\\jmeter_G1_test.csv"
test_df = ds.load_attributes(inputPath)
#获取credit_cards表中列的名称
test_columns=test_df.columns
#删除最后一列，即class列
test_features_columns=test_columns.delete(len(test_columns)-1)
#获取除class列以外的所有特征列
test_features=test_df[test_features_columns]
test_labels=test_df['20']
test_features = ds.process_attributes(test_df,test_features)



#传入数据
from MODEL import textCNN_MLP as m
mlp = m.create_mlp(features.shape[1], regress=False)
config_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_config.json'
checkpoint_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_model.ckpt'

dict_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\vocab.txt'
cnn = m.build_bert_model(config_path,checkpoint_path,2)

from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

import tensorflow as tf


model = Model(inputs=[mlp.input, cnn.input], outputs=x)
#
# # 让我们继续编译、培训和评估我们新形成的模型:
#
# # compile the model using mean absolute percentage error as our loss,
# # implying that we seek to minimize the absolute percentage difference
# # between our price *predictions* and the *actual prices*
#
# model.compile( loss='sparse_categorical_crossentropy',
#         optimizer=Adam(5e-6),
#         metrics=['accuracy'])
#
# # train the model
# print("[INFO] training model...")
# # 加载数据集
# import keras4bert_loaddata as ld
# train_data, train = ld.load_data(
#     'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\train_data_6x\\6x\\jmeter_train03.csv'
#     )
#
# test_data, test = ld.load_data(
#     'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\test_data_6x\\jmeter_test03.csv')
#
# # test_data, test = load_data('bi_train_method.csv')
# print(train['label_class'].value_counts())
#
# #
# emb_columns = train.columns
#
# # 删除最后一列，即class列
# emb_features= train[emb_columns.delete(len(emb_columns) - 1)]
#
# # 获取class列
# emb_labels = train['label_class']
#
# # #划分原始数据训练集和测试集用于oversample模型生成
#
# # RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
# from imblearn.under_sampling import RandomUnderSampler
#
# rus = RandomUnderSampler(random_state=0)
# os_features, os_labels = rus.fit_resample(features, labels)
# # 新生成的数据集
# import pandas as pd
# train = pd.concat([os_features, os_labels], axis=1)
# print(train['label_class'].value_counts())
# train_data = train.values
# batch_size=16
# # 转换数据集
# train_generator = ld.data_generator(train_data, batch_size)
# # val_generator = data_generator(val_data, batch_size)
# test_generator = ld.data_generator(test_data, batch_size)
# earlystop = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=2,
#     verbose=1,
#     mode='min'
# )
#
# bast_model_filepath = 'v3_best_model.weights'
# checkpoint = keras.callbacks.ModelCheckpoint(
#     bast_model_filepath,
#     monitor='val_loss',
#     verbose=2,
#     save_best_only=True,
#     mode='min'
# )
# model.fit_generator(
#     [features, train_generator.forfit()], labels,
#     validation_data=([test_features, test_generator.forfit()], test_labels),
#     steps_per_epoch=len(test_generator),
#     epochs=20,
#     shuffle=True,
#     verbose=1,
#
#     callbacks=[earlystop, checkpoint]
# )
