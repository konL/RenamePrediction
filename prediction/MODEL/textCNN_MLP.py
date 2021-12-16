# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model


def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))

    # return our model
    return model


from bert4keras.backend import keras,set_gelu
from bert4keras.models import build_transformer_model, Lambda
from bert4keras.optimizers import Adam

set_gelu('tanh')

def textcnn(inputs,kernel_initializer):
	# 3,4,5
	cnn1 = keras.layers.Conv1D(
			256,
			3,
			strides=1,
			padding='same',
			activation='relu',
			kernel_initializer=kernel_initializer
		)(inputs) # shape=[batch_size,maxlen-2,256]
	cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]

	cnn2 = keras.layers.Conv1D(
			256,
			4,
			strides=1,
			padding='same',
			activation='relu',
			kernel_initializer=kernel_initializer
		)(inputs)
	cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

	cnn3 = keras.layers.Conv1D(
			256,
			5,
			strides=1,
			padding='same',
			kernel_initializer=kernel_initializer
		)(inputs)
	cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

	output = keras.layers.concatenate(
		[cnn1,cnn2,cnn3],
		axis=-1)
	output = keras.layers.Dropout(0.2)(output)
	return output

def build_bert_model(config_path,checkpoint_path,class_nums):
	bert = build_transformer_model(
		config_path=config_path,
		checkpoint_path=checkpoint_path,
		model='bert',
		return_keras_model=False)

	cls_features = keras.layers.Lambda(
		lambda x: x[:, 0],
		name='cls-token'
	)(bert.model.output)  # shape=[batch_size,768]
	all_token_embedding = keras.layers.Lambda(
		lambda x: x[:, 1:-1],
		name='all-token'
	)(bert.model.output)  # shape=[batch_size,maxlen-2,768]

	cnn_features = textcnn(
		all_token_embedding, bert.initializer)  # shape=[batch_size,cnn_output_dim]
	concat_features = keras.layers.concatenate(
		[cls_features, cnn_features],
		axis=-1)

	dense = keras.layers.Dense(
		units=512,
		activation='relu',
		kernel_initializer=bert.initializer
	)(concat_features)
	# cls_features=Lambda(lambda x:keras.backend.expand_dims(x,axis=-1))(cls_features)
	# cnn_features = textcnn(
	# 	cls_features, bert.initializer)  # shape=[batch_size,cnn_output_dim]


	# dense = keras.layers.Dense(
	# 	units=512,
	# 	activation='relu',
	# 	kernel_initializer=bert.initializer
	# )(cls_features)

	output = keras.layers.Dense(
		units=class_nums,
		activation='softmax',
		kernel_initializer=bert.initializer
	)(dense)
    # check to see if the regression node should be added


    # check to see if the regression node should be added

    # regress=False
    # if :
    # output = Dense(1, activation="linear")(output)

	model = keras.models.Model(bert.model.input, output)
	print(model.summary())
	from tensorflow.keras.utils import plot_model
	plot_model(model, './model.png', show_shapes=True)

	return model
