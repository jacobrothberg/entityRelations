
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.callbacks import History
from keras.layers import concatenate
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Embedding, Input, BatchNormalization, Dense, Bidirectional, LSTM, Dropout, Activation
from keras.backend import sigmoid

def swish(x):
    return (sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})


def train_model(x_train, ancestor_train, dep_train, y_train,
                 x_val, ancestor_val, dep_val, y_val,
                 embedding_matrix, max_length, max_features):
    embedding_size = 300
    batch_size = 64
    epochs = 50
    embedding_layer = Embedding(max_features, output_dim=embedding_size,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm0 = Bidirectional(LSTM(256, activation="tanh", dropout=0.2, return_sequences=True,
                               kernel_initializer='he_uniform'))(embedded_sequences)
    lstm1 = Bidirectional(LSTM(128, activation="tanh", dropout=0.2, return_sequences=True,
                               kernel_initializer='he_uniform'))(lstm0)
    lstm2 = Bidirectional(LSTM(64, activation="tanh", dropout=0.2, return_sequences=False,
                               kernel_initializer='he_uniform'))(lstm1)
    bn = BatchNormalization()(lstm2)


    ancestor_input = Input(shape=(2,))
    ancestor_feature = Dense(64, activation=swish)(ancestor_input)

    dep_input = Input(shape=(34,))
    dep_feature = Dense(128, activation=swish)(dep_input)

    combine_feature = concatenate([bn, ancestor_feature, dep_feature])
    dense1 = Dense(64, activation=swish)(combine_feature)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(32, activation=swish)(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    predictions = Dense(19, activation='softmax')(dropout2)
    model = Model([sequence_input, ancestor_input, dep_input], predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    filepath = "models/LSTM.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True)
    history = History()
    callbacks_list = [checkpoint, history]

    history = model.fit([x_train, ancestor_train, dep_train], y_train,
                        validation_data=([x_val, ancestor_val, dep_val], y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks_list)
    return model, history