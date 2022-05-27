import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from CRF_layer import CRF
from utils import *
from AttentionContext import AttentionWithContext


def build_model(slots_num, intent_num, max_length, bert):
    crf = CRF(slots_num, name='slots')
    input_word_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    input_type_ids = Input(shape=(max_length,), dtype=tf.int32, name='token_type_ids')

    # ====================== BERT PRETRAIN =======================
    sequence_output = bert(input_word_ids, input_type_ids)[0]
    x = Bidirectional(LSTM(256, return_sequences=True, unit_forget_bias=True,
                           kernel_regularizer=regularizers.L2(1e-4)))(sequence_output)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=True, unit_forget_bias=True,
                           kernel_regularizer=regularizers.L2(1e-4)))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, return_sequences=True, unit_forget_bias=True,
                           kernel_regularizer=regularizers.L2(1e-4)))(x)
    x = Dropout(0.2)(x)
    time_layer = TimeDistributed(Dense(256, activation='relu'), name='time')(x)
    slots_output = crf(time_layer)

    # Branch Intent
    x1 = Bidirectional(LSTM(256, return_sequences=True, unit_forget_bias=True, dropout=0.4,
                            kernel_regularizer=regularizers.L2(1e-4)))(sequence_output)
    x1 = Bidirectional(LSTM(256, return_sequences=True, unit_forget_bias=True, dropout=0.4,
                            kernel_regularizer=regularizers.L2(1e-4)))(x1)
    x1 = AttentionWithContext()(x1)
    intents_output = Dense(intent_num, activation='linear', name='intents')(x1)

    model = Model(inputs=[input_word_ids, input_type_ids], outputs=[slots_output, intents_output])

    return model, crf