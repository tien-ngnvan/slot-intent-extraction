import numpy as np
import tensorflow as tf
import official.nlp.optimization

from official import nlp
from sklearn.metrics import f1_score

# ======================= METRIC =======================
def f1_slot(y_true, y_pred):
    y_pred = y_pred.numpy().flatten()
    y_true = y_true.numpy().flatten()
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def f1_intent(y_true, y_pred):
    y_pred = np.argmax(y_pred.numpy(), axis=1)
    value = y_pred.flatten()
    y_true = y_true.numpy().flatten()
    f1 = f1_score(y_true, value, average='macro')
    return f1

def sentence_acc(y_true, y_pred):
    # slot
    y_slot_true = y_true[0].numpy()
    y_slot_pred = y_pred[0]
    # intent
    y_intent_true = y_true[1].numpy().flatten()
    y_intent_pred = np.argmax(y_pred[1], axis=1).flatten()
    acc_intent = y_intent_true == y_intent_pred

    index = 0
    for i, j in zip(y_slot_true, y_slot_pred):
        if len(i) != len(j):
            raise ValueError('error!')

        for t in range(len(i)):
            if i[t] != j[t]:
                acc_intent[index] = False
                break
        index += 1
    acc_intent = acc_intent.astype(float)

    return np.mean(acc_intent) * 100

# ======================= LOSS =======================
def softmax_loss(y_true, y_pred):
    # y_true: sparse target
    # y_pred: logist
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)
    return tf.reduce_mean(ce)

# ======================= OPTIMIZER =======================
def optimizer(learning_rate, epochs, batch_size, len_data, warm_up=0.1):
    train_data_size = len_data
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = num_train_steps * warm_up

    decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=num_train_steps,
        end_learning_rate=0)

    warmup_schedule = nlp.optimization.WarmUp(
        initial_learning_rate=learning_rate,
        decay_schedule_fn=decay_schedule,
        warmup_steps=warmup_steps)

    optimizer = nlp.optimization.AdamWeightDecay(
        learning_rate=warmup_schedule,
        weight_decay_rate=0.01,
        epsilon=1e-8,
        beta_2=0.98,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])

    return optimizer