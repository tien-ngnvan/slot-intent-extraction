import tensorflow as tf
import numpy as np
import pandas as pd

from transformers import AutoTokenizer


class DataLoader():
    def __init__(self, all_text, all_slot_text, lb_intent, lb_slot, max_length: int, tokenizer_name: str):
        super(DataLoader, self).__init__()
        self.all_text = all_text
        self.all_slot_text = all_slot_text
        self.lb_intent = lb_intent
        self.lb_slot = lb_slot
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True, use_fast=True)

    def word_ids(self, tokens, input_ids, tokenizer):
        ids = []
        for i, token in enumerate(tokens):
            ids.extend([i] * len(tokenizer.tokenize(token)))
        eos_ids = np.where(input_ids == 2)[0][0]
        ids = ids[: eos_ids - 1]
        ids = [None] + ids + [None] * (len(input_ids) - len(ids) - 1)
        return ids

    def create_tensor(self):
        X, Y, Z = [], [], []  # X, Y for bert; Z: intent label
        # Tokens word
        for index, text in enumerate(self.all_text):
            # sentence
            tokens = self.tokenizer.encode_plus(self.all_text[index], max_length=self.max_length,
                                                truncation=True, padding='max_length', return_tensors='tf')
            X.append(tokens['input_ids'][0])
            Y.append(tokens['attention_mask'][0])

            # Intent label
            facts = np.unique(np.unique(self.lb_intent), return_index=True)
            mp_intent = dict(zip(*facts))
            intent = [mp_intent[i] for i in self.lb_intent]

            # Slot label
            mp_slot = dict((i, j) for j, i in enumerate(self.lb_slot))

            # Slot sentence label
            e_word_ids = self.word_ids(self.all_text[index], tokens['input_ids'][0], self.tokenizer)
            temp = ['PAD' if i is None else self.all_slot_text[index][i] for i in e_word_ids]
            Z.append([mp_slot[i] for i in temp])

        # sl = []
        # for text in Z:
        #     sl.append([mp_slot.get(i) if mp_slot.get(i) != 'PAD' else 'PAD' for i in text])
        input_bert = (tf.convert_to_tensor(X), tf.convert_to_tensor(Y))
        return input_bert, tf.convert_to_tensor(Z), tf.convert_to_tensor(intent), mp_intent, mp_slot


def TextFromFile(seq_in_file, seq_out_file, intent_file, slot_label_file):
    # read data
    with open(seq_in_file, encoding='utf-8') as f:
        seq_in = f.readlines()
    seq_in = [i.rstrip('\n') for i in seq_in]

    with open(seq_out_file, encoding='utf-8') as f:
        seq_out = f.readlines()
    seq_out = [i.rstrip('\n') for i in seq_out]
    # Intent
    with open(intent_file) as f:
        intent = f.readlines()
    intent = [i.rstrip('\n') for i in intent]
    new_intent = [i[0:i.find('#')] if i.find('#') != -1 else i for i in intent]
    # slot
    with open(slot_label_file) as f:
        slot_label = f.readlines()
    slot_label = [i.rstrip('\n') for i in slot_label]

    # preprocess
    df = pd.DataFrame()
    df['text'] = [i.split(' ') for i in seq_in]
    df['slot_text'] = [i.split(' ') for i in seq_out]
    df['intent'] = new_intent
    df = df.sample(frac=1).reset_index(drop=True)

    return df, slot_label
