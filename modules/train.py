import argparse
import sys

from tensorflow.keras.callbacks import *
from models import build_model
from data_helper import TextFromFile, DataLoader
from utils import optimizer, softmax_loss
from loads_pretrain import convert_tfmodel

def call_backs(path_ckpt, path_log):
    cb = []
    checkpoint = ModelCheckpoint(path_ckpt, monitor='val_loss', mode='auto',
                                 save_best_only=True, verbose=1)
    cb.append(checkpoint)
    tb = TensorBoard(log_dir=path_log, histogram_freq=0)
    cb.append(tb)
    er = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=5, verbose=0, mode="auto")
    cb.append(er)
    return cb

def training(args):
    # get Train data
    df_train, slot_lb = TextFromFile(args.path_train_seq_in, args.path_train_seq_out,
                                     args.path_label, args.path_slot_label)
    data = DataLoader(df_train['text'], df_train['slot_text'], df_train['intent'], slot_lb,
                      max_length=args.max_len, tokenizer_name=args.tokenizer_dir_config)
    train_input_bert, train_slot_sentence, train_lb_intent, mapping_intent, mapping_slot = data.create_tensor()

    # Build model
    bert = convert_tfmodel(args.tokenizer_dir_config, args.tokenizer_dir_config)
    AdamW = optimizer(args.lr, args.epochs, args.batch_size, len(train_lb_intent))
    model, crf = build_model(len(mapping_slot), len(mapping_intent), args.max_len, bert)
    model.summary()


    model.compile(optimizer= AdamW, loss={'slots': crf.get_loss, 'intents': softmax_loss},
                  loss_weights={'slots': 3.0, 'intents': 1.0},
                  metrics={'slots': [crf.get_accuracy], 'intents': ['accuracy']},
                  run_eagerly=True)

    history = model.fit(x=train_input_bert, y=(train_slot_sentence, train_lb_intent),
                        validation_split=0.2,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        verbose=1,
                        shuffle=True,
                        callbacks=call_backs(args.path_ckpt, args.path_log))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_train_seq_in', type=str, default='./ATIS_VI/train/seq.in',
                        help='train_seq.in')
    parser.add_argument('--path_train_seq_out', type=str, default=r'./ATIS_VI/train/seq.out',
                        help='train_seq.out')
    parser.add_argument('--path_label', type=str, default='./ATIS_VI/train/label',
                        help='label')
    parser.add_argument('--path_slot_label', type=str, default='./ATIS_VI/train/slot_label.txt',
                        help='slot_label.txt')
    parser.add_argument('--tokenizer_dir_config', type=str, default='vinai/phobert-base',
                        help='Pretrain with model Transformer')
    parser.add_argument('--l2', type=float, default=0.0001,
                        help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout layer')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='epochs')
    parser.add_argument('--lr', type=int, default=0.004,
                        help='learning_rate')
    parser.add_argument('--max_len', type=int, default=50,
                        help='max_length')
    parser.add_argument('--path_ckpt', type=str, default='./BKAI/ckpt/bkai.h5',
                        help='path_save_ckpt')
    parser.add_argument('--path_log', type=str, default='./BKAI/logs/log',
                        help='path_save_tensorboard')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    # Start training
    training(args)