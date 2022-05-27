import argparse
import sys

from models import *
from data_helper import TextFromFile, DataLoader
from utils import *
from loads_pretrain import convert_tfmodel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_test_seq_in', type=str, default='./ATIS_VI/dev/seq.in',
                        help='test_seq.in')
    parser.add_argument('--path_test_seq_out', type=str, default='./ATIS_VI/dev/seq.out',
                        help='test_seq.out')
    parser.add_argument('--path_label', type=str, default='./ATIS_VI/dev/label',
                        help='label')
    parser.add_argument('--path_slot_label', type=str, default='./ATIS_VI/train/slot_label.txt',
                        help='slot_label.txt')
    parser.add_argument('--tokenizer_dir_config', type=str, default='vinai/phobert-base',
                        help='Pretrain with model Transformer')
    parser.add_argument('--max_len', type=int, default=50,
                        help='max_length')
    parser.add_argument('--path_ckpt', type=str, default='./ckpt/bkai.h5',
                        help='path_save_ckpt')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    df_train, slot_lb = TextFromFile(args.path_test_seq_in, args.path_test_seq_out,
                                     args.path_label, args.path_slot_label)

    data = DataLoader(df_train['text'], df_train['slot_text'], df_train['intent'], slot_lb,
                      max_length=args.max_len, tokenizer_name='vinai/phobert-base')

    dev_input_bert, dev_slot_sentence, dev_lb_intent, mapping_intent, mapping_slot = data.create_tensor()

    # Build model
    bert = convert_tfmodel(args.tokenizer_dir_config, args.tokenizer_dir_config)
    model, _ = build_model(len(mapping_slot), len(mapping_intent), args.max_len, bert)
    model.summary()
    model.load_weights(args.path_ckpt)

    # test model
    slot, intent = model.predict(dev_input_bert)
    sen_acc = sentence_acc((dev_slot_sentence, dev_lb_intent), (slot, intent))
    print('\n\n========================================================')
    print('Sentence accuracy: ', sen_acc)

