# Slot Intent Extraction
## About
Slot-Intent extraction is a task in text classification.
## Requirements
```
  pip install transformers
  pip install tensorflow-addons
  pip install -q tf-models-official==2.7.0
```
## How to use
* **Training model run train.py**
```
python /content/drive/MyDrive/bkai/modules/train.py \
            --path_train_seq_in ./training_data/seq.in \
            --path_train_seq_out ./training_data/seq.out \
            --path_label ./training_data/training_data/label \
            --path_slot_label ./training_data/slot_label.txt \
            --tokenizer_dir_config vinai/phobert-base \
            --batch_size 32 \
            --epochs 20 \
            --path_ckpt ./bkai/ckpt/bk.h5 \
            --path_log ./bkai/logs/log \
```
* **Testing model run test.py**
```
 python /content/drive/MyDrive/bkai/modules/train.py \
            --path_test_seq_in ./dev_data/seq.in \
            --path_test_seq_out ./training_data/seq.out \
            --path_label ./training_data/training_data/label \
            --path_slot_label ./training_data/slot_label.txt \
            --path_ckpt ./bkai/ckpt/bk.h5 \
```
## Dataset
  In this project we use two datasets: **PhoATIS** and **BkAI** for Vietnamese
  ```
data/
├── PhoATIS
│   ├── word-level
|       ├── train
|           ├── label
|           ├── seq.in
|           ├── seq.out
|       ├── test [ ... ]
|       ├── dev [ ...]
|       ├── intent_label.txt
|       ├── slot_label.txt
│   ├── syllable-level [...]
│     
├── bkAI
│   ├── training_data
|       ├── label
|       ├── seq.in
|       ├── seq.out
|       ├── intent_label.txt
|       ├── slot_label.txt
│   ├── dev_data [...]
│   ├── public_test_data [...]
  ```
## Evaluation Metric
  * **Average precision**: F1 score for intent classification and slot filling tasks
  * **Sentence accuracy**: We also compute an exact match (EM) / Sentence accuracy (For each sentence, if the intent and slot of the predicted exactly match the intent and slot of the label, EM = 1, otherwise EM = 0)
    

