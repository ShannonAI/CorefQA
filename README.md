# CorefQA: Coreference Resolution as Query-based Span Prediction

The repository contains the code of the recent research advances in [Shannon.AI](http://www.shannonai.com). Please post github issues or email xiaoya_li@shannonai.com for relevant questions.



**CorefQA: Coreference Resolution as Query-based Span Prediction** <br>
Wei Wu, Fei Wang, Arianna Yuan, Fei Wu and Jiwei Li<br>
In ACL 2020. [paper](https://arxiv.org/abs/1911.01746)<br>
If you find this repo helpful, please cite the following:
```latex
@article{wu2019coreference,
  title={Coreference Resolution as Query-based Span Prediction},
  author={Wu, Wei and Wang, Fei and Yuan, Arianna and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1911.01746},
  year={2019}
}
```


## Contents 
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Install Package Dependencies](#install-package-dependencies)
- [Data Preprocess](#data-preprocess)
- [Download Pretrained MLM](#download-pretrained-mlm)
- [Training](#training)
    - [Finetune the SpanBERT Model on the Combination of Squad and Quoref Datasets](#finetune-the-spanbert-model-on-the-combination-of-squad-and-quoref-datasets)
    - [Train the CorefQA Model on the CoNLL-2012 Coreference Resolution Task](#train-the-corefqa-model-on-the-conll-2012-coreference-resolution-task)
- [Evaluation and Prediction](#evaluation-and-prediction)
- [Download the Final CorefQA Model](#download-the-final-corefqa-model)
- [Descriptions of Directories](#descriptions-of-directories)
- [Acknowledgement](#acknowledgement)
- [Useful Materials](#useful-materials)
- [Contact](#contact)


## Overview 
The model introduces +3.5 (83.1) F1 performance boost over previous SOTA coreference models on the CoNLL benchmark. The current codebase is written in Tensorflow. We plan to release the PyTorch version soon.  The current code version only supports training on TPUs and testing on GPUs (due to the annoying features of TF and TPUs). You thus have to bear the trouble of transferring all saved checkpoints from TPUs to GPUs for evaluation (we will fix this soon). Please follow the parameter setting in the log directionary to reproduce the performance.  


| Model          | F1 (%) |
| -------------- |:------:|
| Previous SOTA  (Joshi et al., 2019a)  | 79.6  |
| CorefQA + SpanBERT-large | 83.1   |


## Hardware Requirements
TPU for training: Cloud TPU v3-8 device (128G memory) with Tensorflow 1.15 Python 3.5 

GPU for evaluation: with CUDA 10.0 Tensorflow 1.15 Python 3.5

## Install Package Dependencies
 
```shell
$ python3 -m pip install --user virtualenv
$ virtualenv --python=python3.5 ~/corefqa_venv
$ source ~/corefqa_venv/bin/activate
$ cd coref-tf
$ pip install -r requirements.txt
# If you are using TPU, please run the following commands:
$ pip install cloud-tpu-client
$ pip install google-cloud-storage
```

## Data Preprocess 

1) Download the offical released [Ontonotes 5.0 (LDC2013T19)](https://catalog.ldc.upenn.edu/LDC2013T19). <br> 
2) Preprocess Ontonotes5 annotations files for the CoNLL-2012 coreference resolution task. <br> 
Run the command with **Python 2**
`bash ./scripts/data/preprocess_ontonotes_annfiles.sh  <path_to_LDC2013T19-ontonotes5.0_directory>  <path_to_save_CoNLL12_coreference_resolution_directory> <language>`<br> 
and it will create `{train/dev/test}.{language}.v4_gold_conll` files in the directory `<path_to_save_CoNLL12_coreference_resolution_directory>`. <br> 
`<language>` can be `english`, `arabic` or `chinese`. In this paper, we set `<language>` to `english`. <br>
If you want to use **Python 3**, please refer to the
[guideline](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/train/training.md#get-the-data) <br> 
3) Generate TFRecord files for experiments. <br> 
Run the command with **Python 3** `bash ./scripts/data/generate_tfrecord_dataset.sh <path_to_save_CoNLL12_coreference_resolution_directory>  <path_to_save_tfrecord_directory> <path_to_vocab_file>`
and it will create `{train/dev/test}.overlap.corefqa.{language}.tfrecord` files in the directory `<path_to_save_CoNLL12_coreference_resolution_directory>`. <br> 

## Download Pretrained MLM
In our experiments, we used pretrained mask language models to initialize the mention_proposal and corefqa models. 

1) Download the pretrained models. <br> 
Run `bash ./scripts/data/download_pretrained_mlm.sh <path_to_save_pretrained_mlm> <model_sign>` to download and unzip the pretrained mlm models. <br> 
`<model_sign>` shoule take the value of `[bert_base, bert_large, spanbert_base, spanbert_large, bert_tiny]`.

- `bert_base, bert_large, spanbert_base, spanbert_large` are trained with a cased(upppercase and lowercase tokens) vocabulary. Should use the cased train/dev/test coreference datasets. 
- `bert_tiny` is trained with a uncased(lowercase tokens) vocabulary. We use the tinyBERT model for fast debugging. Should use the uncased train/dev/test coreference datasets. <br> 

2) Transform SpanBERT from `Pytorch` to `Tensorflow`. <br> 

After downloading `bert_<scale>` to `<path_to_bert_<scale>_tf_dir>` and `spanbert_<scale>` to `<path_to_spanbert_<scale>_pytorch_dir>`, you can start transforming the SpanBERT model to Tensorflow and the model is saved to the directory `<path_to_save_spanbert_tf_checkpoint_dir>`. `<scale>` should take the value of `[base, large]`. <br> 

We need to tranform the SpanBERT checkpoints from Pytorch to TF because the offical relased models were trained with Pytorch. 
Run `bash ./scripts/data/transform_ckpt_pytorch_to_tf.sh <model_name>  <path_to_spanbert_<scale>_pytorch_dir> <path_to_bert_<scale>_tf_dir>  <path_to_save_spanbert_tf_checkpoint_dir>` 
and the `<model_name>` in TF will be saved in `<path_to_save_spanbert_tf_checkpoint_dir>`.

- `<model_name>` should take the value of `[spanbert_base, spanbert_large]`. 
- `<scale>` indicates that the `bert_model.ckpt` in the `<path_to_bert_<scale>_tf_dir>` should have the same scale(base, large) to the `bert_model.bin` in `<path_to_spanbert_<scale>_pytorch_dir>`.


## Training 

Follow the pipeline described in the paper, you need to: <br> 
1) load a pretrained SpanBERT model. <br> 
2) finetune the SpanBERT model on the combination of Squad and Quoref datasets. <br> 
3) pretrain the mention proposal model on the coref dataset. <br>
4) jointly train the mention proposal model and the mention linking model. <br>
 
**Notice:** We provide the options of both pretraining these models yourself and loading the our pretrained models for 2) and 3). <br> 

### Finetune the SpanBERT Model on the Combination of Squad and Quoref Datasets
We finetune the SpanBERT model on the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) and [Quoref](https://allennlp.org/quoref) QA tasks for data augmentation before the coreference resolution task. 

1. You can directly download the pretrained model on the datasets. 
Download Data Augmentation Models on Squad and Quoref [link](https://www.dropbox.com/s/lqjc6kfe0w34jt0/finetune_spanbert_large_squad2.tar.gz?dl=0) <br>
Run `./scripts/data/download_squad2_finetune_model.sh <model-scale> <path-to-save-model>` to download finetuned SpanBERT on SQuAD2.0. <br>
The `<model-scale>` should take the value of `[base, large]`. <br>
The `<path-to-save-model>` is the path to save finetuned spanbert on SQuAD2.0 datasets. <br>


2. Or start to finetune the SpanBERT model on QA tasks yourself. 
- Download SQuAD 2.0 [train](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json) and [dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json) sets. 
- Download Quoref [train and dev](https://quoref-dataset.s3-us-west-2.amazonaws.com/train_and_dev/quoref-train-dev-v0.1.zip) sets.
- Finetune the SpanBERT model on Google Could V3-8 TPU. 

For Squad 2.0, Run the script in [./script/model/squad_tpu.sh]()
  ```bash 
  
   REPO_PATH=/home/shannon/coref-tf
   export TPU_NAME=tf-tpu
   export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
   SQUAD_DIR=gs://qa_tasks/squad2
   BERT_DIR=gs://pretrained_mlm_checkpoint/spanbert_large_tf
   OUTPUT_DIR=gs://corefqa_output_squad/spanbert_large_squad2_2e-5  

   python3 ${REPO_PATH}/run/run_squad.py \
   --vocab_file=$BERT_DIR/vocab.txt \
   --bert_config_file=$BERT_DIR/bert_config.json \
   --init_checkpoint=$BERT_DIR/bert_model.ckpt \
   --do_train=True \
   --train_file=$SQUAD_DIR/train-v2.0.json \
   --do_predict=True \
   --predict_file=$SQUAD_DIR/dev-v2.0.json \
   --train_batch_size=8 \
   --learning_rate=2e-5 \
   --num_train_epochs=4.0 \
   --max_seq_length=384 \
   --do_lower_case=False \
   --doc_stride=128 \
   --output_dir=${OUTPUT_DIR} \
   --use_tpu=True \
   --tpu_name=$TPU_NAME \
   --version_2_with_negative=True
  ```
After getting the best model (choose based on the performance on dev set) on `SQuAD2.0`, you should start finetuning the saved model on `Quoref`. <br>

Run the script in [./script/model/quoref_tpu.sh]() 
  ```bash 
  
   REPO_PATH=/home/shannon/coref-tf
   export TPU_NAME=tf-tpu
   export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
   QUOREF_DIR=gs://qa_tasks/quoref
   BERT_DIR=gs://corefqa_output_squad/panbert_large_squad2_2e-5
   OUTPUT_DIR=gs://corefqa_output_quoref/spanbert_large_squad2_best_quoref_3e-5 

   python3 ${REPO_PATH}/run_quoref.py \
   --vocab_file=$BERT_DIR/vocab.txt \
   --bert_config_file=$BERT_DIR/bert_config.json \
   --init_checkpoint=$BERT_DIR/best_bert_model.ckpt \
   --do_train=True \
   --train_file=$QUOREF_DIR/quoref-train-v0.1.json \
   --do_predict=True \
   --predict_file=$QUOREF_DIR/quoref-dev-v0.1.json \
   --train_batch_size=8 \
   --learning_rate=3e-5 \
   --num_train_epochs=5 \
   --max_seq_length=384 \
   --do_lower_case=False \
   --doc_stride=128 \
   --output_dir=${OUTPUT_DIR} \
   --use_tpu=True \
   --tpu_name=$TPU_NAME 
  ```
We use the best model (choose based on the performance on DEV set) on `Quoref` to initialize the CorefQA Model. 
  
### Train the CorefQA Model on the CoNLL-2012 Coreference Resolution Task 
1.1 Your can  you can download the pre-trained [mention proposal model](https://storage.googleapis.com/public_model_checkpoints/mention_proposal/model.ckpt-22000.data-00000-of-00001). 

1.2. Or train  the mention proposal model yourself. 

The script can be found in [./script/model/mention_tpu.sh]().

```bash 

REPO_PATH=/home/shannon/coref-tf
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
export TPU_NAME=tf-tpu
export TPU_ZONE=europe-west4-a
export GCP_PROJECT=xiaoyli-20-01-4820

BERT_DIR=gs://corefqa_output_quoref/spanbert_large_squad2_best_quoref_1e-5
DATA_DIR=gs://corefqa_data/final_overlap_384_6
OUTPUT_DIR=gs://corefqa_output_mention_proposal/squad_quoref_large_384_6_1e5_8_0.2

python3 ${REPO_PATH}/run/run_mention_proposal.py \
--output_dir=$OUTPUT_DIR \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$BERT_DIR/bert_model.ckpt \
--vocab_file=$BERT_DIR/vocab.txt \
--logfile_path=$OUTPUT_DIR/train.log \
--num_epochs=8 \
--keep_checkpoint_max=50 \
--save_checkpoints_steps=500 \
--train_file=$DATA_DIR/train.corefqa.english.tfrecord \
--dev_file=$DATA_DIR/dev.corefqa.english.tfrecord \
--test_file=$DATA_DIR/test.corefqa.english.tfrecord \
--do_train=True \
--do_eval=False \
--do_predict=False \
--learning_rate=1e-5 \
--dropout_rate=0.2 \
--mention_threshold=0.5 \
--hidden_size=1024 \
--num_docs=5604 \
--window_size=384 \
--num_window=6 \
--max_num_mention=60 \
--start_end_share=False \
--loss_start_ratio=0.3 \
--loss_end_ratio=0.3 \
--loss_span_ratio=0.3 \
--use_tpu=True \
--tpu_name=TPU_NAME \
--tpu_zone=TPU_ZONE \
--gcp_project=GCP_PROJECT \
--num_tpu_scores=1 \
--seed=2333
```

2. Jointly train the mention proposal model and linking model on CoNLL-12. <br> 

After getting the best mention proposal model on the dev set, start jointly training the mention proposal and linking tasks. 

Run and the script can be found in [./script/model/corefqa_tpu.sh]()

```bash

REPO_PATH=/home/shannon/coref-tf
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
export TPU_NAME=tf-tpu
export TPU_ZONE=europe-west4-a
export GCP_PROJECT=xiaoyli-20-01-4820

BERT_DIR=gs://corefqa_output_mention_proposal/output_bertlarge
DATA_DIR=gs://corefqa_data/final_overlap_384_6
OUTPUT_DIR=gs://corefqa_output_corefqa/squad_quoref_mention_large_384_6_8e4_8_0.2

python3 ${REPO_PATH}/run/run_corefqa.py \
--output_dir=$OUTPUT_DIR \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$BERT_DIR/best_bert_model.ckpt \
--vocab_file=$BERT_DIR/vocab.txt \
--logfile_path=$OUTPUT_DIR/train.log \
--num_epochs=8 \
--keep_checkpoint_max=50 \
--save_checkpoints_steps=500 \
--train_file=$DATA_DIR/train.corefqa.english.tfrecord \
--dev_file=$DATA_DIR/dev.corefqa.english.tfrecord \
--test_file=$DATA_DIR/test.corefqa.english.tfrecord \
--do_train=True \
--do_eval=False \
--do_predict=False \
--learning_rate=8e-4 \
--dropout_rate=0.2 \
--mention_threshold=0.5 \
--hidden_size=1024 \
--num_docs=5604 \
--window_size=384 \
--num_window=6 \
--max_num_mention=50 \
--start_end_share=False \
--max_span_width=10 \
--max_candiate_mentions=100 \
--top_span_ratio=0.2 \
--max_top_antecedents=30 \
--max_query_len=150 \
--max_context_len=150 \
--sec_qa_mention_score=False \
--use_tpu=True \
--tpu_name=TPU_NAME \
--tpu_zone=TPU_ZONE \
--gcp_project=GCP_PROJECT \
--num_tpu_scores=1 \
--seed=2333
```

## Evaluation and Prediction

Currently, the evaluation is conducted on a set of saved checkpoints after the training process, and DO NOT support evaluation during training. Please transfer all checkpoints (the output directory is set `--output_dir=<path_to_output_directory>` when running the `run_<model_sign>.py`) from TPUs to GPUs for evaluation. 
This can be achieved by downloading the output directory from the Google Cloud Storage. <br>  


The performance  on the test set is obtained by using  the model achieving the highest F1-score on the dev set. <br> 
Set `--do_eval=True`、 `--do_train=False` and `--do_predict=False` to `run_<model_sign>.py` and start the evaluation process on a set of saved checkpoints. And other parameters should be the same with the training process.
`<model_sign>` should take the value of `[mention_proposal, corefqa]`. <br>

The codebase also provides the option of evaluating a single model/checkpoint. Please set `--do_eval=False`、 `--do_train=False` and `--do_predict=True` to `run_<model_sign>.py` with the checkpoint path `--eval_checkpoint=<path_to_eval_checkpoint_model>`.
`<model_sign>` should take the value of `[mention_proposal, corefqa]`.
<br>
 
## Download the Final CorefQA Model
You can download the final CorefQA model at [link]() and follow the instructions in the prediciton to obtain the score reported in the paper. 


## Descriptions of Directories

Name | Descriptions 
----------- | ------------- 
bert | BERT modules (model,tokenizer,optimization) ref to the `google-research/bert` repository. 
conll-2012 | offical evaluation scripts for CoNLL2012 shared task.
data_utils | modules for processing training data.  
func_builders | the input dataloader and model constructor for CorefQA.
logs | the log files in our experiments. 
models | an implementation of CorefQA/MentionProposal models based on TF.
run | modules for data preparation and training models.
scripts/data | scripts for data preparation and loading pretrained models.
scripts/models | scripts for {train/evaluate} {mention_proposal/corefqa} models on {TPU/GPU}. 
utils | modules including metrics、optimizers. 



## Acknowledgement

Many thanks to `Yuxian Meng` and the previous work `https://github.com/mandarjoshi90/coref`.

## Useful Materials

- TPU Quick Start [link](https://cloud.google.com/tpu/docs/quickstart)
- TPU Available Operations [link](https://cloud.google.com/tpu/docs/tensorflow-ops)

## Contact 

Feel free to discuss papers/code with us through issues/emails!
