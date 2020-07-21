#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li
# description:
# generate tfrecord for train/dev/test set for the model. 
# TODO (xiaoya): need to add help description for args 



import os
import sys 
import re
import json  
import argparse 
import numpy as np 
import tensorflow as tf
from collections import defaultdict

REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from data_utils import conll
from bert.tokenization import FullTokenizer

SPEAKER_START = '[unused19]'
SPEAKER_END = '[unused73]'
subtoken_maps = {}
gold = {}



"""
Desc:
a single training/test example for the squad dataset.
suppose origin input_tokens are :
['[unused19]', 'speaker', '#', '1', '[unused73]', '-', '-', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 
'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 
'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 
'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 
'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 
'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '.'] 
IF sliding window size is 50. 
Args:
doc_idx: a string: cctv/bn/0001
sentence_map: 
    e.g. [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7]
subtoken_map: 
    e.g. [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 53, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 97, 98, 99, 99, 99, 100, 101, 102, 103]
flattened_window_input_ids: [num-window, window-size]
    e.g. before bert_tokenizer convert subtokens into ids:
    [['[CLS]', '[unused19]', 'speaker', '#', '1', '[unused73]', '-', '-', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', '[SEP]'],
    ['[CLS]', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '[SEP]'],
    ['[CLS]', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', '[SEP]'],
    ['[CLS]', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '[SEP]'],
    ['[CLS]', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']] 
flattened_window_masked_ids: [num-window, window-size]
    e.g.: before bert_tokenizer ids:
    [[-3, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
    [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
    [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, -1, -1, -1, -1, -1, -1, 68, 69, 70, 71, 72, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
    [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
    [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, -3, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]]
span_start: 
    e.g.: mention start indices in the original document 
        [17, 20, 26, 43, 60, 85, 86]
span_end:
    e.g.: mention end indices in the original document 
cluster_ids: 
    e.g.: cluster ids for the (span_start, span_end) pairs
    [1, 1, 2, 2, 2, 3, 3] 
check the mention in the subword list: 
    1. ['its']
    1. ['the', 'Chinese', 'securities', 'regulatory', 'department']
    2. ['this', 'stock', 'reform']
    2. ['the', 'stock', 'reform']
    2. ['the', 'stock', 'reform']
    3. ['you']
    3. ['everyone']
"""



def prepare_train_dataset(input_file, output_data_dir, output_filename, window_size, num_window, 
    tokenizer=None, vocab_file=None, language="english", max_doc_length=None, genres=None, 
    max_num_mention=10, max_num_cluster=30, demo=False, lowercase=False):

    if vocab_file is None:
        if not lowercase:
            vocab_file = os.path.join(REPO_PATH, "data_utils", "uppercase_vocab.txt")
        else:
            vocab_file = os.path.join(REPO_PATH, "data_utils", "lowercase_vocab.txt")

    if tokenizer is None:
        tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=lowercase)

    writer = tf.python_io.TFRecordWriter(os.path.join(output_data_dir, "{}.{}.tfrecord".format(output_filename, language)))
    doc_map = {}
    documents = read_conll_file(input_file)
    for doc_idx, document in enumerate(documents):
        doc_info = parse_document(document, language)
        tokenized_document = tokenize_document(genres, doc_info, tokenizer, max_doc_length=max_doc_length)
        doc_key = tokenized_document['doc_key']
        token_windows, mask_windows, text_len = convert_to_sliding_window(tokenized_document, window_size)
        input_id_windows = [tokenizer.convert_tokens_to_ids(tokens) for tokens in token_windows]
        span_start, span_end, mention_span, cluster_ids = flatten_clusters(tokenized_document['clusters'])

        tmp_speaker_ids = tokenized_document["speakers"] 
        tmp_speaker_ids = [[0]*130]* num_window
        instance = (input_id_windows, mask_windows, text_len, tmp_speaker_ids, tokenized_document["genre"], span_start, span_end, cluster_ids, tokenized_document['sentence_map'])   
        write_instance_to_example_file(writer, instance, doc_key, window_size=window_size, num_window=num_window, 
            max_num_mention=max_num_mention, max_num_cluster=max_num_cluster)
        doc_map[doc_idx] = doc_key
        if demo and doc_idx > 3:
            break 
    with open(os.path.join(output_data_dir, "{}.{}.map".format(output_filename, language)), 'w') as fo:
        json.dump(doc_map, fo, indent=2)



def write_instance_to_example_file(writer, instance, doc_key, window_size=64, num_window=5, max_num_mention=20,
    max_num_cluster=30, pad_idx=-1):
    input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, sentence_map = instance 
    input_id_windows = input_ids 
    mask_windows = input_mask 
    flattened_input_ids = [i for j in input_id_windows for i in j]
    flattened_input_mask = [i for j in mask_windows for i in j]
    cluster_ids = [int(tmp) for tmp in cluster_ids]

    max_sequence_len = int(num_window)
    max_seg_len = int(window_size)

    sentence_map = clip_or_pad(sentence_map, max_sequence_len*max_seg_len, pad_idx=pad_idx)
    text_len = clip_or_pad(text_len, max_sequence_len, pad_idx=pad_idx)
    tmp_subtoken_maps = clip_or_pad(subtoken_maps[doc_key], max_sequence_len*max_seg_len, pad_idx=pad_idx)

    tmp_speaker_ids = clip_or_pad(speaker_ids[0], max_sequence_len*max_seg_len, pad_idx=pad_idx)

    flattened_input_ids = clip_or_pad(flattened_input_ids, max_sequence_len*max_seg_len, pad_idx=pad_idx)
    flattened_input_mask = clip_or_pad(flattened_input_mask, max_sequence_len*max_seg_len, pad_idx=pad_idx)
    gold_starts = clip_or_pad(gold_starts, max_num_mention, pad_idx=pad_idx)
    gold_ends = clip_or_pad(gold_ends, max_num_mention, pad_idx=pad_idx)
    cluster_ids = clip_or_pad(cluster_ids, max_num_cluster, pad_idx=pad_idx)

    features = {
        'sentence_map': create_int_feature(sentence_map), 
        'text_len': create_int_feature(text_len), 
        'subtoken_map': create_int_feature(tmp_subtoken_maps), 
        'speaker_ids': create_int_feature(tmp_speaker_ids), 
        'flattened_input_ids': create_int_feature(flattened_input_ids),
        'flattened_input_mask': create_int_feature(flattened_input_mask),
        'span_starts': create_int_feature(gold_starts), 
        'span_ends': create_int_feature(gold_ends), 
        'cluster_ids': create_int_feature(cluster_ids),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def clip_or_pad(var, max_var_len, pad_idx=-1):
    
    if len(var) >= max_var_len:
        return var[:max_var_len]
    else:
        pad_var  = (max_var_len - len(var)) * [pad_idx]
        var = list(var) + list(pad_var) 
        return var 


def flatten_clusters(clusters):

    span_starts = []
    span_ends = []
    cluster_ids = []
    mention_span = []
    for cluster_id, cluster in enumerate(clusters):
        for start, end in cluster:
            span_starts.append(start)
            span_ends.append(end)
            mention_span.append((start, end))
            cluster_ids.append(cluster_id + 1)
    return span_starts, span_ends, mention_span, cluster_ids


def read_conll_file(conll_file_path):
    documents = []
    with open(conll_file_path, "r", encoding="utf-8") as fi:
        for line in fi:
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line.strip())
    return documents


def parse_document(document, language):
    """
    get basic information from one document annotation.
    :param document:
    :param language: english, chinese or arabic
    :return:
    """
    doc_key = document[0]
    sentences = [[]]
    speakers = []
    coreferences = []
    word_idx = -1
    last_speaker = ''
    for line_id, line in enumerate(document[1]):
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            word_idx += 1
            word = normalize_word(row[3], language)
            sentences[-1].append(word)
            speaker = row[9]
            if speaker != last_speaker:
                speakers.append((word_idx, speaker))
                last_speaker = speaker
            coreferences.append(row[-1])
        else:
            sentences.append([])
    clusters = coreference_annotations_to_clusters(coreferences)
    doc_info = {'doc_key': doc_key, 'sentences': sentences[: -1], 'speakers': speakers, 'clusters': clusters}
    return doc_info


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def coreference_annotations_to_clusters(annotations):
    """
    convert coreference information to clusters
    :param annotations:
    :return:
    """
    clusters = defaultdict(list)
    coref_stack = defaultdict(list)
    for word_idx, annotation in enumerate(annotations):
        if annotation == '-':
            continue
        for ann in annotation.split('|'):
            cluster_id = int(ann.replace('(', '').replace(')', ''))
            if ann[0] == '(' and ann[-1] == ')':
                clusters[cluster_id].append((word_idx, word_idx))
            elif ann[0] == '(':
                coref_stack[cluster_id].append(word_idx)
            elif ann[-1] == ')':
                span_start = coref_stack[cluster_id].pop()
                clusters[cluster_id].append((span_start, word_idx))
            else:
                raise NotImplementedError
    assert all([len(starts) == 0 for starts in coref_stack.values()])
    return list(clusters.values())


def checkout_clusters(doc_info):
    words = [i for j in doc_info['sentences'] for i in j]
    clusters = [[' '.join(words[start: end + 1]) for start, end in cluster] for cluster in doc_info['clusters']]
    print(clusters)


def tokenize_document(genres, doc_info, tokenizer, max_doc_length):
    """
    tokenize into sub tokens
    :param doc_info:
    :param tokenizer:
    max_doc_length: pad to max_doc_length
    :return:
    """
    genres = {g: i for i, g in enumerate(genres)}
    sub_tokens = []  # all sub tokens of a document
    sentence_map = []  # collected tokenized tokens -> sentence id
    subtoken_map = []  # collected tokenized tokens -> original token id

    word_idx = -1

    for sentence_id, sentence in enumerate(doc_info['sentences']):
        for token in sentence:
            word_idx += 1
            word_tokens = tokenizer.tokenize(token)
            sub_tokens.extend(word_tokens)
            sentence_map.extend([sentence_id] * len(word_tokens))
            subtoken_map.extend([word_idx] * len(word_tokens))
    if max_doc_length:
        num_to_pad = max_doc_length - len(sub_tokens)
        sub_tokens.extend(["[PAD]"] * num_to_pad)
        sentence_map.extend([sentence_map[-1]+1] * num_to_pad)
        subtoken_map.extend(list(range(word_idx+1, num_to_pad+1+word_idx)))
    subtoken_maps[doc_info['doc_key']] = subtoken_map
    genre = genres.get(doc_info['doc_key'][:2], 0)
    speakers = {subtoken_map.index(word_index): tokenizer.tokenize(speaker)
                for word_index, speaker in doc_info['speakers']}
    clusters = [[(subtoken_map.index(start), len(subtoken_map) - 1 - subtoken_map[::-1].index(end))
                 for start, end in cluster] for cluster in doc_info['clusters']]
    tokenized_document = {'sub_tokens': sub_tokens, 'sentence_map': sentence_map, 'subtoken_map': subtoken_map,
                          'speakers': speakers, 'clusters': clusters, 'doc_key': doc_info['doc_key'], 
                          "genre": genre}
    return tokenized_document


def convert_to_sliding_window(tokenized_document, sliding_window_size):
    """
    construct sliding windows, allocate tokens and masks into each window
    :param tokenized_document:
    :param sliding_window_size:
    :return:
    """
    expanded_tokens, expanded_masks = expand_with_speakers(tokenized_document)
    sliding_windows = construct_sliding_windows(len(expanded_tokens), sliding_window_size - 2)
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window
    text_len = []

    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = expanded_tokens[window_start: window_end]
        original_masks = expanded_masks[window_start: window_end]
        window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = ['[CLS]'] + original_tokens + ['[SEP]'] + ['[PAD]'] * (
                sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
        token_calculate = [tmp for tmp in one_window_mask if tmp >= 0]
        text_len.append(len(token_calculate))
        assert len(one_window_token) == sliding_window_size
        assert len(one_window_mask) == sliding_window_size
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)
    assert len(tokenized_document['sentence_map']) == sum([i >= 0 for j in mask_windows for i in j])

    text_len = np.array(text_len)
    return token_windows, mask_windows, text_len


def expand_with_speakers(tokenized_document):
    """
    add speaker name information
    :param tokenized_document: tokenized document information
    :return:
    """
    expanded_tokens = []
    expanded_masks = []
    for token_idx, token in enumerate(tokenized_document['sub_tokens']):
        if token_idx in tokenized_document['speakers']:
            speaker = [SPEAKER_START] + tokenized_document['speakers'][token_idx] + [SPEAKER_END]
            expanded_tokens.extend(speaker)
            expanded_masks.extend([-1] * len(speaker))
        expanded_tokens.append(token)
        expanded_masks.append(token_idx)
    return expanded_tokens, expanded_masks


def construct_sliding_windows(sequence_length, sliding_window_size):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_files_dir", default="/home/lixiaoya/data", type=str, required=True)
    parser.add_argument("--target_output_dir", default="/home/lixiaoya/tfrecord_data", type=str, required=True)
    parser.add_argument("--num_window", default=5, type=int, required=True)
    parser.add_argument("--window_size", default=64, type=int, required=True)
    parser.add_argument("--max_num_mention", default=30, type=int)
    parser.add_argument("--max_num_cluster", default=20, type=int)
    parser.add_argument("--vocab_file", default="/home/lixiaoya/spanbert_large_cased/vocab.txt", type=str)
    parser.add_argument("--language", default="english", type=str)
    parser.add_argument("--max_doc_length", default=600, type=int)
    parser.add_argument("--lowercase", help="DO or NOT lowercase the datasets.", action="store_true")
    parser.add_argument("--demo", help="Wether to generate a small dataset for testing the code.", action="store_true")
    parser.add_argument('--genres', default=["bc","bn","mz","nw","pt","tc","wb"])
    parser.add_argument("--seed", default=2333, type=int)

    args = parser.parse_args()

    os.makedirs(args.target_output_dir, exist_ok=True)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    return args


def main():
    args_config = parse_args()

    print("*"*60)
    print("***** ***** show configs ***** ***** ")
    print("window_size : {}".format(str(args_config.window_size)))
    print("num_window : {}".format(str(args_config.num_window)))
    print("*"*60)

    for data_sign in ["train", "dev", "test"]:
        source_data_file = os.path.join(args_config.source_files_dir, "{}.{}.v4_gold_conll".format(data_sign, args_config.language))
        output_filename = "{}.overlap.corefqa".format(data_sign)
        
        if args_config.demo:
            if args_config.lowercase:
                output_filename="demo.lowercase.{}.overlap.corefqa".format(data_sign)
            else:
                output_filename="demo.{}.overlap.corefqa".format(data_sign)

        print("$"*60)
        print("generate {}/{}".format(args_config.target_output_dir, output_filename))
        prepare_train_dataset(source_data_file, args_config.target_output_dir, output_filename, args_config.window_size, 
            args_config.num_window, vocab_file=args_config.vocab_file, language=args_config.language, 
            max_doc_length=args_config.max_doc_length, genres=args_config.genres, max_num_mention=args_config.max_num_mention,
            max_num_cluster=args_config.max_num_cluster, demo=args_config.demo, lowercase=args_config.lowercase)




if __name__ == "__main__":
    main()

    # please refer ${REPO_PATH}/scripts/data/generate_tfrecord_dataset.sh 
    # 
    # for generate tfrecord datasets 
    # 
    # python3 build_dataset_to_tfrecord.py \
    # --source_files_dir /xiaoya/data \
    # --target_output_dir /xiaoya/corefqa_data/overlap_64_2 \
    # --num_window 2 \
    # --window_size 64 \
    # --max_num_mention 50 \
    # --max_num_cluster 30 \
    # --vocab_file /xiaoya/pretrain_ckpt/cased_L-12_H-768_A-12/vocab.txt \
    # --language english \
    # --max_doc_length 600 
    # 



