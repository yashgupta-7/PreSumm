import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger
from others.tokenization import BertTokenizer
from pytorch_transformers import XLNetTokenizer

from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET
from cytoolz import curry
nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]


def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)



def load_json(p, lower=False):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        # if (lower):
            # tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            # tgt.append([])
            continue
        if (flag):
            tgt.append(tokens)
            # tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    # print(source, tgt)
    return source, tgt



def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)): #source
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            # print(cur_max_rouge)
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    # print("output", selected)
    # if not selected:
        # print("empty selected")
        # selected.append(0)
    return selected #sorted(selected)

def presumm_reward(doc_sent_list, abstract_sent_list, n_sents=10):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    rwds = [0]
    # cur_max_rouge = max_rouge
    # cur_id = -1
    for i in range(len(sents)): #source
        # if (i in selected):
            # continue
        # c = selected + [i]
        # candidates_1 = [evaluated_1grams[idx] for idx in c]
        # candidates_1 = set.union(*map(set, candidates_1))
        # candidates_2 = [evaluated_2grams[idx] for idx in c]
        # candidates_2 = set.union(*map(set, candidates_2))
        # rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        # rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        if i < n_sents:
            rouge_1 = cal_rouge(evaluated_1grams[i], reference_1grams)['f']
            rouge_2 = cal_rouge(evaluated_2grams[i], reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
        else:
            rouge_score = 0
        rwds.append(rouge_score)

    return rwds[1:]

def presumm_reward2(doc_sent_list, abstract_sent_list, n_sents=3):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    rwds = [0]
    prev_sc = 0
    # cur_max_rouge = max_rouge
    # cur_id = -1
    for i in range(len(sents)): #source
        # if (i >= n_sents):
        #     continue
        c = selected + [i]
        candidates_1 = [evaluated_1grams[idx] for idx in c]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx] for idx in c]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        if i < n_sents:
            rouge_score = rouge_1 + rouge_2 - prev_sc
        else:
            rouge_score = 0
        prev_sc = rouge_1 + rouge_2
        rwds.append(rouge_score)

    return rwds[1:]

def presumm_reward3(doc_sent_list, abstract_sent_list, mode='f'):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    rwds = []
    prev_sc = 0
    # cur_max_rouge = max_rouge
    # cur_id = -1
    for i in range(len(sents)): #source
        # if (i in selected):
        #     continue
        c = selected + [i]
        candidates_1 = [evaluated_1grams[idx] for idx in c]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx] for idx in c]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)[mode]
        rouge_2 = cal_rouge(candidates_2, reference_2grams)[mode]
        rouge_score = rouge_1 + rouge_2 - prev_sc #max with it 0, use the r version, give -0.1, -0.2 ... for EOE
        rwds.append(rouge_score + 1)
        prev_sc = rouge_1 + rouge_2

    return rwds

def presumm_reward4(doc_sent_list, abstract_sent_list, n_sents=10):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    rwds = [0]
    # cur_max_rouge = max_rouge
    # cur_id = -1
    for i in range(len(sents)): #source
        # if (i in selected):
            # continue
        # c = selected + [i]
        # candidates_1 = [evaluated_1grams[idx] for idx in c]
        # candidates_1 = set.union(*map(set, candidates_1))
        # candidates_2 = [evaluated_2grams[idx] for idx in c]
        # candidates_2 = set.union(*map(set, candidates_2))
        # rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        # rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        if i < n_sents:
            rouge_1 = 0 #cal_rouge(evaluated_1grams[i], reference_1grams)['f']
            rouge_2 = cal_rouge(evaluated_2grams[i], reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
        else:
            rouge_score = 0
        rwds.append(rouge_score)

    return rwds[1:]

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False, order=False):
        temp = [[j, i] for i, j in enumerate(sent_labels)]
        # print("hi", sent_labels, temp)
        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        _temp = [[0, -1]] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1
        for l, idx in temp:
            _temp[l] = [1, idx]

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        temp = []
        for ni, oi in enumerate(idxs):
            if _temp[oi][0] == 1:
                temp.append([ni, _temp[oi][1]])
        src = src[:self.args.max_src_nsents]
        temp = [p for p in temp if p[0] < self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        # print("HI2", sent_labels, temp)
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        temp = [p for p in temp if p[0] < len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        temp = sorted(temp, key=lambda x: x[1])
        temp = [x[0] for x in temp]
        # print("bye", sent_labels, temp)
        if order:
            return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, temp
        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
#    corpus_mapping = {}
#    for corpus_type in ['valid', 'test', 'train']:
#        temp = []
#        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
#            temp.append(hashhex(line.strip()))
#        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
#    for f in glob.glob(pjoin(args.raw_path, '*.json')):
#        real_name = f.split('/')[-1].split('.')[0]
#        if (real_name in corpus_mapping['valid']):
#            valid_files.append(f)
#        elif (real_name in corpus_mapping['test']):
#            test_files.append(f)
#        elif (real_name in corpus_mapping['train']):
#            train_files.append(f)
        # else:
        #     train_files.append(f)
    cur = 0
    valid_test_ratio = 0.0
    all_size = len(glob.glob(pjoin(args.raw_path, '*.json')))
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]
        # print(real_name)
        # if (cur < valid_test_ratio*all_size):
        # valid_files.append(f)
        # elif (cur < valid_test_ratio*2*all_size):
        train_files.append(f)
        # else:
            # train_files.append(f)
        cur += 1
        if cur%10000==0:
            print("File", cur)
    corpora = {'train': train_files, 'val': valid_files, 'test': test_files}
    for corpus_type in ['train', 'val', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        # print(a_lst)
        # global vocab_counter
        # vocab_counter = collections.Counter()
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            # print(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}{:s}{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    if p_ct%10000==0:
                        print(pt_file)
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []

        # print("Writing vocab file...")
        # with open(os.path.join("./", "vocab_cnt.pkl"),
                  # 'wb') as vocab_file:
            # pkl.dump(vocab_counter, vocab_file)
        # print("Finished writing vocab file")


def _format_to_lines(params):
    f, args = params
    # print(f)
    source, tgt = load_json(f, args.lower)
    # print()
    # vocab_counter.update(tokens)
    source = [" ".join(sent) for sent in source]
    tgt = [" ".join(sent) for sent in tgt]
    return {'article': source, 'abstract': tgt}




def format_xsum_to_lines(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'valid']

    corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

    for corpus_type in datasets:
        mapped_fnames = corpus_mapping[corpus_type]
        root_src = pjoin(args.raw_path, 'restbody')
        root_tgt = pjoin(args.raw_path, 'firstsentence')
        # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
        realnames = mapped_fnames

        a_lst = [(root_src, root_tgt, n) for n in realnames]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
            if (d is None):
                continue
            dataset.append(d)
            # print(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_xsum_to_lines(params):
    src_path, root_tgt, name = params
    f_src = pjoin(src_path, name + '.restbody')
    f_tgt = pjoin(root_tgt, name + '.fs')
    if (os.path.exists(f_src) and os.path.exists(f_tgt)):
        print(name)
        source = []
        for sent in open(f_src):
            source.append(sent.split())
        tgt = []
        for sent in open(f_tgt):
            tgt.append(sent.split())
        return {'src': source, 'tgt': tgt}
    return None
