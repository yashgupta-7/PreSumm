import argparse
from train import str2bool
from prepro.data_builder import BertData, greedy_selection
import json
import argparse
import glob
import os
import random
import signal
import time

import torch
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

import distributed
import models
from models import data_loader, model_builder
from models.data_loader import load_dataset, DataIterator, Batch
from models.model_builder import ExtSummarizer, ExtSummarizerNew
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger
import bisect

########################################################################################################
########################################################################################################
########################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-pretrained_model", default='bert', type=str)
parser.add_argument("-mode", default='', type=str)
parser.add_argument("-select_mode", default='greedy', type=str)
parser.add_argument("-map_path", default='../../data/')
parser.add_argument("-raw_path", default='../../line_data')
parser.add_argument("-save_path", default='../../data/')
parser.add_argument("-shard_size", default=2000, type=int)
parser.add_argument('-min_src_nsents', default=3, type=int)
parser.add_argument('-max_src_nsents', default=100, type=int)
parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
parser.add_argument('-min_tgt_ntokens', default=5, type=int)
parser.add_argument('-max_tgt_ntokens', default=500, type=int)
parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument('-log_file', default='../../logs/cnndm.log')
parser.add_argument('-dataset', default='')
parser.add_argument('-n_cpus', default=2, type=int)
args_bert = parser.parse_args("")
args_bert.mode = "format_to_bert"
args_bert.lower = True
args_bert.n_cpus = 1

bert = BertData(args_bert)

########################################################################################################
########################################################################################################
########################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
parser.add_argument("-model_path", default='../models/')
parser.add_argument("-result_path", default='../results/cnndm')
parser.add_argument("-temp_dir", default='../temp')
parser.add_argument("-batch_size", default=140, type=int)
parser.add_argument("-test_batch_size", default=200, type=int)
parser.add_argument("-max_pos", default=512, type=int)
parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-load_from_extractive", default='', type=str)
parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-lr_bert", default=2e-3, type=float)
parser.add_argument("-lr_dec", default=2e-3, type=float)
parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-dec_dropout", default=0.2, type=float)
parser.add_argument("-dec_layers", default=6, type=int)
parser.add_argument("-dec_hidden_size", default=768, type=int)
parser.add_argument("-dec_heads", default=8, type=int)
parser.add_argument("-dec_ff_size", default=2048, type=int)
parser.add_argument("-enc_hidden_size", default=512, type=int)
parser.add_argument("-enc_ff_size", default=512, type=int)
parser.add_argument("-enc_dropout", default=0.2, type=float)
parser.add_argument("-enc_layers", default=6, type=int)
parser.add_argument("-ext_dropout", default=0.2, type=float)
parser.add_argument("-ext_layers", default=2, type=int)
parser.add_argument("-ext_hidden_size", default=768, type=int)
parser.add_argument("-ext_heads", default=8, type=int)
parser.add_argument("-ext_ff_size", default=2048, type=int)
parser.add_argument("-label_smoothing", default=0.1, type=float)
parser.add_argument("-generator_shard_size", default=32, type=int)
parser.add_argument("-alpha",  default=0.6, type=float)
parser.add_argument("-beam_size", default=5, type=int)
parser.add_argument("-min_length", default=15, type=int)
parser.add_argument("-max_length", default=150, type=int)
parser.add_argument("-max_tgt_len", default=140, type=int)
parser.add_argument("-param_init", default=0, type=float)
parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-optim", default='adam', type=str)
parser.add_argument("-lr", default=1, type=float)
parser.add_argument("-beta1", default= 0.9, type=float)
parser.add_argument("-beta2", default=0.999, type=float)
parser.add_argument("-warmup_steps", default=8000, type=int)
parser.add_argument("-warmup_steps_bert", default=8000, type=int)
parser.add_argument("-warmup_steps_dec", default=8000, type=int)
parser.add_argument("-max_grad_norm", default=0, type=float)
parser.add_argument("-save_checkpoint_steps", default=5, type=int)
parser.add_argument("-accum_count", default=1, type=int)
parser.add_argument("-report_every", default=1, type=int)
parser.add_argument("-train_steps", default=1000, type=int)
parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)
parser.add_argument('-log_file', default='../logs/cnndm.log')
parser.add_argument('-seed', default=666, type=int)
parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-test_from", default='')
parser.add_argument("-test_start_from", default=-1, type=int)
parser.add_argument("-train_from", default='')
parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
args_train = parser.parse_args("")

args_train.task = "ext"
args_train.mode = "test"
args_train.batch_size = 3000
args_train.test_batch_size = 500
args_train.model_path  =  "../bertext_ckpt/"
args_train.sep_optim  = True
args_train.use_interval = True
args_train.visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"] #"3"
args_train.max_pos = 512
args_train.max_length = 200
args_train.alpha = 0.95
args_train.min_length = 50
if os.environ["FB"] == "1":
    args_train.finetune_bert = True
else:
    args_train.finetune_bert = False
print("Finetuning BERT?", args_train.finetune_bert)
args_train.test_from = "/exp/yashgupta/PreSumm/bertext_ckpt/bertext_cnndm_transformer.pt"
args_train.gpu_ranks = [int(i) for i in range(len(args_train.visible_gpus.split(',')))]
args_train.world_size = len(args_train.gpu_ranks)
os.environ["CUDA_VISIBLE_DEVICES"] = args_train.visible_gpus

# checkpoint = torch.load(args_train.test_from, map_location=lambda storage, loc: storage)
# opt = vars(checkpoint['opt'])
# for k in opt.keys():
#     if (k in model_flags):
#         setattr(args_train, k, opt[k])

device = "cpu" if args_train.visible_gpus == '-1' else "cuda"
# model = ExtSummarizer(args_train, device, checkpoint)
# device_id = 0 if device == "cuda" else -1
# trainer = build_trainer(args_train, device_id, model, None)

# model_trans = ExtSummarizerNew(args_train, device, checkpoint)

########################################################################################################
########################################################################################################
########################################################################################################

def preprocess(args, ex, is_test, order=False):
        src = ex['src']
        tgt = ex['tgt'][:args.max_tgt_len][:-1]+[2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if(not args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:args.max_pos - 1] + end_id
        segs = segs[:args.max_pos]
        max_sent_id = bisect.bisect_left(clss, args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]
        if (is_test and order):
            return src, tgt, segs, clss, src_sent_labels, [x for x in ex["ord_labels"] if x < max_sent_id], src_txt, tgt_txt
        if(is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

def extractor(raw_srcs, raw_tgts, is_test = True): #raw_sents is batch of articles (list of list of sentences)
    dataset = []
    for source, tgt in zip(raw_srcs, raw_tgts):
        source = [s.split(" ") for s in source]
        tgt = [s.split(" ") for s in tgt]
        sent_labels = greedy_selection(source[:args_bert.max_src_nsents], tgt, 3)
        # print(sent_labels)
        if (args_bert.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args_bert.use_bert_basic_tokenizer,
                                is_test=is_test)
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                    "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                    'src_txt': src_txt, "tgt_txt": tgt_txt}
        dataset.append(b_data_dict)

    batch = []  
    for ex in dataset:
        ex = preprocess(args_train, ex, is_test)
        batch.append(ex)
    batch = models.data_loader.Batch(batch, device, is_test)
    step = -1 # so that no ROUGE calculation done
    return trainer.test_extract([batch], step)[0]


def extractor_batch(raw_srcs, raw_tgts, is_test = True, n_ext=3): #raw_sents is batch of articles (list of list of sentences)
    dataset = []
    for source, tgt in zip(raw_srcs, raw_tgts):
        source = [s.split(" ") for s in source]
        tgt = [s.split(" ") for s in tgt]
        sent_labels = greedy_selection(source[:args_bert.max_src_nsents], tgt, n_ext)
        # print(sent_labels)
        if (args_bert.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args_bert.use_bert_basic_tokenizer,
                                is_test=is_test)
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                    "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                    'src_txt': src_txt, "tgt_txt": tgt_txt}
        dataset.append(b_data_dict)

    batch = []  
    for ex in dataset:
        ex = preprocess(args_train, ex, is_test)
        batch.append(ex)
    batch = models.data_loader.Batch(batch, device, is_test)
    step = -1 # so that no ROUGE calculation done
    return trainer.test_extract([batch], step, n_ext=n_ext), batch

def get_batch(raw_srcs, raw_tgts, is_test = True, n_ext=3):
    dataset = []
    for source, tgt in zip(raw_srcs, raw_tgts):
        source = [s.split(" ") for s in source]
        tgt = [s.split(" ") for s in tgt]
        sent_labels = greedy_selection(source[:args_bert.max_src_nsents], tgt, n_ext)
        # print(sent_labels)
        if (args_bert.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args_bert.use_bert_basic_tokenizer,
                                is_test=is_test)
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                    "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                    'src_txt': src_txt, "tgt_txt": tgt_txt}
        dataset.append(b_data_dict)

    batch = []  
    for ex in dataset:
        ex = preprocess(args_train, ex, is_test)
        batch.append(ex)
    batch = models.data_loader.Batch(batch, device, is_test)
    return batch

def get_batch_trans(t_batch, is_test = True, n_ext=5):
    # print(t_batch)
    dataset = []
    for source, tgt in t_batch: #zip(raw_srcs, raw_tgts):
        source = [s.split(" ") for s in source]
        tgt =  [s.split(" ") for s in tgt] #[source[s] for s in tgt]
        # print(source, tgt)
        sent_labels = greedy_selection(source[:args_bert.max_src_nsents], tgt, n_ext)
        # print(sent_labels)
        if (args_bert.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args_bert.use_bert_basic_tokenizer,
                                is_test=is_test, order=True)
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, ord_labels = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                    "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                    'src_txt': src_txt, "tgt_txt": tgt_txt, "ord_labels": ord_labels}
        dataset.append(b_data_dict)

    batch = []  
    for ex in dataset:
        ex = preprocess(args_train, ex, is_test, order=True)
        batch.append(ex)
    batch = models.data_loader.Batch(batch, device, is_test, order=True)
    return batch