from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import string
from pyemd import emd
from torch import nn
from math import log
from itertools import chain

from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertForSequenceClassification
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

import os
import sys
import requests
import zipfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                  output_all_encoded_layers=True)
        return encoded_layers, pooled_output
            
model_nli = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.max_len = 512

model_base = BertModel.from_pretrained('bert-base-uncased')

model_nli.eval()
model_nli.to(device)

model_base.eval()
model_base.to(device)

def truncate(tokens):
    if len(tokens) > tokenizer.max_len - 2:
        tokens = tokens[0:(tokenizer.max_len - 2)]
    return tokens


def process(a):
    a = ["[CLS]"] + truncate(tokenizer.tokenize(a)) + ["[SEP]"]
    a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask):
    model.eval()
    x_seg = torch.zeros_like(x, dtype=torch.long)
    with torch.no_grad():
        x_encoded_layers, pooled_output = model(x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=True)
    return x_encoded_layers


def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]", device=device):
    tokens = [["[CLS]"] + truncate(tokenize(a)) + ["[SEP]"] for a in arr]  # arr 相当于所有句子的list， a是其中一个句子
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]  # i是一个token

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens


def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device=device):
    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                              tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                              idf_dict,
                                                              device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i + batch_size],
                                          attention_mask=mask[i:i + batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens


plus_mask = lambda x, m: x + (1.0 - m).unsqueeze(-1) * 1e30
minus_mask = lambda x, m: x - (1.0 - m).unsqueeze(-1) * 1e30
mul_mask = lambda x, m: x * m.unsqueeze(-1)
masked_reduce_min = lambda x, m: torch.min(plus_mask(x, m), dim=1, out=None)
masked_reduce_max = lambda x, m: torch.max(minus_mask(x, m), dim=1, out=None)
masked_reduce_mean = lambda x, m: mul_mask(x, m).sum(1) / (m.sum(1, keepdim=True) + 1e-10)
masked_reduce_geomean = lambda x, m: np.exp(mul_mask(np.log(x), m).sum(1) / (m.sum(1, keepdim=True) + 1e-10))
idf_reduce_mean = lambda x, m: mul_mask(x, m).sum(1)
idf_reduce_max = lambda x, m, idf: torch.max(mul_mask(minus_mask(x, m), idf), dim=1, out=None)
idf_reduce_min = lambda x, m, idf: torch.min(mul_mask(plus_mask(x, m), idf), dim=1, out=None)


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def slide_window(a, w=3, o=2):
    if a.size - w + 1 <= 0:
        w = a.size
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view.copy().tolist()


def _safe_divide(numerator, denominator):
    return numerator / (denominator + 0.00001)


def load_ngram(ids, embedding, idf, n, o):
    new_a = []
    new_idf = []

    slide_wins = slide_window(np.array(ids), w=n, o=o)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(device)
        tmp = (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0).to(device)
    return new_a, new_idf


def word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True,
                     dropout_rate=0.3, batch_size=256, device=device):
    preds = []
    for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start:batch_start + batch_size]
        batch_hyps = hyps[batch_start:batch_start + batch_size]

        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(batch_refs, model_nli, tokenizer,
                                                                                     idf_dict_ref,
                                                                                     device=device)
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, model_nli, tokenizer,
                                                                                     idf_dict_hyp,
                                                                                     device=device)
    
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

        ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
        hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

        ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
        hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)

        ref_embedding_avg = ref_embedding[-5:].mean(0)
        hyp_embedding_avg = hyp_embedding[-5:].mean(0)

        ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
        hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)


        for i in range(len(ref_tokens)):

            if remove_subwords:
                ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                           w not in set(string.punctuation) and '##' not in w and w not in stop_words]
                hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                           w not in set(string.punctuation) and '##' not in w and w not in stop_words]
            else:
                ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                           w not in set(string.punctuation) and w not in stop_words]
                hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                           w not in set(string.punctuation) and w not in stop_words]

            ref_embedding_i, ref_idf_i = load_ngram(ref_ids, ref_embedding[i], ref_idf[i], n_gram, 1)
            hyp_embedding_i, hyp_idf_i = load_ngram(hyp_ids, hyp_embedding[i], hyp_idf[i], n_gram, 1)

            raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-9)

            distance_matrix = pairwise_distances(raw, raw)

            c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
            c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)

            c1[:len(ref_idf_i)] = ref_idf_i
            c2[-len(hyp_idf_i):] = hyp_idf_i

            c1 = _safe_divide(c1, np.sum(c1))
            c2 = _safe_divide(c2, np.sum(c2))
            score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())

            preds.append(score)
    #print('len(preds): ',len(preds))
    return preds


def BERTScore(hyp, ref, device=device):

    idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_hyp = defaultdict(lambda: 1.)

    ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(ref, model_base, tokenizer, idf_dict_ref,
                                   device=device)
    hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(hyp, model_base, tokenizer, idf_dict_hyp,
                                   device=device)        

    ref_embedding = ref_embedding[8]
    hyp_embedding = hyp_embedding[8]
    
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1) + 1e-30) 
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1) + 1e-30) 
    
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))    
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    
    batch_size = ref_embedding.size(0)

    masks = masks.expand(batch_size, -1, -1)\
                              .contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]
   
    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))

    precision_scale = hyp_idf.unsqueeze(0).expand(1, *hyp_idf.size()).contiguous().view(-1, *hyp_idf.size()[1:]).to(word_precision.device)
    recall_scale = ref_idf.unsqueeze(0).expand(1, *ref_idf.size()).contiguous().view(-1, *ref_idf.size()[1:]).to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1) 
    
    F = 2 * P * R / (P + R)
    F = F.view(1, batch_size)
    
    return F.cpu()