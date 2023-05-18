from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from pyemd import emd, emd_with_flow
from math import log
from itertools import chain

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from transformers import *

model_name = 'bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertModel.from_pretrained(model_name, config=config)
model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:8]])
model.eval()
model.to(device)


def truncate(tokens):
    if len(tokens) > tokenizer.model_max_length - 2:
        tokens = tokens[0:(tokenizer.model_max_length - 2)]
    return tokens

def process(a):
    a = ["[CLS]"]+truncate(tokenizer.tokenize(a))+["[SEP]"]
    a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
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
    with torch.no_grad():
        output, _, x_encoded_layers, _ = model(input_ids = x, token_type_ids = None, attention_mask = attention_mask)
    return x_encoded_layers
       
def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]", device=device):
    
    tokens = [["[CLS]"]+truncate(tokenize(a))+["[SEP]"] for a in arr]  
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]
    
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
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            print(batch_embedding)
            print(padded_sens[i:i+batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens

def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)

def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def MoverScore(hyp, ref, device=device):

    idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_hyp = defaultdict(lambda: 1.)

    ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(ref, model, tokenizer, idf_dict_ref,
                                   device=device)
    hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(hyp, model, tokenizer, idf_dict_hyp,
                                   device=device)
                 
    ref_embedding = ref_embedding[-1]
    hyp_embedding = hyp_embedding[-1]
    
    raw = torch.cat([ref_embedding, hyp_embedding], 1)
                         
    raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
    
    dst = batched_cdist_l2(raw, raw).cpu().numpy()
    
    c1 = np.zeros(raw.shape[1], dtype=np.float)
    c2 = np.zeros_like(c1)
    
    ref_idf = ref_idf.reshape([-1])
    hyp_idf = hyp_idf.reshape([-1])
 
    c1[:len(ref_idf)] = ref_idf
    c2[len(ref_idf):] = hyp_idf
    
    c1 = _safe_divide(c1, np.sum(c1))
    c2 = _safe_divide(c2, np.sum(c2))
    
    score = emd(c1, c2, np.asarray(dst[0], dtype='float64'))        

    return 1./(1. + score)

def BERTScore(hyp, ref, device=device):

    idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_hyp = defaultdict(lambda: 1.)

    ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(ref, model, tokenizer, idf_dict_ref,
                                   device=device)
    hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(hyp, model, tokenizer, idf_dict_hyp,
                                   device=device)        

    ref_embedding = ref_embedding[-1]
    hyp_embedding = hyp_embedding[-1]
    
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