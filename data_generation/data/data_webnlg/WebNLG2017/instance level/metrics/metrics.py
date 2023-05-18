import warnings
warnings.filterwarnings('ignore')

import sacrebleu
#from metrics import S3, BERT, JS_eval, ROUGE
from metrics import S3, JS_eval, ROUGE
import random
import numpy as np
from metrics import moverscore_d
import sklearn.svm
from collections import defaultdict
import torchmetrics

from metrics import bary_score, depth_score, infolm

class Metrics():
    def __init__(self, sent_smooth_method='exp', sent_smooth_value=None, sent_use_effective_order=True, \
       smooth_method='exp', smooth_value=None, force=False, lowercase=False, \
       use_effective_order=False, n_workers=24, ncorder=6, beta=2, ngram=1, we=None, pyr=True):

        """
        BLEU metric: https://github.com/mjpost/sacrebleu
        Args:
                :param smooth_value: For 'floor' smoothing, the floor value to use.
                :param use_effective_order: Account for references that are shorter than the largest n-gram.
                :param force: Ignore data that looks already tokenized
                :param lowercase: Lowercase the data
                :param n_workers: number of processes to use if using multiprocessing
                sent* parameters are the same but specify what is used for evaluate_example
        """

        self.sent_smooth_method = sent_smooth_method
        self.sent_smooth_value = sent_smooth_value
        self.sent_use_effective_order = sent_use_effective_order
        self.smooth_method = smooth_method
        self.smooth_value = smooth_value
        self.force = force
        self.lowercase = lowercase
        self.use_effective_order = use_effective_order

        """
        Chrf++ metric: https://github.com/mjpost/sacrebleu
        Args:
                :param ncorder: character n-gram order
                :param beta: beta parameter to balance precision and recall
        """
        self.ncorder = ncorder
        self.beta = beta

        """
        ROUGE-related metrics by Maxime 
        """
        self.we = we

        """
        S3 metric: https://github.com/UKPLab/emnlp-ws-2017-s3/tree/b524407ada525c81ceacd2590076e20103213e3b
        """
        self.s3_model_folders = "metrics/s3_models"
        self.we = we
        
        """
        JS metric by Maxime 
        """

    def Random(self, sys, ref):
        return random.random()

    def ROUGE_WE_1(self, sys, ref):  
        sys = [sys]
        ref = [[r] for r in ref]
        return - ROUGE.rouge_n_we(sys, ref, self.we, n=1, alpha=0)

    def ROUGE_WE_2(self, sys, ref):  
        sys = [sys]
        ref = [[r] for r in ref]
        return - ROUGE.rouge_n_we(sys, ref, self.we, n=2, alpha=0)

    def ROUGE_1(self, sys, ref):
        sys = [sys]
        ref = [[r] for r in ref]
        
        return - ROUGE.rouge_n(sys, ref, n=1, alpha=0)
    
    def ROUGE_2(self, sys, ref):
        sys = [sys]
        ref = [[r] for r in ref]
        return - ROUGE.rouge_n(sys, ref, n=2, alpha=0)
    
    def ROUGE_L(self, sys, ref):
        sys = [sys]
        ref = [[r] for r in ref]
        return - ROUGE.rouge_l(sys, ref, alpha=0)

    def JS_1(self, sys, ref):     
        sys = [sys]
        ref = [[r] for r in ref]
        return -JS_eval.JS_eval(sys, ref, n=1)

    def JS_2(self, sys, ref):  
        sys = [sys]
        ref = [[r] for r in ref]
        return -JS_eval.JS_eval(sys, ref, n=2)
    
    def S3_pyr(self, sys, ref):     
        sys = [sys]
        ref = [[r] for r in ref]
        return S3.S3(ref, sys, self.we, self.s3_model_folders)[0]

    def S3_resp(self, sys, ref): 
        sys = [sys]
        ref = [[r] for r in ref]
        return S3.S3(ref, sys, self.we, self.s3_model_folders)[1]

    def Chrfpp(self, sys, ref):
#        ref = [ref]
        score = sacrebleu.sentence_chrf(sys, ref, order=self.ncorder, beta=self.beta)
        return - score.score

    def BLEU(self, sys, ref):
#        ref = [ref]
        score = sacrebleu.sentence_bleu(sys, ref, smooth_method=self.sent_smooth_method, \
             smooth_value=self.sent_smooth_value, use_effective_order=self.sent_use_effective_order)
        return score.score

    def BERTScore(self, sys, ref):
#        sys, ref = [sys], [ref]
#        return BERT.BERTScore(sys, ref)
        return moverscore_d.BERTScore([sys] * len(ref), ref).mean().item()

    def MoverScore(self, sys, ref):
#        score = .0
#        for r in ref:
#            score += BERT.MoverScore([sys], [r])
#        return score / len(ref)
    
        idf_dict_ref = defaultdict(lambda: 1.)
        idf_dict_hyp = defaultdict(lambda: 1.)
        return np.mean(moverscore_d.word_mover_score(ref, [sys] * len(ref), idf_dict_ref, idf_dict_hyp))

    def BaryScore_W(sys,ref):
        sys, ref = [sys], [ref]
        return bary_score.BaryScore(sys,ref)['baryscore_W'][0]

    def BaryScore_SD_10(sys,ref):
        sys, ref = [sys], [ref]
        return bary_score.BaryScore(sys,ref)['baryscore_SD_10'][0]

    def BaryScore_SD_1(sys,ref):
        sys, ref = [sys], [ref]
        return bary_score.BaryScore(sys,ref)['baryscore_SD_1'][0]
        
    def BaryScore_SD_5(sys,ref):
        sys, ref = [sys], [ref]
        return bary_score.BaryScore(sys,ref)['baryscore_SD_5'][0]

    def DepthScore(sys,ref):
        sys, ref = [sys], [ref]
        return depth_score.DepthScore(sys,ref)['depth_score'][0]

    def Infolm_kl(sys,ref):
        sys, ref = [sys], [ref]
        return - infolm.Infolm(sys,ref,'kl')['kl'][0]
    def Infolm_alpha(sys,ref):
        sys, ref = [sys], [ref]
        return - infolm.Infolm(sys,ref,'alpha')['alpha'][0]
    def Infolm_renyi(sys,ref):
        sys, ref = [sys], [ref]
        return infolm.Infolm(sys,ref,'renyi')['renyi'][0]
    def Infolm_beta(sys,ref):
        sys, ref = [sys], [ref]
        return infolm.Infolm(sys,ref,'beta')['beta'][0]
    def Infolm_ab(sys,ref):
        sys, ref = [sys], [ref]
        return infolm.Infolm(sys,ref,'ab')['ab'][0]
    def Infolm_l1(sys,ref):
        sys, ref = [sys], [ref]
        return infolm.Infolm(sys,ref,'l1')['l1'][0]
    def Infolm_l2(sys,ref):
        sys, ref = [sys], [ref]
        return infolm.Infolm(sys,ref,'l2')['l2'][0]
    def Infolm_linf(sys,ref):
        sys, ref = [sys], [ref]
        return infolm.Infolm(sys,ref,'linf')['linf'][0]
    def Infolm_fisher_rao(sys,ref):
        sys, ref = [sys], [ref]
        return infolm.Infolm(sys,ref,'fisher_rao')['fisher_rao'][0]
    
    def CharErrorRate(sys,ref):
        sys, ref = [sys], [ref]
        return torchmetrics.CharErrorRate()(sys,ref).item()
    def ExtendedEditDistance(sys,ref):
        sys, ref = [sys], [ref]
        return torchmetrics.ExtendedEditDistance()(sys,ref).item()
    def MatchErrorRate(sys,ref):
        sys, ref = [sys], [ref]
        return torchmetrics.MatchErrorRate()(sys,ref).item()
    # def Perplexity(sys,ref): # Doesn't work
    #     sys, ref = [sys], [ref]
    #     return torchmetrics.Perplexity()(sys,ref).item()
    def TranslationEditRate(sys,ref):
        sys, ref = [sys], [ref]
        return torchmetrics.TranslationEditRate()(sys,ref).item()
    def WordErrorRate(sys,ref):
        sys, ref = [sys], [ref]
        return torchmetrics.WordErrorRate()(sys,ref).item()
    def WordInfoLost(sys,ref):
        sys, ref = [sys], [ref]
        return torchmetrics.WordInfoLost()(sys,ref).item()
    # def WordInfoPreserved(sys,ref): # just the negative of WordInfoLost
    #     sys, ref = [sys], [ref]
    #     return torchmetrics.WordInfoPreserved()(sys,ref).item()