#import warnings
#warnings.filterwarnings('ignore')

import sacrebleu
#from metrics import S3, BERT, JS_eval, ROUGE
from metrics import S3, JS_eval, ROUGE
import random
import numpy as np
from metrics import moverscore_d
import sklearn.svm
from collections import defaultdict
import torchmetrics
import tensorflow as tf
from bleurt import score
from comet import download_model, load_from_checkpoint
from metrics.bary_score import BaryScoreMetric
from metrics.depth_score import DepthScoreMetric
from metrics.infolm import InfoLM

class Metrics():
    def __init__(self, ncorder=6, beta=2, we=None, device='cpu'):

        """
        BLEU metric: https://github.com/mjpost/sacrebleu
        Args:
                :param use_effective_order: Account for references that are shorter than the largest n-gram.
                :param force: Ignore data that looks already tokenized
                :param lowercase: Lowercase the data
                :param n_workers: number of processes to use if using multiprocessing
                sent* parameters are the same but specify what is used for evaluate_example
        """
        self.device=device

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
        """
        S3 metric: https://github.com/UKPLab/emnlp-ws-2017-s3/tree/b524407ada525c81ceacd2590076e20103213e3b
        """
        # self.s3_model_folders = "metrics/s3_models"
        self.we = we
        
        """
        JS metric by Maxime 
        """

            
            

class Random(Metrics):
    def __init__(self,device='cpu'):
        Metrics.__init__()
        self.device = device
    def compute_score(self,sys,ref):
        return random.random()

class ROUGE_WE_1(Metrics):
    def __init__(self,we,device='cpu'):
        Metrics().__init__(self,we=we)
        self.we=we
        self.device = device
    def compute_score(self,sys,ref):    
        sys = [sys]
        ref = [ref]
        ref = [[r] for r in ref]
        return - ROUGE.rouge_n_we(sys, ref, self.we, n=1, alpha=0)

class ROUGE_WE_2(Metrics):
    def __init__(self,we,device='cpu'):
        Metrics().__init__(self,we=we)
        self.we=we
        self.device = device
    def compute_score(self,sys,ref):    
        sys = [sys]
        ref = [ref]
        ref = [[r] for r in ref]
        return - ROUGE.rouge_n_we(sys, ref, self.we, n=2, alpha=0)

class ROUGE_1(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):   
        sys = [sys]
        ref = [ref]
        ref = [[r] for r in ref]
    
        return - ROUGE.rouge_n(sys, ref, n=1, alpha=0)

class ROUGE_2(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):       
        sys = [sys]
        ref = [ref]
        ref = [[r] for r in ref]
        return - ROUGE.rouge_n(sys, ref, n=2, alpha=0)

class ROUGE_L(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):   
        sys = [sys]
        ref = [ref]
        ref = [[r] for r in ref]
        return - ROUGE.rouge_l(sys, ref, alpha=0)

class JS_1(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):       
        sys = [sys]
        ref = [ref]
        ref = [[r] for r in ref]
        return JS_eval.JS_eval(sys, ref, n=1)

class JS_2(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):       
        sys = [sys]
        ref = [ref]
        ref = [[r] for r in ref]
        return JS_eval.JS_eval(sys, ref, n=2)

# def S3_pyr(self, sys, ref):     
#     sys = [sys]
#     ref = [ref]
#     ref = [[r] for r in ref]
#     return S3.S3(ref, sys, self.we, self.s3_model_folders)[0]

# def S3_resp(self, sys, ref): 
#     sys = [sys]
#     ref = [ref]
#     ref = [[r] for r in ref]
#     return S3.S3(ref, sys, self.we, self.s3_model_folders)[1]

class Chrfpp(Metrics):
    def __init__(self,device='cpu'):
        Metrics.__init__(self)
        self.device = device
    def compute_score(self,sys,ref): 
#        ref = [ref]
        #score = sacrebleu.sentence_chrf(sys, ref, order=self.ncorder, beta=self.beta)
        score = torchmetrics.CHRFScore(n_char_order = self.ncorder,beta=self.beta)([sys],[[ref]])
        return - score.item()

class BLEU(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):
        # ref = [ref]
        # score = sacrebleu.sentence_bleu(sys, ref, smooth_method=self.sent_smooth_method, \
        #      smooth_value=self.sent_smooth_value, use_effective_order=self.sent_use_effective_order)
        score = torchmetrics.BLEUScore()([sys],[[ref]])
        return - score.item()

class BERTScore(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):    
        ref = [ref]
#       sys, ref = [sys], [ref]
#       return BERT.BERTScore(sys, ref)
        return -moverscore_d.BERTScore([sys] * len(ref), ref).mean().item()

class MoverScore(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
    def compute_score(self,sys,ref):    
#       score = .0
#       for r in ref:
#           score += BERT.MoverScore([sys], [r])
#       return score / len(ref)
        ref = [ref]
        idf_dict_ref = defaultdict(lambda: 1.)
        idf_dict_hyp = defaultdict(lambda: 1.)
        return - np.mean(moverscore_d.word_mover_score(ref, [sys] * len(ref), idf_dict_ref, idf_dict_hyp,device=self.device))

# def BaryScore_W(self,sys,ref):
#     sys, ref = [sys], [ref]
#     return bary_score.BaryScore(sys,ref)['baryscore_W'][0]

# def BaryScore_SD_10(self,sys,ref):
#     sys, ref = [sys], [ref]
#     return bary_score.BaryScore(sys,ref)['baryscore_SD_10'][0]

# def BaryScore_SD_1(self,sys,ref):
#     sys, ref = [sys], [ref]
#     return bary_score.BaryScore(sys,ref)['baryscore_SD_1'][0]
    
# def BaryScore_SD_5(self,sys,ref):
#     sys, ref = [sys], [ref]
#     return bary_score.BaryScore(sys,ref)['baryscore_SD_5'][0]

class BaryScore_all(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
        self.baryscore_metric_call = BaryScoreMetric(model_name='distilbert-base-uncased',use_idfs=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        self.baryscore_metric_call.prepare_idfs(ref, sys)
        final_preds = self.baryscore_metric_call.evaluate_batch(ref, sys)
        return final_preds

class DepthScore(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.depthscore_metric_call = DepthScoreMetric(model_name='distilbert-base-uncased', layers_to_consider=4,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        # return depth_score.DepthScore(sys,ref)['depth_score'][0]

        final_preds = self.depthscore_metric_call.evaluate_batch(ref, sys)
        return final_preds['depth_score'][0]

class Infolm_kl(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'kl'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['kl'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['kl'].evaluate_batch(ref, sys)
        return - final_preds['kl'][0]
        # return - infolm.Infolm(sys,ref,'kl',device=self.device)['kl'][0]
class Infolm_alpha(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'alpha'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['alpha'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['alpha'].evaluate_batch(ref, sys)
        return - final_preds['alpha'][0]
class Infolm_renyi(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'renyi'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['renyi'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['renyi'].evaluate_batch(ref, sys)
        return - final_preds['renyi'][0]
class Infolm_beta(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'beta'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['beta'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['beta'].evaluate_batch(ref, sys)
        return - final_preds['beta'][0]
class Infolm_ab(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'ab'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):      
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['ab'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['ab'].evaluate_batch(ref, sys)
        return - final_preds['ab'][0]
class Infolm_l1(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'l1'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['l1'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['l1'].evaluate_batch(ref, sys)
        return - final_preds['l1'][0]
class Infolm_l2(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'l2'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['l2'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['l2'].evaluate_batch(ref, sys)
        return - final_preds['l2'][0]
class Infolm_linf(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'linf'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['linf'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['linf'].evaluate_batch(ref, sys)
        return - final_preds['linf'][0]
class Infolm_fisher_rao(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self,device=device)
        self.device = device
        self.meas = 'fisher_rao'
        self.infolm_metric={}
        self.infolm_metric[self.meas] = InfoLM(model_name="distilbert-base-uncased",measure_to_use=self.meas, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False,device=self.device)
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        idf_ref, idf_hypot = self.infolm_metric['fisher_rao'].prepare_idfs(ref, sys)
        final_preds = self.infolm_metric['fisher_rao'].evaluate_batch(ref, sys)
        return - final_preds['fisher_rao'][0]

class CharErrorRate(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        return torchmetrics.CharErrorRate()(sys,ref).item()
class ExtendedEditDistance(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref):  
        sys, ref = [sys], [ref]
        return torchmetrics.ExtendedEditDistance()(sys,ref).item()
class MatchErrorRate(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref): 
        sys, ref = [sys], [ref]
        return torchmetrics.MatchErrorRate()(sys,ref).item()
# def Perplexity(sys,ref): # Doesn't work
#     sys, ref = [sys], [ref]
#     return torchmetrics.Perplexity()(sys,ref).item()
class TranslationEditRate(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref): 
        sys, ref = [sys], [ref]
        return torchmetrics.TranslationEditRate()(sys,ref).item()
class WordErrorRate(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref): 
        sys, ref = [sys], [ref]
        return torchmetrics.WordErrorRate()(sys,ref).item()
class WordInfoLost(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
    def compute_score(self,sys,ref): 
        sys, ref = [sys], [ref]
        return torchmetrics.WordInfoLost()(sys,ref).item()
# def WordInfoPreserved(sys,ref): # just the negative of WordInfoLost
#     sys, ref = [sys], [ref]
#     return torchmetrics.WordInfoPreserved()(sys,ref).item()
class Bleurt(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
        self.bleurt_ops = score.create_bleurt_ops()
    def compute_score(self,sys,ref): 
        references = tf.constant([ref])
        candidates = tf.constant([sys])
        bleurt_out = self.bleurt_ops(references=references, candidates=candidates)
        return - bleurt_out["predictions"].numpy()[0]

class Comet_wmt20_comet_da(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
        self.model_path_wmt20 = download_model("wmt20-comet-da",saving_directory="/gpfs/users/himmian/.cache/torch/unbabel_comet/")
        self.model_wmt20 = load_from_checkpoint(self.model_path_wmt20)
    def compute_score(self,src,sys,ref):
        d = [{"src":src,"mt":sys,"ref":ref}]
        model_output = self.model_wmt20.predict(d, batch_size=8, gpus=0)
        return model_output[1]

class Comet_wmt21_comet_qe_mqm(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
        self.model_path_wmt21 = download_model("wmt21-comet-qe-mqm",saving_directory="/gpfs/users/himmian/.cache/torch/unbabel_comet/")
        self.model_wmt21 = load_from_checkpoint(self.model_path_wmt21)
    def compute_score(self,src,sys,ref):
        d = [{"src":src,"mt":sys,"ref":ref}]
        model_output = self.model_wmt21.predict(d, batch_size=8, gpus=0)
        return model_output[1]

class Comet_eamt22_cometinho_da(Metrics):
    def __init__(self,device='cpu'):
        Metrics().__init__(self)
        self.device = device
        self.model_path_eamt22 = download_model("eamt22-cometinho-da",saving_directory="/gpfs/users/himmian/.cache/torch/unbabel_comet/")
        self.model_eamt22 = load_from_checkpoint(self.model_path_eamt22)
    def compute_score(self,src,sys,ref):
        d = [{"src":src,"mt":sys,"ref":ref}]
        model_output = self.model_eamt22.predict(d, batch_size=8, gpus=0)
        return model_output[1]

