import numpy as np
import random
from decimal import Decimal
from scipy.stats import rankdata

facts = np.ones(300+1,dtype=Decimal) # CAUTION, precompute factorials, but set to a max of n=300
for i in range(1,len(facts)):
  facts[i] = facts[i-1]*Decimal(i)
facts

# k the fixed (partial) ranking
# n-k the rest, the unknown
# TODO precomput the next 3 functions in a table to speed up
def V(n,m):
  return Decimal(facts[m])/Decimal(facts[m-n])
def S(m,n): #show many ways of shuffling two piles of this lengths
  return Decimal(facts[n+m])/(Decimal(facts[n])*Decimal(facts[m]))
def perms_total(n,k):
  return Decimal(facts[n-k]) * S(k,n-k)


def perms_with_nan_before(n,k,p_rank):
  # given k partially ordered items
  # how many permutations of n items are consistent with the k
  # and place one of the unknown items BEFORE the item with p_rank partial rank
  acum = Decimal(0)
  for i in range(0,n-k): # aprate de ese q ya has cogido, cuanto mas coges para poner delante
    acum += V(i, n-k-1) * Decimal(i+1) * S(p_rank, i+1) * facts[(n-k-i-1)] * S(n-k-i-1, k-p_rank-1)
  return acum

def p_rank_to_mat(perm, return_ratios=True, return_num_comparisons=False):
  n = len(perm)
  p = np.zeros((n,n),dtype=Decimal)
  compared = np.zeros((n,n),dtype=Decimal)
  k = (~np.isnan(perm)).sum()
  for i in range(n):
    for j in range(i+1,n):
      if perm[i] < perm[j]:
        p[i,j] = perms_total(n,k)
        compared[i,j] += 1
        compared[j,i] += 1
      elif perm[j] < perm[i]:
        p[j,i]=perms_total(n,k)
        compared[i,j] += 1
        compared[j,i] += 1
      elif  np.isnan(perm[i]) and np.isnan(perm[j]):
        p[j,i] = perms_total(n,k)/2
        p[i,j] = perms_total(n,k)/2
      elif np.isnan(perm[j]) :
        p_rank = int(perm[i])
        p[j,i] = perms_with_nan_before(n,k,p_rank)# / perms_total(n,k)
        p[i,j] = perms_total(n,k) - p[j,i]
      else:
        p_rank = int(perm[j])
        p[j,i] = (perms_total(n,k)-perms_with_nan_before(n,k,p_rank)) #/ perms_total(n,k)
        p[i,j] = perms_total(n,k) - p[j,i]
  if return_ratios: 
    p = p/perms_total(n,k)
  if return_num_comparisons: return p, compared, perms_total(n,k)
  p=p.astype('float64')
  return p


# a = [np.nan, 4,3,4,np.nan,2]
def rank_data_nan(a):
  a = rankdata(a,method="average",nan_policy="omit")-1 # a = [np.nan,2.5,1,2.5,np.nan,0]
  # remove ties while saving a dict from the new_ranks to the old_ranks
  d = {} # new_d = {0:[0],1:[1,3],2:[2],3:[4],4:[5]}
  new_a = [] # new_a = [np.nan,2.5,1,np.nan,0]
  for i in range(len(a)):
    if a[i] not in new_a:
      new_a.append(a[i])
      d[len(new_a)-1] = [i]
    else:
      d[new_a.index(a[i])].append(i)
  new_a_ranked = np.argsort(rand_argsort(new_a)) # rank the non tied values
  a_ranked = np.zeros(len(a)) # reputting the ties in the right place
  for i in d:
    for j in d[i]:
      a_ranked[j] = new_a_ranked[i]
  return rankdata(a_ranked,method="average",nan_policy="omit")-1






def rand_argsort(arr):
  # if all nan: return np.random.permutation(range(len(arr)))
  if np.isnan(arr).all(): return np.random.permutation(range(len(arr)))
  # if np.all(np.isclose(arr, arr[0])): return np.random.permutation(range(len(arr)))
  #print(arr,np.argsort(arr))
  nan_indices = np.argwhere(np.isnan(arr)).reshape(1,-1)[0]
  #tied_indices: add a random num that does not change the rder
  unique_vals, counts = np.unique(arr, return_counts=True)
  # print(arr)
  # print(unique_vals, counts)
  if any(counts>1):
    #print(unique_vals, counts)
    #check if the non nan values are all the same
    if len([i for i in unique_vals if not np.isnan(i)])==1:
      smallest_diff = 1
    else:
      smallest_diff = np.min(np.diff(unique_vals[~np.isnan(unique_vals)]))
    ran = np.random.random(len(arr))*smallest_diff
    arr = arr + ran
  #nan: shuffle the 2 lists
  lis = []
  k = len(arr) - len(nan_indices)
  pile1, pile2 = list(nan_indices),list(np.argsort(arr)[:k])
  #print(pile1, pile2)
  for _ in range(len(arr)):#shuffle
    if len(pile1)==0: lis.append(pile2.pop())
    elif len(pile2)==0: lis.append(pile1.pop())
    elif np.random.random()>.5:   
      lis.append(pile2.pop())
    else:
      lis.append(pile1.pop()) 
  lis = lis[::-1]
  return lis

def partial_scores_to_ranking(scores):
  scores = np.array(scores)
  non_nan_scores = scores[~np.isnan(scores)]
  non_nan_ranks = rank_data_nan(non_nan_scores)
  # put the nan values in the same place as scores
  ranks = []
  for i in range(len(scores)):
    if np.isnan(scores[i]):
      ranks.append(np.nan)
    else:
      ranks.append(non_nan_ranks[0])
      non_nan_ranks = non_nan_ranks[1:]
  return np.array(ranks)

def borda_mat(p):
  return rank_data_nan(p.sum(axis=0))
