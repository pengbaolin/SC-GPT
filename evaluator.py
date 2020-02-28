import numpy as np
import os
import operator
from math import sqrt
import random
from ast import literal_eval
import pickle
from copy import deepcopy
import argparse

from utils.loader.DataReader import *
from utils.loader.GentScorer import *
from nltk.tokenize import sent_tokenize, word_tokenize

random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)
np.set_printoptions(precision=4)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None, type=str, required=True, help="Please specify a domain")
    parser.add_argument("--target_file", default=None, type=str, required=True, help="Please specify the result file")
    args = parser.parse_args()

    domain = args.domain
    target_file = args.target_file

    train       = f'data/{domain}/train.json'
    valid       = f'data/{domain}/train.json'
    test        = f'data/{domain}/test.json'
    vocab       = 'utils/resource/vocab'
    percentage  = 100
    topk        = 5
    detectpairs = 'utils/resource/detect.pair'

    reader = DataReader(random_seed, domain, 'dt', vocab, train, valid, test, 100 , 0, lexCutoff=4)
    gentscorer = GentScorer(detectpairs)

    da2sents = {}
    templates = reader.readall(mode='train')+\
                reader.readall(mode='valid')
    for a,sv,s,v,sents,dact, base in templates:
        key = (tuple(a),tuple(sv))
        if key in da2sents.keys():
            da2sents[key].extend(sents)
            da2sents[key] = list(set(da2sents[key]))
        else:
            da2sents[key] = sents

    results_from_gpt = json.load(open(target_file))
    idx = 0
    parallel_corpus, hdc_corpus = [], []
    gencnts, refcnts = [0.0,0.0,0.0],[0.0,0.0,0.0]
    while True:
        # read data point
        data = reader.read(mode='test',batch=1)
        if data==None:
            break
        a,sv,s,v,sents,dact,bases,cutoff_b, cutoff_f = data
        
        # remove batch dimension
        a,sv,s,v = a[0],sv[0],s[0],v[0]
        sents,dact,bases = sents[0],dact[0],bases[0]
        
        # score DA similarity between testing example and train+valid set
        template_ranks = []
        for da_t,sents_t in da2sents.items():
            a_t,sv_t = [set(x) for x in da_t]
            score =float(len(a_t.intersection(set(a)))+\
                    len(sv_t.intersection(set(sv))))/\
                    sqrt(len(a_t)+len(sv_t))/sqrt(len(a)+len(sv))
            template_ranks.append([score,sents_t])
        # rank templates
        template_ranks = sorted(template_ranks,key=operator.itemgetter(0))
        # gens = deepcopy(template_ranks[-1][1])
        # score= template_ranks[-1][0]
        score = 1
        
        gen_strs = results_from_gpt[idx]
        gen_strs_single = []
        gen_strs_ = []
        for gen_str in gen_strs:
            cl_idx = gen_str.find('<|endoftext|>')
            gen_str = gen_str[:cl_idx].strip().lower()
            gen_str = ' '.join(word_tokenize(gen_str))
            gen_str.replace('-s','')
            gen_str = gen_str.replace('watts','watt -s').replace('televisions','television -s').replace('ports', 'port -s').replace('includes', 'include -s').replace('restaurants','restaurant -s').replace('kids','kid -s').replace('childs','child -s').replace('prices','price -s').replace('range','range -s').\
                replace('laptops','laptop -s').replace('familys','family -s').replace('specifications','specification -s').replace('ratings','rating -s').replace('products','product -s').\
                    replace('constraints','constraint -s').replace('drives','drive -s').replace('dimensions','dimension -s')
            gen_strs_single.append(gen_str)
            gen_strs_.append(gen_str)                    
            

        

        gens = gen_strs_
        idx += 1
        topk = 1
        felements = [reader.cardinality[x+reader.dfs[1]]\
                for x in sv]
        gens_with_penalty = []

        for i in range(len(gens)):
            # score slot error rate
            delexed = reader.delexicalise(gens[i], dact)
            cnt, total, caty = gentscorer.scoreERR(a,felements, delexed)
            gens[i] = reader.lexicalise(gens[i],dact)
            gens_with_penalty.append((caty, len(gens[i].split()), gens[i]))
        
        gens_with_penalty = sorted(gens_with_penalty, key=lambda k:k[0])[:topk]

        gens = [g[2] for g in gens_with_penalty][:1]

        for i in range(len(gens)):
            # score slot error rate
            delexed = reader.delexicalise(gens[i], dact)
            cnt, total, caty = gentscorer.scoreERR(a,felements, delexed)
            gens[i] = reader.lexicalise(gens[i],dact)
            # accumulate slot error cnts
            gencnts[0]  += cnt
            gencnts[1]  += total
            gencnts[2]  += caty
        
        # compute gold standard slot error rate
        for sent in sents:
            # score slot error rate
            cnt, total, caty = gentscorer.scoreERR(a,felements,
                    reader.delexicalise(sent,dact))
            # accumulate slot error cnts
            refcnts[0]  += cnt
            refcnts[1]  += total
            refcnts[2]  += caty

        parallel_corpus.append([[g for g in gens], sents])
        hdc_corpus.append([bases[:1],sents])
    
    predicted_sentences = []
    
    for i in parallel_corpus:
        predicted_sentences.append(i[0][0])

    bleuModel   = gentscorer.scoreSBLEU(parallel_corpus)
    bleuHDC     = gentscorer.scoreSBLEU(hdc_corpus)
    print ('##############################################')
    print ('BLEU SCORE & SLOT ERROR on GENERATED SENTENCES')
    print ('##############################################')
    print ('Metric       :\tBLEU\tT.ERR\tA.ERR')
    print ('HDC          :\t%.4f\t%2.2f%%\t%2.2f%%'% (bleuHDC,0.0,0.0))
    print ('Ref          :\t%.4f\t%2.2f%%\t%2.2f%%'% (1.0,
            100*refcnts[1]/refcnts[0],100*refcnts[2]/refcnts[0]))
    print ('----------------------------------------------')
    print ('This Model   :\t%.4f\t%2.2f%%\t%2.2f%%'% (bleuModel,
            100*gencnts[1]/gencnts[0],100*gencnts[2]/gencnts[0]))
    
    print(f'FIELNAME: {target_file}, BLEU: {bleuModel}, ERR:{100*gencnts[1]/gencnts[0]}')


if __name__ == "__main__":
    main()