from __future__ import division
from nltk import ngrams
from collections import defaultdict
import ujson as json
from sklearn.feature_extraction import FeatureHasher
import argparse
from scipy import sparse, io
import nltk
import socket
from sklearn.feature_extraction import DictVectorizer
#nltk.download()
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
stopwords.add('DT')
stopwords.add('CD')
import sys
import os
nltk.data.path.append(os.getcwd() + "/../downloads/nltk_data/")
sys.path.append(os.getcwd() + "/src/")
#print sys.path
from models.stanford_corenlp_pywrapper.stanford_corenlp_pywrapper.sockwrap import CoreNLP
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from collections import OrderedDict

#use https://github.com/brendano/stanford_corenlp_pywrapper
#cc=CoreNLP(annotators="tokenize,ssplit,pos,depparse")

#example to manually change path of corenlp 
#cc=CoreNLP(annotators="tokenize,ssplit,pos,depparse", corenlp_jars=["/Users/KatieKeith/stanford-corenlp-full-2016-10-31/*"])
#cc=CoreNLP(annotators="tokenize,ssplit,pos,depparse", cor  enlp_jars=["/home/smsarwar/work/coreNLPServer/stanford-corenlp-python/stanford-corenlp-full-2014-08-27/*"])

#Loading word embedding



#--------------BELOW IS FOR N-GRAMS---------
def get_all_ngrams(tok_idxs):
    ng = []
    for i in [1, 2, 3]:
        for n in ngrams(tok_idxs, i): ng.append(n)
    return ng

def a1(ng, tokens):
    if len(ng) == 1: return tokens[ng[0]]
    else:
        s = ''
        for i in range(len(ng)-1):
            s+= tokens[ng[i]]+u','
        s+=tokens[ng[-1]]
        return s

def a2(ng, tokens, pos_tags):
    if len(ng) == 1: return tokens[ng[0]]+','+pos_tags[ng[0]]
    else:
        s = ''
        for i in range(len(ng)-1):
            s+= tokens[ng[i]]+','+pos_tags[ng[i]]+','
        s+=tokens[ng[-1]]+','+pos_tags[ng[-1]]
        return s

def a3(ng, tokens, targ_idxs):
    result = []
    for t in targ_idxs:
        if len(ng) == 1: 
            result.append(tokens[ng[0]]+'_'+str(ng[0]-t))
        else:
            s = ''
            for i in range(len(ng)-1):
                s+= tokens[ng[i]]+'_'+str(ng[i]-t)+','
            s+=tokens[ng[-1]]+'_'+str(ng[-1]-t)
            result.append(s)
    return result

def get_sides_targ(tokens, targ_idxs):
    #get two on either side of target or close enough
    result = []
    for t in targ_idxs:
        start = t - 2
        end = t + 2
        if start < 0: start = 0 
        if end >= len(tokens): end = len(tokens) - 1
        result.append((start, end))
    return result 

def a4(sides_targ, pos_tags):
    s = ''
    for i in range(sides_targ[0], sides_targ[1]+1):
        s+=pos_tags[i]+','
    return s.strip(',')

def a5(sides_targ, tokens, pos_tags):
    s = ''
    for i in range(sides_targ[0], sides_targ[1]+1):
        s+=tokens[i]+','+pos_tags[i]+','
    return s.strip(',')


#--------------BELOW IS FOR DEPENDENCIES ------------
def get_edges_dir(deps):
    #takes core nlp deps and returns edges dict and direc 
    edges = defaultdict(set)
    direc = {}
    deptups = {}
    for dep in deps: 
        if dep[1]==-1 or dep[2]==-1: continue 
        edges[dep[1]].add(dep[2])
        edges[dep[2]].add(dep[1])   
        direc[(dep[1], dep[2])] = u'>'+dep[0]
        direc[(dep[2], dep[1])] = u'<'+dep[0]
    return edges, direc

def paths(node, edges, visited, path=(), length=1):
    #gets all the paths of specified length 
    #node: you will have to add the node you start with to this
    #reuturns tuples of idexes 
    if length==1:
        for neigh in edges[node]:
            if neigh in visited: continue
            yield path+(node, neigh)
    else:
        visited.add(node)
        for neigh in edges[node]:
            if neigh in visited: continue 
            for result in paths(neigh, edges, visited, path, length=length-1):
                yield result

#b1-b3
def get_paths_incl_targ(edges, targ_idxs):
    #gets all the paths that include one of the targ indexs 
    start_nodes =  get_start_nodes(edges, targ_idxs)
    allpaths = [] #all paths with the target
    #length denotes the number of edges between 
    for strt in start_nodes:
        for p in paths(strt, edges, set(), (), length=2):
            path = (strt, )+p
            for targ in targ_idxs: 
                if targ in path: 
                    allpaths.append(path)
    return allpaths

def b1(path, tokens, direc, pos_tags):
    #changes the output from paths() into unicode 
    s = u''
    for i in range(len(path)-1):
        s+=tokens[path[i]]+','+pos_tags[path[i]]+','+direc[(path[i], path[i+1])]+','
    s+=tokens[path[-1]]+','+pos_tags[path[-1]]
    return s

def b2(path, tokens, direc):
#changes the output from paths() into unicode 
    s = u''
    for i in range(len(path)-1):
        s+=tokens[path[i]]+u','+direc[(path[i], path[i+1])]+u','
    s+=tokens[path[-1]]
    return s

def b3(path, tokens, pos_tags):
    #changes the output from paths() into unicode 
    s = u''
    for i in range(len(path)-1):
        s+=tokens[path[i]]+u','+pos_tags[path[i]]+u','
    s+=tokens[path[-1]]+u','+pos_tags[path[-1]]
    return s

#b4 (will need to send thru b1)
def get_len_2(direc):
    return direc.keys()

def get_start_nodes(edges, targ_idxs):
    #this is needed to every starting 3-length dep that will have TARGET
    start_nodes = set()
    for t in targ_idxs:
        start_nodes.add(t)
        neighs = edges[t]
        start_nodes = start_nodes.union(neighs)
        for n in neighs: 
            start_nodes = start_nodes.union(edges[n])
    return start_nodes

def extr_all_feats(filename, output_file, hasNgrams=True, hasDeps=True):
    allfeats = []
    doc_count = 0 
    zero_feats = 0
    with open(filename, 'r') as r:
        for line in r:
            doc_count += 1
            sent = json.loads(line)['sent_alter']
            #sometimes TARGET and PERSON gets weird merge with other characters 
            sent = sent.replace('TARGET', ' TARGET ')
            sent = sent.replace('PERSON', ' PERSON ')
            assert type(sent) == unicode 
            d = cc.parse_doc(sent)
            feats = defaultdict(float)

            for s in d['sentences']:
                SYMBOLS = set("TARGET PERSON".split())
                tokens=[w.lower() if w not in SYMBOLS else w for w in s['tokens']]
                if 'TARGET' not in tokens: continue
                assert u'TARGET' in tokens 
                deps = s['deps_cc']
                pos_tags = s['pos']
                targ_idxs =[i for i, x in enumerate(tokens) if x == 'TARGET'] #there can be multiple TARGETS in a sentence
                tok_idxs = [i for i in range(len(tokens))]

                #-----EXTRACT FEATURES-------
                if hasNgrams:
                    for ng in get_all_ngrams(tok_idxs):
                        feats[a1(ng, tokens)] += 1.0
                        feats[a2(ng, tokens, pos_tags)] += 1.0
                        for f in a3(ng, tokens, targ_idxs): #multiple for multiple TARGET indexs
                            feats[f] += 1.0
                    for sides_targ in get_sides_targ(tokens, targ_idxs):
                        feats[a4(sides_targ, pos_tags)] += 1.0
                        feats[a5(sides_targ, tokens, pos_tags)] += 1.0
                
                if hasDeps:
                    edges, direc = get_edges_dir(deps)
                    #getting b1 thru b3 feats
                    for path in get_paths_incl_targ(edges, targ_idxs):
                        feats[b1(path, tokens, direc, pos_tags)] += 1.0
                        feats[b2(path, tokens, direc)] += 1.0
                        feats[b3(path, tokens, pos_tags)] += 1.0

                    #getting b4 feats
                    for path in get_len_2(direc):
                        feats[b1(path, tokens, direc, pos_tags)] += 1.0
            if len(feats) == 0: zero_feats+= 1
            allfeats.append(feats)
    assert len(allfeats) == doc_count
    print "READ {0} DOCS FROM FILE {1}".format(doc_count, filename)
    print "NUM DOCS WITH ZERO FEATS=",zero_feats
    output_file = 'feats_'+output_file+'.json'
    w = open(output_file, 'w')
    json.dump(allfeats, w)
    print 'wrote allfeats to ', output_file
    return allfeats


"""
This function scores all the features in a dictionary. Features come as keys and this function 
sets the values of those features using user feature feedback. 
"""
def score_feature_dictionary(data, user_features_expanded, model):
    for feature in data.keys():
        #features are splitted by comma
        feature_splitted = feature.split(',')
        #all features are preceeeded by u
        prefix = "u'"
        feature_splitted_cleaned = []
        for token in feature_splitted:
            if token.startswith(prefix):
                token = token[len(prefix):]
            feature_splitted_cleaned.append(token)

        #only some features are valid
        feature_splitted_valid = [token for token in feature_splitted_cleaned if token in model.wv and token not in stopwords]
        user_feature_scores = np.zeros(len(user_features_expanded))
        #print feature_splitted_valid
        for valid_token in feature_splitted_valid:
            user_feature_scores_temp = []
            for user_feature in user_features_expanded:
                user_feature_scores_temp.append(model.wv.similarity(w1=valid_token, w2=user_feature))
            user_feature_scores_temp = np.asarray(user_feature_scores_temp)
            user_feature_scores+=user_feature_scores_temp
        if len(feature_splitted_valid) > 0:
            user_feature_scores/=float(len(feature_splitted_valid))

        #print feature
        data[feature] = np.average(user_feature_scores)


"""
This function will extract features with prior information
"""
def extr_all_feats_with_prior(feature_file, model, prior_sentences):
    data = json.load(open(feature_file))
    # User features to be taken into consideration
    user_features = ['civilian', 'killed', 'kill', 'police', 'US', 'NYPD', 'United_States', 'victim', 'department',
                     'homicide', 'cop', 'police_officer']
    user_features_expanded = []
    allfeats = []
    allfeats_with_prior = []
    allfeats_with_prior_dict = {}
    # User feature expansion
    for feature in user_features:
        if feature in model.wv.vocab:
            for tuple in model.wv.most_similar(positive=feature, topn=5):
                user_features_expanded.append(tuple[0])
            user_features_expanded.append(feature)

    doc_count = 0
    for item in data:
        allfeats.append(item.copy())
        if doc_count % 1000 == 0:
            print doc_count
        if str(doc_count) in prior_sentences.keys():
            print 'found doc_count ' + str(doc_count)
            for feature in item.keys():
                feature_splitted = feature.split(',')
                prefix = "u'"
                feature_splitted_cleaned = []
                for token in feature_splitted:
                    if token.startswith(prefix):
                        token = token[len(prefix):]
                    feature_splitted_cleaned.append(token)

                feature_splitted_valid = [token for token in feature_splitted if
                                          token in model.wv and token not in stopwords]
                user_feature_scores = np.zeros(len(user_features_expanded))
                # print feature_splitted_valid
                for valid_token in feature_splitted_valid:
                    user_feature_scores_temp = []
                    for user_feature in user_features_expanded:
                        user_feature_scores_temp.append(model.wv.similarity(w1=valid_token, w2=user_feature))
                    user_feature_scores_temp = np.asarray(user_feature_scores_temp)
                    user_feature_scores += user_feature_scores_temp
                if len(feature_splitted_valid) > 0:
                    user_feature_scores /= float(len(feature_splitted_valid))
                    sum = np.sum(user_feature_scores)
                    if sum==0:
                        item[feature] = 0.1
                    else:
                        item[feature] = sum / len(user_feature_scores)
                else:
                    item[feature] = 0.1
            allfeats_with_prior.append(item)
            allfeats_with_prior_dict[doc_count] = item
        doc_count += 1

    allfeats.extend(allfeats_with_prior)
    assert len(allfeats) == doc_count + len(prior_sentences)
    return allfeats, allfeats_with_prior_dict


def go_feathash(feats, feats_with_prior, output_file, max_feats=125000, save=True):
    #feature hasher for training
    print "hasing to {0} features".format(max_feats)
    fh_feats = FeatureHasher(n_features=max_feats)
    X = fh_feats.transform(feats)
    assert X.shape[1] == max_feats

    print "hasing to {0} features".format(max_feats)
    fh_feats_prior = FeatureHasher(n_features=max_feats)
    X_prior = fh_feats_prior.transform(feats_with_prior)
    assert X_prior.shape[1] == max_feats

    #X.shape[0]==doc_count
    if save: 
        io.mmwrite(output_file, X)
        print "feat matrix saved to '{0}'.mtx".format(output_file)

        io.mmwrite(output_file + "_prior", X_prior)
        print "feat matrix saved to '{0}'.mtx".format(output_file + "_prior")

    else: return X

def go_feathash_mod(feats, output_file, save=True):
    #feature hasher for training
    fh_feats = DictVectorizer(sparse=True)
    X = fh_feats.fit_transform(feats)
    # fh_feats_prior = DictVectorizer(sparse=True)
    # X_prior = fh_feats_prior.fit_transform(feats_with_prior)
    print "shape of feature matrix" + str(X.shape)


    if save:
        io.mmwrite(output_file, X)
        print "feat matrix saved to '{0}'.mtx".format(output_file)
        # io.mmwrite(output_file + "_prior", X_prior)
        # print "feat matrix saved to '{0}'.mtx".format(output_file + "_prior")

    else: return X

if __name__ == "__main__":
    #print 'check'
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ngrams', action='store_true', help='includes A feats, ngram-like feats')
    arg_parser.add_argument('--deps', action='store_true', help='includes B feats, dep-like feats')
    arg_parser.add_argument('--prior', action='store_true', help='includes feature priors')
    arg_parser.add_argument('input', type=str, help='input file')
    args = arg_parser.parse_args()
    #print 'started'

    output_file = args.input.split('/')[-1].split('.')[0]
    if args.ngrams: output_file+= '_'+'ng'
    if args.deps: output_file += '_'+'dep'

    if socket.gethostname() == 'brooloo':
        if args.prior==False:
            cc = CoreNLP(annotators="tokenize,ssplit,pos,depparse",
                     corenlp_jars=["/home/smsarwar/work/stanford-corenlp-full-2016-10-31/*"])
        model = KeyedVectors.load_word2vec_format(
            '/home/smsarwar/PycharmProjects/GoogleNews-vectors-negative300.bin.gz',
            binary=True, limit=10000)
    else:
        if args.prior == False:
            cc = CoreNLP(annotators="tokenize,ssplit,pos,depparse",
                     corenlp_jars=[os.getcwd() + "/../../downloads/stanford-corenlp-full-2016-10-31/*"])
        model = KeyedVectors.load_word2vec_format(
            os.getcwd() + "/../../downloads/GoogleNews-vectors-negative300.bin.gz", binary=True, limit=100000)

    print("word embedding model loaded")

    prior_sentences = json.load(open("src/models/logreg/sentence_ids.json"))
    print "NGRAMS={0}, DEPS={1}".format(args.ngrams, args.deps)
    MAX_FEATS = 450000 #change based on the dimension you wish to feature hash to 
    if args.prior:
        #the following lines are for testing
        #prior_sentences[str(0)] = None
        #feature_file = os.getcwd() + "/sample_dep.json"
        #the following line is for original file
        feature_file = os.getcwd() + "/feats_all_ng_dep.json"
        #print prior_sentences
        allfeats, allfeats_with_prior_dict = extr_all_feats_with_prior(feature_file, model, prior_sentences)
        json.dump(allfeats_with_prior_dict, open("feats_all_ng_dep_prior.json", "w+"))

    else:
        allfeats = extr_all_feats(args.input, output_file, hasNgrams=args.ngrams, hasDeps=args.deps)

    go_feathash_mod(allfeats, output_file)






    













