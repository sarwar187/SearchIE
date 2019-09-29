from __future__ import division
import json
import nltk
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
stopwords.add('DT')
stopwords.add('CD')
import sys
import os
nltk.data.path.append(os.getcwd() + "/../downloads/nltk_data/")
sys.path.append(os.getcwd() + "/Code/")
print os.getcwd()
print sys.path.append(os.getcwd() + "../../src")
print sys.path
import string
import re
import subprocess as sp
import numpy as np
from informative_prior_logistic_regresssion_sw import InformativePriorLogisticRegressionWeight
from eval.entity_level_evaluation import load_gold
from eval.entity_level_evaluation import load_dictionary
import operator
import matplotlib.pyplot as plt
"""
Model to train 
"""
def train_informative_logreg(_X, _Q, w0, b0, C=5):
    """
    _X: size (Nmentions by Nfeat)
    _Q: vector length Nmentions. for each, P(z_i=1|x, y)
    model: for example a LogisticRegression object.
    """
    Nmention = _X.shape[0]
    X = np.vstack((_X, _X))
    Y = np.concatenate((np.ones(Nmention), np.zeros(Nmention)))
    weights = np.concatenate((_Q, 1 - _Q))
    #If we want to use prior we would have to comment the following line
    #Now we are using zero prior as it give the maximum gain
    w0 = np.zeros(len(w0))
    model = InformativePriorLogisticRegressionWeight(w0, b0, C)
    model = model.fit(X, Y, weights)
    return model

"""
training data transformation module. It trains the LR model and spits out two things:
the feature names and their weights. At the time of building a query we consider these
features and their weights. 
"""
def transform_data_and_train(feats, data_quality):
    #feature hasher for training
    fh_feats = DictVectorizer(sparse=True)
    X = fh_feats.fit_transform(feats)
    #fh_feats_prior = DictVectorizer(sparse=True)
    #X_prior = fh_feats_prior.fit_transform(feats_with_prior)
    #print "shape of feature matrix" + str(X.shape)
    data_quality_estimate = data_quality #this variable estimates the probability of each weak supervised data being one
    y = np.ones(X.shape[0])
    X =  X.toarray()
    train = X.shape[0]-1
    model = train_informative_logreg(X[0:train, :], np.ones(train) * data_quality_estimate, X[train, :], 0)
    return fh_feats.get_feature_names(), model.get_params()[0]

"""
Returns indri query prefix
"""
def get_query_prefix():
    st = ''
    st += "<parameters>\n"
    st += "<index>/home/smsarwar/PycharmProjects/civilian_killing/data/corpus_index_feature</index>\n"
    st += "<count>1000</count>\n"
    st += "<trecFormat>true</trecFormat>\n"
    st += "<runID>lr_ir</runID>\n"
    st += "<retModel>indri</retModel>\n"
    return st

def get_query_string(query_number):
    st = ''
    st += "<query>\n"
    st += "<number>" + str(query_number) + "</number>\n"
    st += "<text>"
    st += '#weight[feature]('
    return st

"""
Returns indri query suffix
"""
def get_query_suffix(prf=False):
    st = ''
    #st += "</text>\n"
    #st += "</query>\n"
    if prf==True:
        st += "<fbDocs>10</fbDocs>\n"
        st+= "<fbTerms>100</fbTerms>\n"
        st+= "<fbMu>0.5</fbMu>\n"
        st+= "<fbOrigWeight>0.5</fbOrigWeight>\n"
    st += "</parameters>\n"
    return st

"""
Returns indri query given features and their weight vectors. 
top-k specifies the number of features to take into account in the query. 
"""
def prepare_indri_query_top_k_features(features, weights, query_number, topk, prf=False):
    number_of_valid_features = 0
    st = get_query_string(query_number)
    feature_tuples = []
    for i in np.arange(len(features)):
        feature_tuples.append((features[i], weights[i], abs(weights[i])))

    sorted_by_second = sorted(feature_tuples, key=lambda tup: tup[1], reverse=True)

    for tuple in sorted_by_second[0:topk]:
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        feature_string = tuple[0]
        feature_string = regex.sub(' ', feature_string)
        feature_string = feature_string.strip()
        #print feature_string
        if feature_string:
            feature_splitted = feature_string.split(" ")
            feature_string_final = ' '.join(feature_splitted)
            weight = tuple[1]
            weight = 1
            st += str(weight) + " #10(" + feature_string_final + ") "
            number_of_valid_features+= 1
    print 'number of valid features ' + str(number_of_valid_features)
    st = st.strip()
    st += ")"
    st += "</text>\n"
    st += "</query>\n"
    #st += get_query_suffix(prf)
    return st


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


#1 Q0 1498571_322_0 1 -6.30739 IR
def get_metric_values(results, id_dict, name_set, train_id_dictionary):
    name_dict = {}
    results = results[0:1000]
    for result in results:
        line = result
        #print line
        line_splitted = line.split()
        name_list = id_dict[line_splitted[2].strip()]
        for name in name_list:
            if name in name_dict:
                name_dict[name] = name_dict[name] + 1
            else:
                name_dict[name] = 1
    sorted_data = sorted(name_dict.items(), key=operator.itemgetter(1), reverse=True)
    precs=[]
    recs=[]
    set_of_retrieved_names = set()
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for e,p in sorted_data:
        if e in name_set and e not in train_id_dictionary:
            set_of_retrieved_names.add(e)
            tp += 1
            fn -= 1
        else:
            fp += 1
        precs.append(tp/(tp+fp))
        recs.append(1)
    return precs[10], precs[20], precs[30], float(len(set_of_retrieved_names)), set_of_retrieved_names


def main():
    # loading training feature data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = json.load(open(dir_path + "/../../../feats_all_ng_dep_prior.json"))
    output_file = open("weak_supervision_output_file.tsv", "w")
    query_folder = "/home/smsarwar/PycharmProjects/civilian_killing/data/query_configurations/iterative_query_lr_dir"
    # allfeats will hold all the features in the training data and their weights
    onlyfiles = [os.path.join(query_folder, file) for file in os.listdir(query_folder) if os.path.isfile(os.path.join(query_folder, file))]
    # print onlyfiles
    #results = {}
    number_of_files = 50
    #query_terms = np.arange(10, 41, 10)
    data_qualities = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    res = np.zeros((4, len(data_qualities)))

    for idx, file in enumerate(onlyfiles[:number_of_files]):
        #results[file] = {}
        print file
        lines = open(file).readlines()
        lines = [line.strip() for line in lines]
        line = lines[20]
        sentence_ids = line.strip().split()
        allfeats = []
        #loading all the features
        for item in data.keys():
            if item in sentence_ids:
                feats = {}
                for feature in data[item]:
                    feats[feature] = 1
                allfeats.append(feats)

        #prior feats will contain all the features in the training dataset
        prior_feats = {}
        for item in data.keys():
            prior_feats = merge_two_dicts(prior_feats, data[item])

        terms = []
        p10 = []
        p20 = []
        p30 = []
        num_civilian = []

        allfeats.append(prior_feats)
        name_set = load_gold()
        id_dict, train_id_dict = load_dictionary()
        #query_terms = np.arange(1, 1002, 100)
        #query_terms = np.arange(1, 1002, 100)
        query_number = 0
        for data_quality in data_qualities:
            features, weights = transform_data_and_train(allfeats, data_quality)
            number_of_query_terms = min(len(features), 200)
            file_temp = open("indri_lr_query_file.xml", "w")
            file_temp.write(get_query_prefix())
            query = prepare_indri_query_top_k_features(features, weights, query_number, topk=number_of_query_terms, prf=False)
            file_temp.write(query + '\n')
            query_number+=1
            file_temp.write(get_query_suffix())
            file_temp.close()
            print "now querying"
            cmd =  "IndriRunQuery indri_lr_query_file.xml"
            output = sp.check_output(cmd.split())
            # #print output
            precs10, precs20, precs30, number_of_civilians_found, name_of_civilians = get_metric_values(output.split("\n"), id_dict, name_set, train_id_dict)
            # #results.append((number_of_query_terms, precs10, precs20, precs30, number_of_civilians_found))
            terms.append(number_of_query_terms)
            p10.append(precs10)
            p20.append(precs20)
            p30.append(precs30)
            num_civilian.append(number_of_civilians_found)

        res[0]+= np.asarray(p10)
        res[1]+= np.asarray(p20)
        res[2]+= np.asarray(p30)
        res[3]+= np.asarray(num_civilian)

        # print res[0]/(idx+1)
        # print res[1]/(idx+1)
        # print res[2]/(idx+1)
        # print res[3]/(idx+1)

        res_P10 = ''
        res_P20 = ''
        res_P30 = ''
        res_num_civilian = ''

        for index in np.arange(len(data_qualities)):
            res_P10 += str(res[0][index]/(idx+1)) + "\t"
            res_P20 += str(res[1][index]/(idx+1)) + "\t"
            res_P30 += str(res[2][index]/(idx+1)) + "\t"
            res_num_civilian += str(res[3][index]/(idx+1)) + "\t"

        print data_qualities
        print res_P10.strip()
        print res_P20.strip()
        print res_P30.strip()
        print res_num_civilian.strip()

        output_file.write(str(data_qualities) + "\n")
        output_file.write(res_P10.strip() + "\n")
        output_file.write(res_P20.strip() + "\n")
        output_file.write(res_P30.strip() + "\n")
        output_file.write(res_num_civilian.strip() + "\n")
        output_file.write("-----------------------------------------------------\n")

    output_file.close()

if __name__ == '__main__':
    main()
