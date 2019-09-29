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
def transform_data_and_train(feats):
    #feature hasher for training
    fh_feats = DictVectorizer(sparse=True)
    X = fh_feats.fit_transform(feats)
    #fh_feats_prior = DictVectorizer(sparse=True)
    #X_prior = fh_feats_prior.fit_transform(feats_with_prior)
    #print "shape of feature matrix" + str(X.shape)
    data_quality_estimate = 0.8 #this variable estimates the probability of each weak supervised data being one
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
    st += "<runID>sarwar</runID>\n"
    st += "<retModel>indri</retModel>\n"
    st += "<query>\n"
    st += "<number>1</number>\n"
    st += "<text>"
    st += '#weight[feature]('
    return st

"""
Returns indri query suffix
"""
def get_query_suffix(prf=False):
    st = ''
    st += "</text>\n"
    st += "</query>\n"
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
def prepare_indri_query_top_k_features(features, weights, topk, prf=False):
    number_of_valid_features = 0
    st = get_query_prefix()
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
    st += get_query_suffix(prf)
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
    # loading training feature dataa
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = json.load(open(dir_path + "/../../../feats_all_ng_dep_prior.json"))
    # allfeats will hold all the features in the training data and their weights

    allfeats = []
    print len(data.keys())
    #loading all the features
    for item in data.keys():
        feats = {}
        for feature in data[item]:
            feats[feature] = 1
        allfeats.append(feats)

    #prior feats will contain all the features in the training dataset
    prior_feats = {}
    for item in data.keys():
        prior_feats = merge_two_dicts(prior_feats, data[item])

    print 'length of the feature set ' + str(len(prior_feats))

    allfeats.append(prior_feats)
    features, weights = transform_data_and_train(allfeats)
    name_set = load_gold()
    id_dict, train_id_dict = load_dictionary()
    query_terms = np.arange(1, 1002, 100)

    terms = []
    p10 = []
    p20 = []
    p30 = []
    num_civilian = []

    for number_of_query_terms in query_terms:
        file = open("indri_lr_query_file.xml", "w+")
        file.write(prepare_indri_query_top_k_features(features, weights, topk=number_of_query_terms))
        file.close()

        print "now querying"
        cmd =  "IndriRunQuery indri_lr_query_file.xml"
        output = sp.check_output(cmd.split())
        #print output
        precs10, precs20, precs30, number_of_civilians_found, name_of_civilians = get_metric_values(output.split("\n"), id_dict, name_set, train_id_dict)
        #results.append((number_of_query_terms, precs10, precs20, precs30, number_of_civilians_found))
        terms.append(number_of_query_terms)
        p10.append(precs10)
        p20.append(precs20)
        p30.append(precs30)
        num_civilian.append(number_of_civilians_found)
        # file = open("2.xml.run", 'w+')
        # for i in np.arange(30):
        #     file.write(output)
        # file.close()
        #
        # cmd = "mv 2.xml.run /home/smsarwar/PycharmProjects/civilian_killing/data/runs/iterative_lr_ir_run_dir/"
        # output = sp.check_output(cmd.split())
        # #print "querying done"
        # #print output
        #
        # cmd = "python /home/smsarwar/PycharmProjects/civilian_killing/src/eval/entity_level_evaluation.py"
        # output = sp.check_output(cmd.split())
        # print "evaluation done"
        # print output
    print p10
    print p20
    print p30
    print num_civilian

    f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].plot(terms, p10)
    # axarr[0, 0].set_title('Precision@10')
    # axarr[0, 1].scatter(terms, p20)
    # axarr[0, 1].set_title('Precision@20')
    # axarr[1, 0].plot(terms, p30)
    # axarr[1, 0].set_title('Precision@30')
    # axarr[1, 1].scatter(terms, number_of_civilians_found)
    # axarr[1, 1].set_title('\#civilians (top-1000)')
    # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    #print terms
    axarr[0, 0].plot(terms, p10, marker='o', markerfacecolor='black', markersize=6, color='darkgray', linewidth=2,
             label='P@10', linestyle='-.')
    axarr[0, 0].set_title('Precision@10')

    axarr[0, 1].plot(terms, p20, marker='v', markerfacecolor='blue', markersize=6, color='gray', linewidth=2,
             label='P@20', linestyle='--')
    axarr[0, 1].set_title('Precision@20')

    axarr[1, 0].plot(terms, p30, marker='x', markerfacecolor='red', markersize=6, color='black', linewidth=2,
             label='P@30', linestyle=':')
    axarr[1, 0].set_title('Precision@30')

    axarr[1, 1].plot(terms, num_civilian, marker='s', markerfacecolor='red', markersize=6, color='black', linewidth=2,
                     label='P@30', linestyle=':')
    axarr[1, 1].set_title('Relevant entities')

    f.text(0.5, 0.005 , 'Number of Features in Query', ha='center', fontsize=10)

    #plt.xlabel('Number of Examples in Query')
    # plt.ylabel('Precision@100')
    # plt.legend()
    # plt.savefig("p@100")
    # plt.show()
    # f.subplots_adjust(hspace=0.3)
    plt.show()

if __name__ == '__main__':
    main()
