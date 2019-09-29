#test line
import Queue as Q
import json
import numpy as np
import subprocess as sp
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from random import shuffle
from collections import OrderedDict
import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')
project_directory = "/home/smsarwar/PycharmProjects/civilian_killing/"

def concatenate_train_test():
    train_file = open("../../data/json_corpus/train.json")
    test_file = open("../../data/json_corpus/test.json")
    corpus_file_json = open("../../data/json_corpus/all.json", "w")

    count = 0
    for line in train_file:
        data = json.loads(line)
        #data["docid"] = str(count) + "_" + data["docid"]
        data["type"] = "train"
        string_to_write = json.dumps(data)
        corpus_file_json.write(string_to_write + "\n")
        count+=1

    for line in test_file:
        data = json.loads(line)
        #data["docid"] = str(count) + "_" + data["docid"]
        data["type"] = "test"
        string_to_write = json.dumps(data)
        corpus_file_json.write(string_to_write + "\n")
        count+=1

    corpus_file_json.close()


"""
The query string would contain entity names


"""
def query_container(query_strings, field, keyword_query):
    """
    Given a query dictionary generate indri query xml
    :param query_dict: containing seeds
    :param field: text, name etc.
    :return: string that contains the query
    """
    fielded_query_string = ''
    st = ''
    count = 1
    for query_string in query_strings:
        #print 'index ' + str(index)
        query_string_splitted = query_string.split()
        for i in np.arange(len(query_string_splitted)):
            fielded_query_string += query_string_splitted[i].replace("'","") + "." + field + ' '

        st+= '<query>\n'
        st+= '<number>' + str(count) + '</number>\n'
        st+= '<text>' + fielded_query_string + keyword_query + '</text>\n'
        st+= '</query>\n'
        count+=1
    return st

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def query_container_for_lr(query_strings, training_data_dictionary):
    """
    Given a query dictionary generate indri query xml
    :param query_dict: containing seeds
    :param field: text, name etc.
    :return: string that contains the query
    """
    query_id_sets = []
    query_id_set = set()
    for query_string in query_strings:
        keys = training_data_dictionary[query_string]
        query_id_set = query_id_set.union(keys)
        query_id_sets.append(query_id_set)

    st = ''
    for query_id_set in query_id_sets:
        st_sentence_ids = ''
        for item in query_id_set:
            st_sentence_ids+=str(item) + " "
        st_sentence_ids.strip()
        st_sentence_ids+="\n"
        st+=st_sentence_ids

    #returns a string of the format "1 4 54\n 2 4 22"
    return st

def query_xml_container(query_list, retrieval_approach):
    st = ''
    st+= '<parameters>\n'
    st+= '<index>' + project_directory + 'data/corpus_index</index>\n'
    st+= '<count>1000</count>\n'
    st+= '<trecFormat>true</trecFormat>\n'
    st+= '<runID>IR</runID>\n'
    st+= '<retModel>indri</retModel>\n'
    for query in query_list:
        st+= query
    if retrieval_approach=='lm':
        st += '</parameters>\n'
    else:
        st+= '<fbDocs>10</fbDocs>\n'
        st+= '<fbTerms>20</fbTerms>\n'
        st+= '<fbMu>0.5</fbMu>\n'
        st+= '<fbOrigWeight>0.5</fbOrigWeight>\n'
        st+= '</parameters>\n'
    return st


def prepare_qrel(corpus_file_json, qrel_file):

    corpus_dictionary = {}
    for line in corpus_file_json:
        data = json.loads(line)
        corpus_dictionary.setdefault(data['docid'], data)
    #print 'corpus dictionary length ' + str(len(corpus_dictionary))
    query_ids = np.arange(1,31)
    for key in corpus_dictionary.keys():
        for query_id in query_ids:
            st = str(query_id) + " 0 " + key + " " + str(corpus_dictionary[key]['plabel']) + "\n"
            #print st
            qrel_file.write(st)
    qrel_file.close()




"""
This function is for creating an idex of the features as well as the documents. 
"""
def create_trec_data_with_feature_index():
    corpus_file_json = open("../../data/json_corpus/all.json")
    feature_file = open("../../feats_all_ng_dep.json")
    corpus_feature_file_trec = open("../../data/corpus/feature_corpus/feature_corpus_unique_docno.xml", "w")
    feature_data = json.load(feature_file)
    line_id = 0 #using line_id to map corpus to feature file
    count = 1
    for line in corpus_file_json:
        feature_of_line = feature_data[line_id]
        feature_string = ' '.join(feature_of_line.keys())
        feature_string = feature_string.strip()
        if line_id == 0:
            print feature_string
        data = json.loads(line)
        trec_doc = "<DOC>\n"
        #trec_doc += "<DOCNO>" + str(count) + "</DOCNO>\n"
        trec_doc += "<DOCNO>" + data["docid"] + "</DOCNO>\n"
        #trec_doc+= "<DOCID>" + data["docid"] + "</DOCID>\n"
        trec_doc += "<NAME>" + data["name"] + "</NAME>\n"
        trec_doc += "<TEXT>" + data["sent_org"] + "</TEXT>\n"
        trec_doc += "<FEATURE>" + feature_string + "</FEATURE>\n"
        trec_doc += "</DOC>\n"
        if count%10000 == 0:
            print "processed doc " + str(count)
        count+=1
        corpus_feature_file_trec.write(trec_doc)
        line_id+=1
    corpus_feature_file_trec.close()


"""
This function is for creating an idex of the features as well as the documents. 
"""
def create_trec_data_with_feature_index_unique_docno():
    corpus_file_json = open("../../data/json_corpus/all.json")

    doc_line_id_mapping = {}
    line_id = 0
    for line in corpus_file_json:
        data = json.loads(line)
        doc_line_id_mapping.setdefault(data["docid"],[])
        doc_line_id_mapping[data["docid"]].append(line_id)
        line_id+=1

    corpus_file_json = open("../../data/json_corpus/all.json")

    feature_file = open("../../feats_all_ng_dep.json")
    corpus_feature_file_trec = open("../../data/corpus/feature_corpus_unique_docno/feature_corpus.xml", "w")
    feature_data = json.load(feature_file)
    doc_id_used = set()

    line_id = 0 #using line_id to map corpus to feature file
    count = 1
    for line in corpus_file_json:

        data = json.loads(line)
        if data["docid"] in doc_id_used:
            print data["docid"]
            continue
        doc_id_used.add(data["docid"])
        line_ids = doc_line_id_mapping[data["docid"]]
        feature_string_combined = ""
        for line_id in line_ids:
            feature_of_line = feature_data[line_id]
            feature_string = ' '.join(feature_of_line.keys())
            feature_string = feature_string.strip()
            feature_string_combined+= feature_string + " "
        feature_string_combined = feature_string_combined.strip()
        if line_id == 0:
            print feature_string

        trec_doc = "<DOC>\n"
        #trec_doc += "<DOCNO>" + str(count) + "</DOCNO>\n"
        trec_doc += "<DOCNO>" + data["docid"] + "</DOCNO>\n"
        #trec_doc+= "<DOCID>" + data["docid"] + "</DOCID>\n"
        trec_doc += "<NAME>" + data["name"] + "</NAME>\n"
        trec_doc += "<TEXT>" + data["sent_org"] + "</TEXT>\n"
        trec_doc += "<FEATURE>" + feature_string_combined + "</FEATURE>\n"
        trec_doc += "</DOC>\n"
        if count%10000 == 0:
            print "processed doc " + str(count)
        count+=1
        corpus_feature_file_trec.write(trec_doc)
        line_id+=1
    corpus_feature_file_trec.close()



def create_trec_data():
    corpus_file_json = open("../../data/json_corpus/all.json")
    corpus_file_trec = open("../../data/corpus/text_corpus/corpus.xml", "w")
    training_dict = {}
    count = 1
    dictionary_length = 0
    line_id = 0

    for line in corpus_file_json:
        data = json.loads(line)
        trec_doc = "<DOC>\n"
        #trec_doc += "<DOCNO>" + str(count) + "</DOCNO>\n"
        trec_doc += "<DOCNO>" + data["docid"] + "</DOCNO>\n"
        #trec_doc+= "<DOCID>" + data["docid"] + "</DOCID>\n"
        trec_doc += "<NAME>" + data["name"] + "</NAME>\n"
        trec_doc += "<NAMES_ORG>" + ",".join(data["names_org"]) + "</NAMES_ORG>\n"
        trec_doc += "<SENTNAMES>" + ",".join(data["sentnames"]) + "</SENTNAMES>\n"
        trec_doc += "<DOWNLOADTIME>" + data["downloadtime"] + "</DOWNLOADTIME>\n"
        trec_doc += "<TEXT>" + data["sent_org"] + "</TEXT>\n"
        trec_doc += "<TEXT_ALT>" + data["sent_alter"] + "</TEXT_ALT>\n"
        trec_doc += "<TEXT_ORG>" + data["sent_org"] + "</TEXT_ORG>\n"
        trec_doc += "<PLABEL>" + str(data["plabel"]) + "</PLABEL>\n"
        trec_doc += "<TYPE>" + data["type"] + "</TYPE>\n"
        trec_doc += "</DOC>\n"
        print "processed doc " + str(count)

        if data["plabel"] == 1 and data["type"]=='train':
            #print 'found'
            training_dict.setdefault(data['name'], [])
            training_dict[data['name']].append(line_id)
            if len(training_dict) > dictionary_length and len(training_dict) <= 30:
                dictionary_length = len(training_dict)

        count+=1
        corpus_file_trec.write(trec_doc)
        line_id+=1

    print 'training dictionary size ' + str(len(training_dict))
    #retrieval_approaches = ["lm", "prf", "lr"]
    retrieval_approaches = ["lr"]

    query_term_list = list(training_dict.keys())[0:30]
    print len(query_term_list)

    for i in np.arange(50):
        shuffle(query_term_list)
        for retrieval_approach in retrieval_approaches:
            if retrieval_approach == 'lm':
                iterative_query_file_xml_indri_dir = "../../data/query_configurations/iterative_query_lm_dir/"
            elif retrieval_approach == "prf":
                iterative_query_file_xml_indri_dir = "../../data/query_configurations/iterative_query_prf_dir/"
            else:
                iterative_query_file_xml_indri_dir = "../../data/query_configurations/iterative_query_lr_dir/"

            #print 'opened file'
            iterative_train_file_xml_indri = open(iterative_query_file_xml_indri_dir + str(i+1) + ".xml", "w")

            if retrieval_approach == "lr":
                query_list_string = query_container_for_lr(query_term_list, training_data_dictionary=training_dict)
                #print query_list_string
                iterative_train_file_xml_indri.write(query_list_string)
                iterative_train_file_xml_indri.close()
                #print iterative_train_file_xml_indri
            else:
                query_list = query_container(query_term_list, "text", "civilians.text killed.text police.text")
                #query_list = query_container(query_term_list, "text", "")
                #print query_list
                iterative_train_file_xml_indri.write(query_xml_container(query_list, retrieval_approach))
                iterative_train_file_xml_indri.close()
            #print 'wrote file'
    corpus_file_trec.close()

# def main():
#     np.random.seed(0)
#     corpus_file_json = open("../../data/all.json")
#     iterative_qrel_file = open("../../data/runs/iterative_qrel", "w")
#     prepare_qrel(corpus_file_json, iterative_qrel_file)
#     print "qrel preparation done"
#     concatenate_train_test()
#     create_trec_data()
#     print "trec document created"
#     print "query files created"
#     cmd =  "IndriBuildIndex " + project_directory + "data/corpus_config.xml"
#     output = sp.check_output(cmd.split())
#     print "now indexing"
#     print output
#     print "indexing done"
                
#this main function is for creating feature index
def main():
    # np.random.seed(0)
    # create_trec_data_with_feature_index_unique_docno()
    # print "trec document created"
    # print '---------------------'
    # print "now indexing"
    cmd =  "IndriBuildIndex " + project_directory + "data/config/corpus_config_feature_unique_docno.xml"
    output = sp.check_output(cmd.split())
    print output
    print "indexing done"

if __name__ == '__main__':
    main()