import json
import subprocess as sp
import os
from src.models.logreg.runlr import *
sys.path.append("../../../src")
#sys.path.append("../../../src/models/logreg")
from models.logreg.runlr import *

corpus_file_json = open("/home/smsarwar/PycharmProjects/civilian_killing/data/json_corpus/all.json")
corpus_dictionary = {}

for line in corpus_file_json:
    data = json.loads(line)
    #print(list(data.keys()))
    #print (key)
    corpus_dictionary.setdefault(data['docid'] , data)


#print(len(corpus_dictionary))
#Queries the index with 'hello world' and returns the first 1000 results.
#results = index.query('victims civilians killed united states police officers Alton Sterling', results_requested=10)

#outputs = sp.check_output("IndriRunQuery query_configurations/query_xml_prf".split())
#outputs = sp.check_output("IndriRunQuery query_configurations/name_search_query".split())

retrieval_methods = ["lm", "prf"]
#retrieval_methods = ["lr"]

#retrieval_methods = ["single"]

data_mtx = io.mmread("../../../all_ng_dep.mtx")
data_mtx = data_mtx.tocsr()

for retrieval_method in retrieval_methods:
    if retrieval_method=='lm':
        path = "../../../data/query_configurations/iterative_query_lm_dir/"
        #path = "../../../data/query_configurations/iterative_query_lm_dir/"
        os.chdir(path)
        onlyfiles = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        for file in onlyfiles:
            cmd = "IndriRunQuery " + file
            outputs = sp.check_output(cmd.split())
            run_file = open("../../../data/runs/iterative_lm_run_dir/" + os.path.basename(file) + ".run", "w")
            run_file.write(outputs)
            run_file.close()
    elif retrieval_method=='prf':
        path = "../../../data/query_configurations/iterative_query_prf_dir/"
        os.chdir(path)
        onlyfiles = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        for file in onlyfiles:
            cmd = "IndriRunQuery " + file
            outputs = sp.check_output(cmd.split())
            run_file = open("../../../data/runs/iterative_prf_run_dir/" + os.path.basename(file) + ".run", "w")
            run_file.write(outputs)
            run_file.close()
    # else:
    #     path = "../../../data/query_configurations/iterative_query_lr_dir/"
    #     os.chdir(path)
    #     onlyfiles = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    #     count = 0
    #     for file in onlyfiles:
    #         run_file = open("../../../data/runs/iterative_lr_run_dir/" + os.path.basename(file) + ".run", "w")
    #         for line in open(file):
    #             sentence_ids = line.split()
    #             sentence_ids = [int(item) for item in sentence_ids]
    #             #corpus_file_json = open("../../../data/all.json")
    #             #data_mtx_file = open("../../../all_ng_dep.mtx")
    #             run_logistic_regression("../../../data/all.json", data_mtx, sentence_ids, run_file)
    #         run_file.close()
            # cmd = "IndriRunQuery " + file
            # outputs = sp.check_output(cmd.split())
            # run_file = open("../../../data/runs/iterative_lr_run_dir/" + os.path.basename(file) + ".run", "w")
            # run_file.write(outputs)
            # run_file.close()
        #outputs = sp.check_output("IndriRunQuery ../../query_configurations/query_xml".split())



    # outputs = sp.check_output("IndriRunQuery ../../query_configurations/iterative_query_lm".split())
    #
    # run_file = open("../../../data/runs/" + retrieval_method + ".run", "w")
    #
    # outputs = outputs.decode("utf-8")
    #
    # #print(type(outputs))
    # outputs_splitted = outputs.split("\n")
    # outputs_splitted = outputs_splitted[0:-1]
    #
    # for output in outputs_splitted:
    #
    #     print(output)
    #     output_splitted = output.split(" ")
    #     docid = output_splitted[2]
    #     output = output + "\n"
    #     run_file.write(output)
    #     print(corpus_dictionary[docid]['sent_org'])
    #     print(corpus_dictionary[docid]['name'])
    #     print(corpus_dictionary[docid]['plabel'])
    #
    # run_file.close()

# for int_document_id, score in results:
#     ext_document_id, _ = index.document(int_document_id)
#     print(math.exp(score), corpus_dictionary[ext_document_id]['sent_org'])
    #print(index.document(int_document_id))
    #print(ext_document_id, score)
