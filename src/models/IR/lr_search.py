import json
import subprocess as sp
import os
from src.models.logreg.runlr import *
sys.path.append("../../../src")
#sys.path.append("../../../src/models/logreg")
from models.logreg.runlr import *
import sys

def main():
    print "parameters to the program " + sys.argv[1] + "\t" + sys.argv[2]
    file = sys.argv[1]
    number_of_em_iterations = int(sys.argv[2])


    corpus_file_json = open("/home/smsarwar/PycharmProjects/civilian_killing/data/all.json")
    corpus_dictionary = {}

    for line in corpus_file_json:
        data = json.loads(line)
        #print(list(data.keys()))
        #print (key)
        corpus_dictionary.setdefault(data['docid'] , data)

    retrieval_methods = ["lr"]

    #retrieval_methods = ["single"]

    data_mtx = io.mmread("../../../all_ng_dep.mtx")
    data_mtx = data_mtx.tocsr()

    for retrieval_method in retrieval_methods:
        if retrieval_method=='lr':
            path = "../../../data/query_configurations/iterative_query_lr_dir/"
            os.chdir(path)
            #onlyfiles = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
            count = 0
            #for file in onlyfiles:
            run_file = open("../../../data/runs/iterative_lr_run_dir/" + os.path.basename(file) + ".run", "w")
            line_id = 1
            for line in open(file):
                sentence_ids = line.split()
                sentence_ids = [int(item) for item in sentence_ids]
                #corpus_file_json = open("../../../data/all.json")
                #data_mtx_file = open("../../../all_ng_dep.mtx")
                print "starting logistic regression for query number " + line_id
                run_logistic_regression("../../../data/all.json", data_mtx, sentence_ids, run_file, number_of_iterations=number_of_em_iterations)
                print "------------------------------------------------------------"
                line_id+=1
            run_file.close()

if __name__=="__main__":
    main()