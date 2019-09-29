import json
import subprocess as sp
import os
import sys
sys.path.append("../../../src")
#sys.path.append("../../../src/models/logreg")
from models.logreg.runlr import *

def main():
    corpus_file_json = open("../../../data/all.json")
    corpus_dictionary = {}

    for line in corpus_file_json:
        data = json.loads(line)
        corpus_dictionary.setdefault(data['docid'] , data)

    data_mtx = io.mmread("../../../all_ng_dep.mtx")
    #data_mtx = data_mtx.todense()
    data_mtx = data_mtx.tocsr()
    file = "../../../data/query_configurations/iterative_query_lr_dir/1.xml"
    run_file = open("../../../data/runs/iterative_lr_run_dir/1_test.run", "w")
    line_id = 1
    for line in open(file):
        sentence_ids = line.split()
        sentence_ids = [int(item) for item in sentence_ids]
        run_logistic_regression("../../../data/all.json", data_mtx, sentence_ids, run_file, number_of_iterations=20)
        print "starting logistic regression for query number " + line_id
        print "------------------------------------------------------------"
        line_id += 1
    run_file.close()

if __name__=="__main__":
    main()