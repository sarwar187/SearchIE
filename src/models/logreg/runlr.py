from __future__ import division
import argparse, json
import numpy as np
import sys
from collections import defaultdict
import pickle
import em 
from scipy import sparse, io

def read_file(file_name):
    #reads in .json train data
    Y = [] #the pseudolabel classes
    E = [] #entity ID for each mention
    Epos = {} #whether each entity is positive or not. 
    toprint = []
    with open(file_name, 'r') as r:
        for line in r:
            obj = json.loads(line)
            docid = obj["docid"]
            plabel = float(obj["plabel"])
            assert type(plabel) == float
            assert plabel == 0.0 or plabel == 1.0
            name = obj["name"]
            Y.append(plabel)
            E.append(name)
            if plabel == 1.0: Epos[name] = True
            elif plabel ==0.0: Epos[name]= False
            toprint.append({'id': docid, 'name': name})
    Y = np.array(Y)
    assert len(Y) == len(E) == len(toprint)
    print "READ {0} with {1} files".format(file_name, len(Y))
    return Y, E, Epos, toprint

def read_file_all(file_name, sentence_ids):
    #reads in .json train data
    Y_train = [] #the pseudolabel classes
    E_train = [] #entity ID for each mention
    Epos_train = {} #whether each entity is positive or not.
    toprint_train = []
    Y_test = []  # the pseudolabel classes
    E_test = []  # entity ID for each mention
    Epos_test = {}  # whether each entity is positive or not.
    toprint_test = []
    line_id = 0
    with open(file_name, 'r') as r:
        for line in r:
            obj = json.loads(line)
            docid = obj["docid"]
            plabel = float(obj["plabel"])
            assert type(plabel) == float
            assert plabel == 0.0 or plabel == 1.0
            name = obj["name"]
            if line_id in sentence_ids:
                Y_train.append(plabel)
                E_train.append(name)
                if plabel == 1.0: Epos_train[name] = True
                elif plabel ==0.0: Epos_train[name]= False
                toprint_train.append({'id': docid, 'name': name})
            #else:
            Y_test.append(plabel)
            E_test.append(name)
            if plabel == 1.0:
                Epos_test[name] = True
            elif plabel == 0.0:
                Epos_test[name] = False
            toprint_test.append({'id': docid, 'name': name})
            line_id+=1

    Y_train = np.array(Y_train)
    assert len(Y_train) == len(E_train) == len(toprint_train)
    print "READ {0} with {1} files".format(file_name, len(Y_train))

    Y_test = np.array(Y_test)
    assert len(Y_test) == len(E_test) == len(toprint_test)
    print "READ {0} with {1} files".format(file_name, len(Y_test))

    return Y_train, E_train, Epos_train, toprint_train, Y_test, E_test, Epos_test, toprint_test, line_id

def read_mtx(data_mtx, sentence_ids, line_id):
    #data_mtx = io.mmread(data_mtx_file)
    data_mtx = data_mtx[0:line_id, :]
    prior_w0 = np.sum(data_mtx[line_id: , :], axis=0)
    X_train = data_mtx[sentence_ids, :]

    #test_sentence_ids = [x for x in np.arange(201758) if x not in sentence_ids]
    #X_test = data_mtx[test_sentence_ids, :]
    return X_train, data_mtx, prior_w0

#read_mtx("../../../all_ng_dep.mtx", [1,3])

def run_logistic_regression(data_file, data_mtx, sentence_ids, output_file, number_of_iterations):
    #Y_train, X_train, E, Epos, toprint = read_file_all(data, sentence_ids)
    Y_train, E_train, Epos_train, toprint_train, Y_test, E_test, Epos_test, toprint_test, line_id = read_file_all(data_file, sentence_ids)
    X_train, X_test, prior_w0 = read_mtx(data_mtx, sentence_ids, line_id)
    EM_ITERS = number_of_iterations
    emModel = em.go_em(output_file, X_train, X_test, Y_test, E_train, Epos_train, toprint_test, prior_w0,  Niter=EM_ITERS)

def run_ir_logistic_regression(data_file, feature_file, sentence_ids, output_file, number_of_iterations):
    # Y_train, X_train, E, Epos, toprint = read_file_all(data, sentence_ids)
    Y_train, E_train, Epos_train, toprint_train, Y_test, E_test, Epos_test, toprint_test, line_id = read_file_all(data_file, sentence_ids)
    X_train, X_test, prior_w0 = read_mtx(data_mtx, sentence_ids, line_id)
    EM_ITERS = number_of_iterations
    emModel = em.go_em(output_file, X_train, X_test, Y_test, E_train, Epos_train, toprint_test, prior_w0, Niter=EM_ITERS)

    # pickle that iteration of models
    # pkl = 'lr-soft-final.pkl'
    # w = open(pkl, 'w')
    # pickle.dump(emModel, w)
    # w.close()
    # print "model saved to ", pkl


# def main():
#     arg_parser = argparse.ArgumentParser()
#     arg_parser.add_argument('train', type=str, help='training .json file, both 0 and 1 examples')
#     arg_parser.add_argument('test', type=str, help='testing .json file, both 0 and 1 examples')
#     arg_parser.add_argument('X_train', type=str, nargs='?', help='training feature matrix, like train.mtx')
#     arg_parser.add_argument('X_test', type=str, nargs='?', help='testing feature matrix, like test.mtx')
#     arg_parser.add_argument('--emiters', type=int, help='number of iters for EM', default=50)
#     args = arg_parser.parse_args()
#
#     #emModel = em.go_em(X_train, X_test, Y_test, E_train, Epos_train, toprint_test, Niter=EM_ITERS)
#     EM_ITERS = args.emiters
#
#     #---FULL DATASET-----
#     Y_train, E_train, Epos_train, toprint_train = read_file(args.train)
#     Y_test, E_test, Epos_test, toprint_test = read_file(args.test)
#
#     #----READ IN FEATURE MATRICES
#     X_train = io.mmread(args.X_train)
#     X_test = io.mmread(args.X_test)
#
#     # print "----------EM--------------"
#     # emModel = em.go_em(X_train, X_test, Y_test, E_train, Epos_train, toprint_test, Niter=EM_ITERS)
#     #
#     # #pickle that iteration of models
#     # pkl = 'lr-soft-final.pkl'
#     # w = open(pkl, 'w')
#     # pickle.dump(emModel, w)
#     # w.close()
#     #print "model saved to ", pkl
#
# if __name__ == "__main__":
#     main()
