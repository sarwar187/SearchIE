import numpy as np
import json
import operator
import os
#201758
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({'font.size': 12})
x = np.arange(1, 31)
print x
range = 30
x = x[:range]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def load_gold():
    fe_all_filename = os.path.join(ROOT_DIR, "../../data/gold/fatalencs/fe-all.json")
    alldata = [json.loads(line) for line in open(fe_all_filename)]
    testents = set(d['name'] for d in alldata if '2016-09-01' <= d['date'] <= '2016-12-31')
    histents= set(d['name'] for d in alldata if d['date'] < '2016-09-01')
    all = testents.union(histents)
    #print all.pop()
    return all

#load_gold()
def load_dictionary():
    corpus_file_json = open("../../data/all.json")
    id_dict = {}
    for line in corpus_file_json:
        data = json.loads(line)
        id_dict.setdefault(data['docid'], [])
        id_dict[data['docid']].append(data['name'])
    return id_dict

#1 Q0 1498571_322_0 1 -6.30739 IR
def get_metric_values(file, chunk_size, id_dict, name_set, retrieval_size):
    name_dict = {}
    for i in np.arange(chunk_size):
        line = file.readline()
        #print line
        line_splitted = line.split()
        name_list = id_dict[line_splitted[2].strip()]
        #print i
        if i < retrieval_size:
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
        #rank_incl_hist += 1
        #name_set contains the gold data
        if e in name_set:
            #print 'matched'
            set_of_retrieved_names.add(e)
            #rank += 1
            #if e in ts:
            tp += 1
            fn -= 1
        else:
            fp += 1
        precs.append(tp/(tp+fp))
        recs.append(1)
    #print 'precs 50 ' + str(precs[10])
    #print precs[20]
    #print len(set_of_retrieved_names)
    #print set_of_retrieved_names
    return precs[30], precs[50], precs[100], float(len(set_of_retrieved_names)), set_of_retrieved_names

id_dictionary = load_dictionary()
name_set = load_gold()
#print 'printing name set'
#print len(name_set)
p_30 = np.zeros(30)
p_50 = np.zeros(30)
p_100 = np.zeros(30)
civilians_found = np.zeros(30)

files = ["iterative_lr_run_dir/clean", "iterative_prf_run_dir", "iterative_lm_run_dir"]

#configurations
number_of_samples = 1
number_of_examples = 30
number_of_sentences = 1000
retrieval_size = 1000
result = np.zeros((number_of_examples, 3))
retrieval_approach = 'lm'

for i in np.arange(number_of_samples):
    list_lr = []
    list_lm = []
    list_prf = []

    for j in np.arange(len(files)):
        #print 'at least printing'
        #file = open("../../data/runs/iterative_lr_run_dir/clean/" + str(i + 1) + ".xml.run")
        #file = open("../../data/runs/iterative_prf_run_dir/" + str(i + 1) + ".xml.run")
        file = open("../../data/runs/" + files[j] + "/" + str(i + 1) + ".xml.run")

        #file_to_write = open("clean/" + str(i + 1) + ".xml.run", "w")
        #print 'starting with file ' + str(i + 1)
        # 1 Q0 1482418_115_0 3 -6.45492 IR
        p_30_list = []
        p_50_list = []
        p_100_list = []
        civilians_found_list = []
        #print file
        for k in np.arange(number_of_examples):
            #print 'working with query number ' + str(i + 1)
            precs30, precs50, precs100, number_of_civilians_found, name_of_civilians = get_metric_values(file, number_of_sentences, id_dictionary, name_set, retrieval_size)
            p_30_list.append(precs30)
            p_50_list.append(precs50)
            p_100_list.append(precs100)
            civilians_found_list.append(number_of_civilians_found)
            #print (name_of_civilians)
            if files[j]=='iterative_lr_run_dir/clean':
                list_lr.append(name_of_civilians)
                #print(len(list_lr))
                #print(list_lr)
            elif files[j]=='iterative_prf_run_dir':
                list_prf.append(name_of_civilians)
            else:
                list_lm.append(name_of_civilians)

            #print list_lr
        p_30 += np.asarray(p_30_list)
        p_50 += np.asarray(p_50_list)
        p_100 += np.asarray(p_100_list)
        civilians_found+=np.asarray(civilians_found_list)

    #print list_lr

    if retrieval_approach=='lm':
        for m in np.arange(number_of_examples):
            #print len(list_lr)
            set_union = list_lr[m] | list_lm[m]
            set_intersection = list_lr[m] & list_lm[m]
            only_lr = set_union - (set_intersection | list_lm[m])
            only_lm = set_union - (set_intersection | list_lr[m])
            result[m][0]+= len(only_lr)
            result[m][1]+= len(only_lm)
            result[m][2]+= len(set_intersection)
    else:
        for m in np.arange(number_of_examples):
            #print len(list_lr)
            set_union = list_lr[m] | list_prf[m]
            set_intersection = list_lr[m] & list_prf[m]
            only_lr = set_union - (set_intersection | list_prf[m])
            only_prf = set_union - (set_intersection | list_lr[m])
            result[m][0]+= len(only_lr)
            result[m][1]+= len(only_prf)
            result[m][2]+= len(set_intersection)

print ("extraction\tretrieval\tcommon")
print ("--------------------------------------------------")

for m in np.arange(number_of_examples):
    result[m][0]/= number_of_samples
    result[m][1]/= number_of_samples
    result[m][2]/= number_of_samples
    total = result[m][0] + result[m][1] + result[m][2]
    print ("%0.3f" % (result[m][0]/total) + "\t" + "%0.3f" % (result[m][1]/total) + "\t" + "%0.3f" % (result[m][2]/total))
    result[m][0]/= total
    result[m][1]/= total
    result[m][2]/= total

plt.plot( x, result[:range, 0], marker='o', markerfacecolor='black', markersize=6, color='darkgray', linewidth=2, label='Extraction', linestyle='-.')
plt.plot( x, result[:range, 1], marker='v', markerfacecolor='blue', markersize=6, color='gray', linewidth=2, label='Retrieval', linestyle='--')
plt.plot( x, result[:range, 2], marker='x', markerfacecolor='red', markersize=6, color='black', linewidth=2, label='Common', linestyle=':')

plt.xlabel('Number of Examples in Query')
plt.ylabel('Percentage of Retrieved Entities for Different Models')
plt.legend()
plt.savefig("model_diversity_prf")
plt.show()

p_30 /= 50
p_50 /= 50
p_100 /= 50
civilians_found /= 50


p_30_str = '['
for i in p_30:
    p_30_str += str(i) + ","
print p_30_str + ']'


p_50_str = '['
for i in p_50:
    p_50_str += str(i) + ","
print p_50_str + ']'

p_100_str = '['
for i in p_100:
    p_100_str += str(i) + ","
print p_100_str + ']'

civilians_found_str = '['
for i in civilians_found:
    civilians_found_str += str(i) + ","
print civilians_found_str + ']'

#civilians_found_str

    #return sorted_data[:1000]
# file =  open("44.xml.run")
# file_to_write = open("cleaned/44.xml.run")
# #1 Q0 1482418_115_0 3 -6.45492 IR
# for i in np.arange(30):
#     sorted_data = read_chunk(file)
#     rank = 1
#     for tuple in sorted_data:
#         str_to_write = str(i+1) + " Q0 " + tuple[0] + " " + str(rank) + " " + str(tuple[1]) + " LR\n"
#         file_to_write.write(str_to_write)
#         rank+=1
# file_to_write.close()