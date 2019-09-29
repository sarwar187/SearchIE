from trectools import TrecRun, TrecQrel, TrecRes, misc
import matplotlib.pylab as plt
import collections

from os import listdir
from os.path import isfile, join
import numpy as np


#retrieval_approaches = ['lm', 'prf']
#retrieval_approaches = ['lm', 'prf', 'lr']
retrieval_approaches = ['prf']


myQrel = TrecQrel("../../data/runs/iterative_qrel")
print 'qrel description '
#print myQrel.describe()




#results = []

for retrieval_approach in retrieval_approaches:
    print retrieval_approach
    if retrieval_approach=='lm':
        mypath = "../../data/runs/iterative_lm_run_dir"
    elif retrieval_approach =='prf':
        mypath = "../../data/runs/iterative_prf_run_dir"
    elif retrieval_approach == 'lr':
        mypath = "../../data/runs/iterative_lr_run_dir/clean"
    else:
        mypath = "../../data/runs/iterative_lr_ir_run_dir/"

    run_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.run')]
    print run_files
    p_10 = np.zeros(30)
    p_20 = np.zeros(30)
    count = 0

    for run_file in run_files:
        run = TrecRun(run_file)
        #print 'run loaded'
        res = run.evaluate_run(myQrel)
        #print 'run evaluated'
        keys = [item for item in res.get_results_for_metric("P_20").keys()]
        keys = sorted(keys, key=int)
        values_p20 = [res.get_results_for_metric("P_20")[i] for i in keys]
        values_p20 = np.asarray(values_p20)
        p_20+=values_p20

        keys = [item for item in res.get_results_for_metric("P_10").keys()]
        keys = sorted(keys, key=int)
        values_p10 = [res.get_results_for_metric("P_10")[i] for i in keys]
        values_p10 = np.asarray(values_p10)
        p_10 += values_p10
        count+=1
        #print 'processed ' + str(count) + 'file'

    p_10/=count
    p_20 /= count

    p_10_str = '['
    for i in p_10:
        p_10_str+= str(i) + ","
    print p_10_str + ']'

    p_20_str = '['
    for i in p_20:
        p_20_str += str(i) + ","
    print p_20_str + ']'



#plt.plot('x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
#plt.plot('x', 'y2', data=df, marker='', color='olive', linewidth=2)
#plt.plot('x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")

#plt.show()

# lm_run = TrecRun("../../data/runs/lm.run")
# print 'run loaded'
# #print myRun
# print 'topics '
# print lm_run.topics()
#
# print 'top documents '
# print lm_run.get_top_documents(topic=1,n=2)
#
# myQrel = TrecQrel("../../data/runs/iterative_qrel")
# print 'qrel description '
# print myQrel.describe()
#
# print 'number of queries qrel '
# print myQrel.get_number_of(1)
#
# print 'number of queries qrel '
# print myQrel.get_number_of(2)
#
# print 'agreement '
# print myQrel.check_agreement(myQrel)
#
# lm_res = lm_run.evaluate_run(myQrel)
# print 'precision at 10 '
# print lm_res.get_result(metric="P_10")
#
# print 'p at 10'
# print lm_res.get_results_for_metric("P_10")
#
# #print len(*zip(*sorted(myRes.get_results_for_metric("P_10").items())))
#
# prf_run = TrecRun("../../data/runs/prf.run")
# prf_res = prf_run.evaluate_run(myQrel)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
#
# keys = [item for item in prf_res.get_results_for_metric("P_20").keys()]
# keys = sorted(keys, key=int)
#
# values = [prf_res.get_results_for_metric("P_20")[i] for i in keys]
# print values
# ax1.plot(values, label='first')
#
# keys = [item for item in lm_res.get_results_for_metric("P_20").keys()]
# keys = sorted(keys, key=int)
# values = [lm_res.get_results_for_metric("P_20")[i] for i in keys]
# print values
# ax1.plot(values, label='second')
#
# plt.show()
#
# print prf_res.get_results_for_metric("P_10")
#
# print 'result comparison '
# print lm_res.compare_with(prf_res, metric="map")
#
# list_of_results = [lm_res, prf_res]
#
# misc.sort_systems_by(list_of_results, "P_10")
# misc.get_correlation( misc.sort_systems_by(list_of_results, "P_10"), misc.sort_systems_by(list_of_results, "map") )
# misc.get_correlation( misc.sort_systems_by(list_of_results, "P_10"), misc.sort_systems_by(list_of_results, "map"), correlation="tauap" )
