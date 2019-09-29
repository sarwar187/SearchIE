import numpy as np

file = open("scripts/run_lr" , "w")
directory = "/mnt/nfs/work1/smsarwar/civilian_killing/src/models/IR/"

for i in np.arange(50):
    input_file = str(i+1) + ".xml"
    input_file_script = open("scripts/" + str(i+1) + ".sh", "w")
    input_file_script.write("python ../lr_search.py " + input_file + " 20\n")
    input_file_script.close()
    file.write("sbatch -o " + directory + "output/" + str(i+1) + ".out -e " + directory + "output/"+ str(i+1) + ".err --mem=20G " + directory + str(i+1) + ".sh\n")
file.close()


