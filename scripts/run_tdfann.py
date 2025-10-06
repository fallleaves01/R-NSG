from subprocess import run
import sys

exec_path = "../build/linux/x86_64/release/TDFANN"

k = 1500
beam_size = 40
range_step = 1500
qnumber = 10
dataset_file = "/mnt/win-dai/Vectors/sift/sift_base.fvecs"
query_file = "/mnt/win-dai/Vectors/sift/sift_query.fvecs"
result_file = "../test_data/sift_result_1000.bin"
groundtruth_file = "/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/sif_gt_sel_6_1_100000_random_10.json"
knng_file = "../test_data/knng_1500.graph"
qrange_file = "/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/qrangesel_6_1_100000_random.json"
label_file = "/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/attr_sel_1_100000_random.json"
index_file = "../test_data/tdfg_1500.graph"

knng_cmd = f"{exec_path} --verbose knng\
 --dataset_file {dataset_file}\
 -k {k}\
 --graph_file {knng_file}\
"

build_cmd = f"{exec_path} --verbose build\
 --range_step {range_step}\
 --dataset_file {dataset_file}\
 --knng_file {knng_file}\
 --label_file {label_file}\
 --index_file {index_file}\
"

query_cmd = f"{exec_path} --verbose query\
 --dataset_file {dataset_file}\
 --index_file {index_file}\
 --query_file {query_file}\
 --label_file {label_file}\
 --qrange_file {qrange_file}\
 --qnumber {qnumber}\
 --beam_size {beam_size}\
 --result_file {result_file}\
 --groundtruth_file {groundtruth_file}\
"

if sys.argv[1] == "build":
    print(build_cmd)
    run(build_cmd.split(" "))
elif sys.argv[1] == "query":
    print(query_cmd)
    run(query_cmd.split(" "))
elif sys.argv[1] == "knng":
    print(knng_cmd)
    run(knng_cmd.split(" "))
