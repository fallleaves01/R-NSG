from subprocess import run
import sys
from random import randint
import json, math

exec_path = "../build/linux/x86_64/release/TDFANN"
dataset = "siftsmall"
num_vectors = 1000000
num_queries = 10000
k = 50              # ef_spatial
beam_size = 120     # ef_search
range_step = 600    # ef_attribute
qnumber = 100       # k number of query
ef_max = 200        # m
s = 2               # selectivity index
sel_list = [1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
knng_file = f"../test_data/knng_{dataset}_{k}.graph"
index_file = f"../test_data/tdfg_{dataset}_{k}_{range_step}_ef{ef_max}.graph"
result_file = "../test_data/result.json"
groundtruth_file = f"../test_data/{dataset}/gt_sel_{s}_{qnumber}.json"
qrange_file = f"../test_data/{dataset}/qrangesel_{s}.json"
label_file = f"../test_data/{dataset}/attr.json"

def init(dataset, k, range_step = 1500, qnumber = 10, s = 1, ef_max = 500):
    global knng_file, index_file, result_file, groundtruth_file, qrange_file, label_file
    global dataset_file, query_file, num_vectors, num_queries
    knng_file = f"../test_data/knng_{dataset}_{k}.graph"
    index_file = f"../test_data/tdfg_{dataset}_{k}_{range_step}_ef{ef_max}.graph"
    groundtruth_file = f"../test_data/{dataset}/gt_sel_{s}_{qnumber}.json"
    qrange_file = f"../test_data/{dataset}/qrangesel_{s}.json"
    label_file = f"../test_data/{dataset}/attr.json"

    if dataset == "sift":
        num_vectors = 1000000
        num_queries = 10000
        dataset_file = "../test_data/sift/sift_base.fvecs"
        query_file = "../test_data/sift/sift_query.fvecs"
    elif dataset == "siftsmall":
        num_vectors = 10000
        num_queries = 100
        dataset_file = "../test_data/siftsmall/siftsmall_base.fvecs"
        query_file = "../test_data/siftsmall/siftsmall_query.fvecs"


def knng(dataset, k):
    init(dataset, k)
    knng_cmd = f"{exec_path} --verbose knng\
 --dataset_file {dataset_file}\
 -k {k}\
 --graph_file {knng_file}\
"
    print(knng_cmd, flush=True)
    run(knng_cmd.split(" "))

def build(dataset, k, range_step, ef_max):
    init(dataset, k, range_step, ef_max=ef_max)

    build_cmd = f"{exec_path} --verbose build\
 --range_step {range_step}\
 --dataset_file {dataset_file}\
 --knng_file {knng_file}\
 --label_file {label_file}\
 --index_file {index_file}\
 --ef_max {ef_max}\
"
    print(build_cmd, flush=True)
    run(build_cmd.split(" "))

def query(dataset, k, range_step, ef_max, beam_size, qnumber, s = 2, trunc_size = 50):
    init(dataset, k, range_step, qnumber, s, ef_max)
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
 --trunc_size {trunc_size}\
"
    print(query_cmd, flush=True)
    run(query_cmd.split(" "))

def gen_groundtruth(dataset, qnumber, s = 2):
    init(dataset, 50, qnumber=qnumber, s=s)
    gt_cmd = f"{exec_path} --verbose groundtruth\
 --dataset_file {dataset_file}\
 --query_file {query_file}\
 --label_file {label_file}\
 --qrange_file {qrange_file}\
 --qnumber {qnumber}\
 --result_file {result_file}\
"
    print(gt_cmd, flush=True)
    run(gt_cmd.split(" "))
    run(f"cp {result_file} {groundtruth_file}".split(" "))


def gen_attr(dataset, range_l = 0, range_r = 10 ** 5):
    init(dataset, 50)
    r = [randint(range_l, range_r) for _ in range(num_vectors)]
    json.dump(r, open(label_file, "w"))

def gen_qrange(dataset, range_l = 0, range_r = 10 ** 5, s = 2):
    init(dataset, 50, s = s)
    qr = []
    d = max(1, int(math.floor((range_r - range_l + 1) * sel_list[s])))
    for _ in range(num_queries):
        l = randint(range_l, range_r - d + 1)
        r = l + d - 1
        qr.append(l)
        qr.append(r)
    json.dump(qr, open(qrange_file, "w"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_py_api.py [knng|build|query|groundtruth|prepare]")
        exit(1)
    if sys.argv[1] == "knng":
        knng(dataset, k)
    elif sys.argv[1] == "build":
        build(dataset, k, range_step, ef_max)
    elif sys.argv[1] == "query":
        query(dataset, k, range_step, ef_max, beam_size, qnumber, s)
    elif sys.argv[1] == "groundtruth":
        gen_groundtruth(dataset, qnumber)
    elif sys.argv[1] == "prepare":
        print(f"Generating attribute for {dataset}")
        gen_attr(dataset)
        print(f"Generating query range for {dataset}")
        gen_qrange(dataset)
        print(f"Generating ground truth for {dataset}")
        gen_groundtruth(dataset, qnumber)
        print("Done")
    else:
        print("Unknown command")
        exit(1)
