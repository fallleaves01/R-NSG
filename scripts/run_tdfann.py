from subprocess import run
import sys

exec_path = "../build/linux/x86_64/release/TDFANN"
dataset_list = ["sift", "wit", "tripclick", "yt_audio_all", "yt_rgb_all"]
dataset = "tripclick"

k = 300
beam_size = 50
range_step = 1500
trunc_size = 50
qnumber = 10
s = 1 # selectivity 1->mix, 8->25%, 10->10% 19->1%
knng_file = f"../test_data/knng_{dataset}_{k}.graph"
index_file = f"../test_data/tdfg_{dataset}_{k}_{range_step}_new.graph"
result_file = "../test_data/tmp.bin"
groundtruth_file = f"/mnt/win-dai/Vectors/{dataset}/label/sel_1_100000_real/sif_gt_sel_{s}_1_100000_real_10.json"
qrange_file = f"/mnt/win-dai/Vectors/{dataset}/label/sel_1_100000_real/qrangesel_{s}_1_100000_real.json"
label_file = f"/mnt/win-dai/Vectors/{dataset}/label/sel_1_100000_real/attr_sel_1_100000_real.json"

if dataset == "sift":
    dataset_file = "/mnt/win-dai/Vectors/sift/sift_base.fvecs"
    query_file = "/mnt/win-dai/Vectors/sift/sift_query.fvecs"
    groundtruth_file = f"/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/sif_gt_sel_{s}_1_100000_random_10.json"
    knng_file = f"../test_data/knng_{k}.graph"
    qrange_file = f"/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/qrangesel_{s}_1_100000_random.json"
    label_file = "/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/attr_sel_1_100000_random.json"
    index_file = f"../test_data/tdfg_{range_step}.graph"
elif dataset == "wit":
    dataset_file = "/mnt/win-dai/Vectors/wit/wit1M_embeddings.fvecs"
    query_file = "/mnt/win-dai/Vectors/wit/wit1k_queries.fvecs"
elif dataset == "tripclick":
    dataset_file = "/mnt/win-dai/Vectors/tripclick/tripclick_base.fvecs"
    query_file = "/mnt/win-dai/Vectors/tripclick/tripclick_query.fvecs"
elif dataset == "yt_audio_all":
    dataset_file = "/mnt/win-dai/Vectors/yt_audio_all/audio.fvecs"
    query_file = "/mnt/win-dai/Vectors/yt_audio_all/query_audio.fvecs"
elif dataset == "yt_rgb_all":
    dataset_file = "/mnt/win-dai/Vectors/yt_rgb_all/rgb.fvecs"
    query_file = "/mnt/win-dai/Vectors/yt_rgb_all/query_rgb.fvecs"

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
 --trunc_size {trunc_size}\
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
