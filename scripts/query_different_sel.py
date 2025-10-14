from subprocess import run
import sys

exec_path = "../build/linux/x86_64/release/TDFANN"
dataset_list = ["sift", "wit", "tripclick", "yt_audio_all", "yt_rgb_all"]
# dataset = "tripclick"
def run_query(dataset, s, beam_size):
    k = 300
    # beam_size = 100
    range_step = 10000
    qnumber = 10
    trunc_size = 200
    # s = 1 # selectivity 1->mix, 8->25%, 10->10% 19->1%
    knng_file = f"../test_data/knng_{dataset}_{k}.graph"
    index_file = f"../test_data/tdfg_{dataset}_{k}_{range_step}_new.graph"
    result_file = "../test_data/tmp.bin"
    groundtruth_file = f"/mnt/win-dai/Vectors/{dataset}/label/sel_1_100000_real/sif_gt_sel_{s}_1_100000_real_10.json"
    qrange_file = f"/mnt/win-dai/Vectors/{dataset}/label/sel_1_100000_real/qrangesel_{s}_1_100000_real.json"
    label_file = f"/mnt/win-dai/Vectors/{dataset}/label/sel_1_100000_real/attr_sel_1_100000_real.json"

    if dataset == "sift":
        range_step = 1500
        trunc_size = 50
        dataset_file = "/mnt/win-dai/Vectors/sift/sift_base.fvecs"
        query_file = "/mnt/win-dai/Vectors/sift/sift_query.fvecs"
        groundtruth_file = f"/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/sif_gt_sel_{s}_1_100000_random_10.json"
        knng_file = f"../test_data/knng_{k}.graph"
        qrange_file = f"/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/qrangesel_{s}_1_100000_random.json"
        label_file = "/mnt/win-dai/Vectors/sift/label/sel_1_100000_random/attr_sel_1_100000_random.json"
        index_file = f"../test_data/tdfg_{range_step}.graph"
    elif dataset == "wit":
        range_step = 6000
        trunc_size = 50
        index_file = f"../test_data/tdfg_{dataset}_{k}_{range_step}_new.graph"
        dataset_file = "/mnt/win-dai/Vectors/wit/wit1M_embeddings.fvecs"
        query_file = "/mnt/win-dai/Vectors/wit/wit1k_queries.fvecs"
    elif dataset == "tripclick":
        trunc_size = 60
        dataset_file = "/mnt/win-dai/Vectors/tripclick/tripclick_base.fvecs"
        query_file = "/mnt/win-dai/Vectors/tripclick/tripclick_query.fvecs"
    elif dataset == "yt_audio_all":
        trunc_size = 50
        range_step = 3000
        index_file = f"../test_data/tdfg_{dataset}_{k}_{range_step}_new.graph"
        dataset_file = "/mnt/win-dai/Vectors/yt_audio_all/audio.fvecs"
        query_file = "/mnt/win-dai/Vectors/yt_audio_all/query_audio.fvecs"
    elif dataset == "yt_rgb_all":
        dataset_file = "/mnt/win-dai/Vectors/yt_rgb_all/rgb.fvecs"
        query_file = "/mnt/win-dai/Vectors/yt_rgb_all/query_rgb.fvecs"
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
    print(query_cmd)
    run(query_cmd.split(" "))

for dataset in ["yt_audio_all"]:
    print(f"##?Running dataset: {dataset}")
    for sel in [1, 8, 10, 19]:
        for beam in [10, 12, 15, 17, 20, 25, 30, 40, 50, 60, 70, 90, 110, 130, 150]:
            run_query(dataset, sel, beam)