from subprocess import run

run(["../build/linux/x86_64/release/TDFANN", "query", "-v", "../dataset/sift/sift_base.fvecs", "-i", "../test_data/tdfg_100.graph", "-n", "10", "-b", "50", "-q", "../dataset/sift/sift_query.fvecs", "-r", "../test_data/sift1m_result_10.idx", "-l"])