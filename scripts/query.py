from subprocess import run

# ../build/linux/x86_64/release/TDFANN query -v ../dataset/sift/sift_base.fvecs -i ../test_data/tdfg_100.graph -n 10 -b 300 -q ../dataset/sift/sift_query.fvecs -r ../test_data/sift1m_result_10.idx -a ../test_data/sift1m_ans_10.idx

for i in [50, 100, 150, 200, 250, 300]:
    for j in [20, 50, 75, 100, 150, 200, 250, 300]:
        run(["../build/linux/x86_64/release/TDFANN", "query", "-v", "../dataset/sift/sift_base.fvecs", "-i", f"../test_data/tdfg_{i}.graph", "-n", "10", "-b", f"{j}", "-q", "../dataset/sift/sift_query.fvecs", "-r", "../test_data/sift1m_result_10.idx", "-a", "../test_data/sift1m_ans_10.idx"])
        with open("logs/tdfann.log", "r") as f:
            lines = f.readlines()
            with open("result.txt", "a") as fout:
                fout.write(f"{i} {j} {lines[-2].split(' ')[-2]} {lines[-1].split(' ')[-1]}")