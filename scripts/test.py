from subprocess import run

#../build/linux/x86_64/release/TDFANN --verbose build -v ../dataset/sift/sift_base.fvecs -k ../test_data/knng_300.graph -i ../test_data/tdfg_300.graph
# for i in [50, 100, 150, 200, 250, 300]:
#     run(["../build/linux/x86_64/release/knng_builder", "--verbose", "-v", "../dataset/sift/sift_base.fvecs", "-k", str(i), "-g", f"../test_data/knng_{i}.graph"])

for i in [50, 100, 150, 200, 250, 300]:
    run(["../build/linux/x86_64/release/TDFANN", "--verbose", "build", "-v", "../dataset/sift/sift_base.fvecs", "-k", f"../test_data/knng_{i}.graph", "-i", f"../test_data/tdfg_{i}.graph"])