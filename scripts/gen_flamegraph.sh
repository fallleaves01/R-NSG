perf record -g --call-graph dwarf ../build/linux/x86_64/release/TDFANN query -v ../dataset/sift/sift_base.fvecs -i ../test_data/tdfg_300.graph -n 10 -b 150 -q ../dataset/sift/sift_query.fvecs -r ../test_data/sift1m_result_10.idx -a ../test_data/sift1m_ans_10.idx

perf script | perl ./stackcollapse-perf.pl | perl ./flamegraph.pl > flame.svg