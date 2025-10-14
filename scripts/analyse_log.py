import json

f = open("q_sel_audio.out")
l = f.readlines()

ds = None
data = {}

for line in l:
    for n in ["wit", "tripclick", "yt_audio", "yt_rgb", "sift"]:
        if n in line:
            ds = n
    if "QPS" in line:
        q = float(line.split()[-1])
        if ds not in data:
            data[ds] = {"qps": [], "recall": []}
        data[ds]["qps"].append(q)
    if "Recall" in line:
        r = float(line.split()[-1])
        data[ds]["recall"].append(r)
print(data)

res = {
    # "sift1M":{
    #     "mixed":{
    #         "recall": [0.9003, 0.9217, 0.9427, 0.965, 0.9727, 0.9768, 0.9835, 0.9884, 0.9916, 0.9943, 0.9951, 0.9967, 0.9977, 0.9983, 0.999, 0.9994],
    #         "qps": [5543.65, 4968.74, 4236.13, 3401.17, 3079.93, 2868.15, 2392.83, 2116.95, 1914.28, 1675.33, 1577.01, 1332.65, 1162.79, 997.21, 849.63, 657.03]
    #     },
    #     "25%":{
    #         "recall": [0.8965, 0.9178, 0.9332, 0.9451, 0.9503, 0.9615, 0.9812, 0.9894, 0.9936, 0.995, 0.9979, 0.9987, 0.9991, 0.9993, 0.9994],
    #         "qps": [5760.13, 4951.42, 4227.09, 3794.17, 3436.17, 3044.03, 2227.16, 1692.11, 1460.63, 1241.26, 938.54, 755.82, 656.24, 627.85, 600.48]
    #     },
    #     "10%":{
    #         "recall": [0.8880, 0.9092, 0.9259, 0.9394, 0.9486, 0.9568, 0.9704, 0.9786, 0.9880, 0.9927, 0.9954, 0.9969, 0.9985, 0.9991, 0.9996, 0.9997],
    #         "qps": [7065.56, 6173.09, 5591.37, 5139.53, 4683.13, 4244.77, 3631.42, 3120.48, 2439.36, 1916.06, 1655.68, 1359.68, 1078.86, 862.28, 765.05, 638.73]
    #     },
    #     "1%":{
    #         "recall": [0.9143, 0.9351, 0.9573, 0.9761, 0.9829, 0.9917, 0.9967, 0.9987, 0.9994, 0.9996, 0.9998, 0.9999, 1.000],
    #         "qps": [9952.89, 9268.90, 8064.87, 6225.02, 5847.11, 4473.69, 3652.46, 3018.50, 2666.75, 2399.89, 2302.69, 2018.65, 1827.92]
    #     }
    # },
    # "YT-Audio":{
    #     "mixed":{
    #         "recall": [0.9627, 0.9674, 0.9724, 0.9827, 0.9899, 0.993, 0.9949, 0.9971, 0.9979, 0.9981],
    #         "qps": [6726.14, 6182.51, 5835.71, 5183.30, 4192.30, 3485.55, 3115.46, 2395.74, 1698.54, 1272.41]
    #     },
    # },
}
dsmap = {
    "tripclick": "TripClick",
    "yt_audio": "YT-Audio",
    "yt_rgb": "YT-RGB",
    "wit": "WIT",
    "sift": "sift1M"
}
smap = {
    1: "mixed",
    8: "25%",
    10: "10%",
    19: "1%"
}
for ds in ["yt_audio"]:
    if ds not in data:
        continue
    for sel in [1, 8, 10, 19]:
        # if ds == "yt_audio" and sel == 1:
        #     continue
        if not dsmap[ds] in res:
            res[dsmap[ds]] = {}
        if not smap[sel] in res[dsmap[ds]]:
            res[dsmap[ds]][smap[sel]] = {"recall": [], "qps": []}
        res[dsmap[ds]][smap[sel]]["recall"] = data[ds]["recall"][:15]
        res[dsmap[ds]][smap[sel]]["qps"] = data[ds]["qps"][:15]
        data[ds]["recall"] = data[ds]["recall"][15:]
        data[ds]["qps"] = data[ds]["qps"][15:]
for ds in data:
    for sel in data[ds]:
        if len(data[ds][sel]) != 0:
            print(f"Warning: not all data used for {ds} {sel}")
with open("q_sel_audio.json", "w") as f:
    json.dump(res, f, indent=4)
print("done")