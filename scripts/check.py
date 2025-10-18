# -*- coding: utf-8 -*-
"""
fvecs_reader.py
读取 .fvecs 文件并检查维度一致性；若一致，可加载为 numpy 数组 (n, d)。

用法：
    python fvecs_reader.py /path/to/data.fvecs

不带参数时，会提示输入路径。
"""

import os
import sys
import struct
from collections import Counter
from typing import Tuple, Dict, Optional

import numpy as np


def scan_fvecs_dims(path: str, max_vectors: Optional[int] = None) -> Tuple[Dict[int, int], int, bool]:
    """
    扫描 fvecs 文件里每条记录的维度（int32，小端），返回：
      - dims_count: {维度: 该维度出现的条数}
      - n: 实际扫描到的记录条数
      - ok: 文件结构是否看起来正常（不会提前 EOF；每条记录的浮点数完整）
    如果 max_vectors 给定，只扫描前 max_vectors 条（用于大文件快速抽样检查）。
    """
    dims_count = Counter()
    n = 0
    ok = True

    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        while True:
            if max_vectors is not None and n >= max_vectors:
                break

            hdr = f.read(4)
            if not hdr:
                break  # 正常到达 EOF
            if len(hdr) != 4:
                ok = False
                break

            (d,) = struct.unpack("<i", hdr)
            if d < 0:
                ok = False
                break

            # 跳过 d 个 float32
            need = d * 4
            chunk = f.read(need)
            if len(chunk) != need:
                ok = False
                break

            dims_count[d] += 1
            n += 1

    # 再做一个粗略校验：如果维度唯一且为 d，则文件大小应是 (1+d)*4 的整数倍
    if ok and len(dims_count) == 1:
        d_unique = next(iter(dims_count.keys()))
        rec_bytes = (1 + d_unique) * 4
        if file_size % rec_bytes != 0:
            ok = False

    return dict(dims_count), n, ok


def load_fvecs_as_ndarray(path: str) -> np.ndarray:
    """
    在确认所有记录的维度一致时，将 fvecs 文件加载为形状 (n, d) 的 numpy.ndarray (float32)。
    采用内存映射 dtype=[('d','<i4'),('v','<f4',d)] 的方式高效读取。
    若维度不一致或文件结构异常将抛出 ValueError。
    """
    # 先读第一条记录的维度
    with open(path, "rb") as f:
        hdr = f.read(4)
        if len(hdr) != 4:
            raise ValueError("文件过小，无法读取第一条记录的维度。")
        (d,) = struct.unpack("<i", hdr)
        if d <= 0:
            raise ValueError(f"非法维度 d={d}。")

    file_size = os.path.getsize(path)
    rec_bytes = (1 + d) * 4
    if file_size % rec_bytes != 0:
        # 可能存在维度不一致，做全面扫描确认
        dims_count, n, ok = scan_fvecs_dims(path)
        raise ValueError(
            "文件大小与首条记录推断的记录长度不整除，"
            f"可能维度不一致或文件损坏；维度分布={dims_count}, 扫描条数={n}, ok={ok}"
        )

    n = file_size // rec_bytes
    # 采用结构化 dtype 进行内存映射：每条 (int32, float32[d])
    dt = np.dtype([("d", "<i4"), ("v", "<f4", d)])
    mm = np.memmap(path, dtype=dt, mode="r", shape=(n,))

    # 校验所有 d 是否一致
    if not np.all(mm["d"] == d):
        # 退一步给出分布信息
        unique, counts = np.unique(mm["d"], return_counts=True)
        dims_count = dict(zip(unique.tolist(), counts.tolist()))
        raise ValueError(f"检测到维度不一致：{dims_count}")

    # 返回 (n, d) 的普通 ndarray（如需懒加载可返回 mm['v'] 本身）
    return np.asarray(mm["v"])  # 复制到内存；若数据很大，可返回 mm["v"] 保持为 memmap 视图


def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = input("请输入 .fvecs 文件路径：").strip()

    if not os.path.isfile(path):
        print(f"错误：文件不存在：{path}")
        sys.exit(1)

    print(f"正在扫描维度分布：{path}")
    dims_count, n_scanned, ok = scan_fvecs_dims(path)
    total_bytes = os.path.getsize(path)

    print("—— 扫描结果 ——")
    print(f"文件大小（字节）：{total_bytes}")
    print(f"扫描到的记录条数：{n_scanned}")
    print(f"维度分布：{dims_count}")
    print(f"文件结构看起来{'正常' if ok else '异常/可疑'}")

    if len(dims_count) == 1 and ok:
        d = next(iter(dims_count.keys()))
        rec_bytes = (1 + d) * 4
        n_total = total_bytes // rec_bytes
        print(f"\n判定：所有向量维度一致，为 d={d}。")
        print(f"按一致维度推算的总记录数：{n_total}")

        # 尝试加载为 ndarray
        try:
            X = load_fvecs_as_ndarray(path)
            print(f"成功加载为 numpy 数组：形状 {X.shape}, dtype={X.dtype}")
        except Exception as e:
            print(f"注意：维度一致但加载失败：{e}")
    else:
        print("\n判定：存在维度不一致或文件结构异常。请检查数据来源或重新导出。")


if __name__ == "__main__":
    main()
