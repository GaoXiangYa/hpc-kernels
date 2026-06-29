import sys
import re
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================ #
# Parse ONE benchmark txt file #
# ============================ #
def parse_gemv_file(filename):
    # Regex to extract: kernel, M, N, time(ms), gflops
    pattern = r"(\w+):\s*M\s*=\s*(\d+)\s*N\s*=\s*(\d+)\s*->\s*([\d.]+)\s*ms\s*([\d.]+)\s*GFlops"

    data = defaultdict(list)

    with open(filename, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                kernel = match.group(1)
                m = int(match.group(2))
                n = int(match.group(3))
                time = float(match.group(4))
                gflops = float(match.group(5))

                data[kernel].append((m, n, time, gflops))

    # Sort by size
    for k in data:
        data[k].sort(key=lambda x: x[0])
    return data


# ============================ #
# Merge all parsed data        #
# ============================ #
def merge_all_files(file_list):
    merged = defaultdict(list)

    for fname in file_list:
        print(f"[LOAD] {fname}")
        data = parse_gemv_file(fname)

        for kernel, rows in data.items():
            merged[kernel].extend(rows)

    # Sort all
    for k in merged:
        merged[k].sort(key=lambda x: x[0])

    return merged


# ============================ #
# Plotting                     #
# ============================ #
def plot_gemv_performance(data):
    plt.figure(figsize=(9, 6))

    for kernel, rows in data.items():
        sizes = [m for (m, _, _, _) in rows]
        gflops = [g for (_, _, _, g) in rows]

        plt.plot(sizes, gflops, marker="o", label=kernel)

        # peak annotation
        peak_idx = max(range(len(gflops)), key=lambda i: gflops[i])
        plt.annotate(
            f"{gflops[peak_idx]:.2f}",
            (sizes[peak_idx], gflops[peak_idx]),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xlabel("Matrix Size M (M=N)")
    plt.ylabel("GFLOPS")
    plt.title("GEMV Performance (merged)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================ #
# Main entry                   #
# ============================ #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_gemv.py file1.txt file2.txt ...")
        sys.exit(1)

    file_list = sys.argv[1:]

    data = merge_all_files(file_list)
    plot_gemv_performance(data)
