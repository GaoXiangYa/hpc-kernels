import re
import sys
import matplotlib.pyplot as plt

def parse_file(filename):
    """
    解析 gemm_xxx.csv 文件，提取 (m, GFLOPS)
    返回两个列表：m_values, gflops_values
    """
    m_values = []
    gflops_values = []
    pattern = re.compile(r"\[m, n, k\]:\s+(\d+).*GFLOPS:\s+([0-9.]+)")

    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                m = int(match.group(1))
                gflops = float(match.group(2))
                m_values.append(m)
                gflops_values.append(gflops)
    return m_values, gflops_values


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_gemm.py gemm_v0 [gemm_v1 ...]")
        sys.exit(1)

    plt.figure(figsize=(8, 6))

    for arg in sys.argv[1:]:
        filename = f"{arg}"
        m_values, gflops_values = parse_file(filename)
        plt.plot(m_values, gflops_values, marker='o', label=arg)

    plt.xlabel("Matrix size (m=n=k)")
    plt.ylabel("GFLOPS")
    plt.title("GEMM Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存 PNG 文件
    output_file = "gemm_perf.png"
    plt.savefig(output_file, dpi=300)
    print(f"图像已保存为 {output_file}")

    # 显示图像
    plt.show()


if __name__ == "__main__":
    main()
