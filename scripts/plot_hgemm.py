#!/usr/bin/env python3
"""Generate an SVG line chart comparing HGEMM kernel performance (no dependencies).

Usage:
  ./build/src/cuda/hgemm/hgemm_benchmark | python3 scripts/plot_hgemm.py
  python3 scripts/plot_hgemm.py build/hgemm_bench.csv
"""

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# SVG chart builder — pure stdlib (no matplotlib, no numpy)
# ---------------------------------------------------------------------------

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def read_csv(source) -> dict:
    data: dict[str, list] = {}
    for row in csv.DictReader(source):
        ver = row["version"].strip()
        size = int(row["M"])
        gflops = float(row["gflops"])
        data.setdefault(ver, []).append((size, gflops))
    for ver in data:
        data[ver].sort(key=lambda p: p[0])
    return data


def svg_chart(data: dict, out_path: Path, width=900, height=550):
    """Draw a multi-line chart as standalone SVG and save to *out_path*."""
    # ---- margins ----
    left, right, top, bottom = 80, 40, 40, 60
    pw = width - left - right
    ph = height - top - bottom

    # ---- data ranges ----
    all_sizes = set()
    all_gflops = set()
    for pts in data.values():
        for x, y in pts:
            all_sizes.add(x)
            all_gflops.add(y)
    x_min, x_max = min(all_sizes), max(all_sizes)
    x_range = max(x_max - x_min, 1)
    y_max = max(all_gflops) * 1.08
    y_min = 0

    def xf(v):
        return left + (v - x_min) / x_range * pw

    def yf(v):
        return top + ph - (v / y_max) * ph

    # ---- SVG helpers ----
    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
    )
    svg.append(
        '<style>.grid{stroke:#444;stroke-dasharray:4,4;stroke-width:0.8}'
        '.axis{stroke:#888;stroke-width:1.2}'
        '.tick{font:11px monospace;fill:#bbb}'
        '</style>'
    )

    # ---- background ----
    svg.append(f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>')

    # ---- grid + y-axis labels ----
    n_ticks = 8
    for i in range(n_ticks + 1):
        yv = y_max * i / n_ticks
        y = yf(yv)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
        svg.append(f'<text class="tick" x="{left-8}" y="{y+4:.1f}" text-anchor="end">{yv:.0f}</text>')

    # x-axis labels
    for size in [x for x in all_sizes if x % 512 == 0 or x == x_min]:
        svg.append(
            f'<text class="tick" x="{xf(size):.1f}" y="{height-15}" text-anchor="middle">{size}</text>'
        )
        svg.append(
            f'<line class="grid" x1="{xf(size):.1f}" y1="{top}" x2="{xf(size):.1f}" y2="{top+ph}"/>'
        )

    # ---- axes ----
    svg.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>')
    svg.append(f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>')

    # ---- axis labels ----
    svg.append(
        f'<text x="{left+pw/2:.0f}" y="{height-4}" '
        'text-anchor="middle" font-size="13" fill="#aaa">M = N = K</text>'
    )
    svg.append(
        f'<text x="20" y="{top+ph/2:.0f}" text-anchor="middle" '
        'font-size="13" fill="#aaa" transform="rotate(-90,20,'
        f'{top+ph/2:.0f})">GFLOPS</text>'
    )

    # ---- title ----
    svg.append(
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" '
        'font-size="17" font-weight="bold" fill="#eee">'
        'GEMM Kernel Performance Comparison</text>'
    )

    # ---- data lines ----
    for i, (ver, pts) in enumerate(sorted(data.items())):
        color = COLORS[i % len(COLORS)]
        coords = " ".join(f"{xf(x):.1f},{yf(y):.1f}" for x, y in pts)
        svg.append(
            f'<polyline points="{coords}" fill="none" '
            f'stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>'
        )
        # markers
        for x, y in pts:
            svg.append(
                f'<circle cx="{xf(x):.1f}" cy="{yf(y):.1f}" r="3.5" '
                f'fill="{color}" stroke="#1a1a2e" stroke-width="1"/>'
            )

    # ---- legend ----
    lx, ly = left + pw - 160, top + 10
    for i, ver in enumerate(sorted(data)):
        color = COLORS[i % len(COLORS)]
        svg.append(f'<rect x="{lx}" y="{ly + i*22}" width="16" height="12" fill="{color}" rx="2"/>')
        svg.append(
            f'<text x="{lx+22}" y="{ly + i*22 + 11}" '
            f'font-size="13" fill="#ddd">{ver}</text>'
        )

    svg.append("</svg>")

    out_path.write_text("\n".join(svg))
    print(f"Saved → {out_path}")


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path) as fh:
            data = read_csv(fh)
    elif not sys.stdin.isatty():
        data = read_csv(sys.stdin)
    else:
        csv_path = (
            Path(__file__).resolve().parent.parent / "build" / "hgemm_bench.csv"
        )
        if not csv_path.exists():
            print(
                "Pipe benchmark output here:\n"
                "    ./build/src/cuda/hgemm/hgemm_benchmark | python3 scripts/plot_hgemm.py",
                file=sys.stderr,
            )
            sys.exit(1)
        with open(csv_path) as fh:
            data = read_csv(fh)

    if not data:
        print("Error: no data to plot", file=sys.stderr)
        sys.exit(1)

    out = Path(__file__).resolve().parent.parent / "docs" / "hgemm_perf.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    svg_chart(data, out)


if __name__ == "__main__":
    main()
