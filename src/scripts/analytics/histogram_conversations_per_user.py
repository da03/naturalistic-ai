#!/usr/bin/env python3
import argparse
import os
import sys
import json
from collections import Counter
from typing import Dict, Iterable, Tuple, List

import pandas as pd
import numpy as np
import ijson
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def stream_conversations(json_path: str) -> Iterable[Dict]:
    """
    Stream conversations from a large JSON file without loading it fully.
    Supports two layouts:
      - {"dataset_id": ..., "data": [ {conversation objects...} ]}
      - [ {conversation objects...} ]
    """
    with open(json_path, "rb") as f:
        # Peek to decide whether top-level is object or list
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if event == 'start_map' and prefix == '':
                # object with possibly "data" array
                break
            if event == 'start_array' and prefix == '':
                # rewind and iterate array items
                f.seek(0)
                for item in ijson.items(f, 'item'):
                    yield item
                return
        # If here, top-level is an object; we need to iterate data array
        f.seek(0)
        for item in ijson.items(f, 'data.item'):
            yield item


def compute_user_counts(json_path: str) -> Counter:
    user_counts: Counter = Counter()
    for conv in stream_conversations(json_path):
        user_id = conv.get('user_id')
        if user_id is None:
            continue
        user_counts[user_id] += 1
    return user_counts


def build_histogram(user_counts: Counter) -> pd.DataFrame:
    # value_counts of counts: how many users have N conversations
    count_series = pd.Series(list(user_counts.values()))
    hist = count_series.value_counts().sort_index()
    df = hist.rename_axis('num_conversations').reset_index(name='num_users')
    return df


def save_outputs(user_counts: Counter, hist_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Save per-user counts
    user_counts_path = os.path.join(out_dir, 'user_conversation_counts.csv')
    pd.DataFrame({
        'user_id': list(user_counts.keys()),
        'num_conversations': list(user_counts.values())
    }).to_csv(user_counts_path, index=False)

    # Save histogram
    hist_path = os.path.join(out_dir, 'histogram_conversations_per_user.csv')
    hist_df.to_csv(hist_path, index=False)


def _apply_log_formatting(ax, x_log: bool, y_log: bool):
    if x_log:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}" if x >= 1 else f"{x:g}"))
    if y_log:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{int(y):,}" if y >= 1 else f"{y:g}"))


def maybe_plot(hist_df: pd.DataFrame, out_dir: str, max_x: int = None, y_log: bool = False, x_log: bool = False, grid: bool = False):
    if hist_df.empty:
        return
    plt.figure(figsize=(10, 6))
    x = hist_df['num_conversations']
    y = hist_df['num_users']
    if max_x is not None:
        mask = x <= max_x
        x = x[mask]
        y = y[mask]
    plt.bar(x, y, width=0.9, align='center')
    ax = plt.gca()
    _apply_log_formatting(ax, x_log=x_log, y_log=y_log)
    if grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlabel('Conversations per user')
    plt.ylabel('Number of users')
    plt.title('Histogram of conversations per user')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'histogram_conversations_per_user.png')
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ccdf(hist_df: pd.DataFrame, out_dir: str, y_log: bool = True, x_log: bool = False, grid: bool = False):
    if hist_df.empty:
        return
    df = hist_df.sort_values('num_conversations')
    counts = df['num_users'].to_numpy()
    total = counts.sum()
    # Tail cumulative sum: P(X >= x)
    tail = np.cumsum(counts[::-1])[::-1] / float(total)
    x = df['num_conversations'].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x, tail, marker='o', linestyle='-')
    ax = plt.gca()
    _apply_log_formatting(ax, x_log=x_log, y_log=y_log)
    if grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xlabel('Conversations per user (k)')
    plt.ylabel('P(X ≥ k)')
    plt.title('CCDF of conversations per user')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'ccdf_conversations_per_user.png')
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize(user_counts: Counter, tail_percentiles: List[float]) -> Tuple[int, int, float, Dict[str, float], Dict[str, float], np.ndarray]:
    if not user_counts:
        return 0, 0, 0.0, {}, {}, np.array([], dtype=np.int64)
    values = np.array(list(user_counts.values()), dtype=np.int64)
    mn = int(values.min())
    mx = int(values.max())
    avg = float(values.mean())
    percentiles = {
        'p50': float(np.percentile(values, 50)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
    }
    tail_pct = {f"p{p}": float(np.percentile(values, p)) for p in tail_percentiles}
    return mn, mx, avg, percentiles, tail_pct, values


def save_ccdf_tail(values: np.ndarray, thresholds: List[int], out_dir: str) -> pd.DataFrame:
    n = int(values.size)
    rows = []
    for k in thresholds:
        c = int((values >= k).sum())
        frac = (c / n) if n else 0.0
        rows.append({'threshold': int(k), 'num_users': c, 'fraction': frac})
    df = pd.DataFrame(rows).sort_values('threshold')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'tail_ccdf_conversations_per_user.csv'), index=False)
    return df


def _parse_percentile_list(s: str) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='Compute histogram of conversations per user from a large dataset JSON (streaming).')
    parser.add_argument('--input', required=True, help='Path to input JSON (e.g., datasets/wildchat_private/full.json)')
    parser.add_argument('--out-dir', default='res/analytics', help='Output directory for CSVs and optional plot')
    parser.add_argument('--plot', action='store_true', help='Save a PNG plot of the histogram')
    parser.add_argument('--max-x', type=int, default=None, help='Optional max x-axis for the plot (e.g., 200)')
    parser.add_argument('--y-log', action='store_true', help='Use log scale for y-axis')
    parser.add_argument('--x-log', action='store_true', help='Use log scale for x-axis')
    parser.add_argument('--grid', action='store_true', help='Show gridlines on plots')
    parser.add_argument('--ccdf', action='store_true', help='Also save a CCDF plot (P[X ≥ k])')
    parser.add_argument('--tail-percentiles', type=str, default='99.0,99.1,99.2,99.5,99.8,99.9,99.95,99.99', help='Comma-separated list of high-percentiles to report (e.g., 99.1,99.2,99.5)')
    parser.add_argument('--tail-thresholds', type=str, default='1,2,3,4,5,6,7,8,9,10,20,50,100,200,500,1000', help='Comma-separated list of k to report CCDF >= k')
    args = parser.parse_args()

    user_counts = compute_user_counts(args.input)
    hist_df = build_histogram(user_counts)
    save_outputs(user_counts, hist_df, args.out_dir)

    if args.plot:
        maybe_plot(hist_df, args.out_dir, max_x=args.max_x, y_log=args.y_log, x_log=args.x_log, grid=args.grid)
        if args.ccdf:
            plot_ccdf(hist_df, args.out_dir, y_log=args.y_log, x_log=args.x_log, grid=args.grid)

    tail_percentiles = _parse_percentile_list(args.tail_percentiles)
    mn, mx, avg, percentiles, tail_pct, values = summarize(user_counts, tail_percentiles)

    thresholds = _parse_int_list(args.tail_thresholds)
    ccdf_df = save_ccdf_tail(values, thresholds, args.out_dir)

    # Print a short human-readable CCDF summary
    for row in ccdf_df.itertuples(index=False):
        print(f">= {int(row.threshold)}: {row.fraction:.6f} ({int(row.num_users):,})")

    # Compose JSON summary
    summary_obj = {
        'num_users': len(user_counts),
        'min_conversations_per_user': mn,
        'max_conversations_per_user': mx,
        'avg_conversations_per_user': avg,
        'percentiles': percentiles,
        'tail_percentiles': tail_pct,
        'suggested_min_conversations': max(1, int(percentiles.get('p50', 1))),
        'suggested_max_conversations': int(percentiles.get('p95', mx)),
        'ccdf_tail': [
            {'threshold': int(t), 'fraction': float(f), 'num_users': int(c)}
            for t, f, c in zip(ccdf_df['threshold'], ccdf_df['fraction'], ccdf_df['num_users'])
        ]
    }

    print(json.dumps(summary_obj, indent=2))


if __name__ == '__main__':
    main() 