"""Dump useful stats about a parametric-capture run directory.

A run directory is produced/consumed by the scripts in this folder and may
contain any subset of the following artifacts:

    cv_samples.npy                (N, 4) cv_a, cv_b, morph, amplitude
    cv_buffers.z                  zarr (N*L, 4) sent buffers; nchunks == N captures
    capture_buffers.z             zarr (N*L, 4) received buffers (ch0 morph_out, ch3 triangle)
    model_data.z                  zarr (N*L, 5) x0..x3, y_true (training join)
    audio_stats.pkl               list[dict] rms, spectral_centroid, thd, flatness, odd_even
    candidate_generation_stats.txt  text report from candidate generation
    cv_buffers/ capture_buffers/   pre-packed per-capture .npy (older format)
    plots/                        per-capture jpgs

Only the artifacts that exist are reported.

    uv run run_stats.py --run runs/001
    uv run run_stats.py --run runs/001 runs/002
"""

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024
    return f"{x:.1f}TB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _section(title: str) -> None:
    print(f"\n-- {title} --")


STATS = ("mean", "std")
PCT_STATS = tuple(f"p{p}" for p in range(0, 101, 10))


@dataclass
class Section:
    """A comparable block of per-label stats for a single artifact."""

    title: str
    header: str | None = None  # descriptive line (shape / dtype / size / counts)
    stats: list = field(default_factory=list)  # stat column names
    rows: list = field(default_factory=list)  # [(label, {stat: float})]
    text: list = field(default_factory=list)  # extra freeform lines (e.g. corr)
    pct_stats: list = field(default_factory=list)  # p0..p100 column names
    pct_rows: list = field(default_factory=list)  # [(label, {pct: float})]


def _col_stats(arr: np.ndarray, labels, stats=STATS):
    """Return (stats, rows, pct_stats, pct_rows) where rows is [(label, {stat: value})]."""

    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    rows = []
    pct_rows = []
    pct_percentiles = list(range(0, 101, 10))
    pct_keys = list(PCT_STATS)
    for c, label in enumerate(labels):
        col = arr[:, c]
        vals = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
        }
        rows.append((str(label), {s: vals[s] for s in stats if s in vals}))
        pct_vals_arr = np.percentile(col, pct_percentiles)
        pct_rows.append(
            (str(label), {k: float(v) for k, v in zip(pct_keys, pct_vals_arr)})
        )
    return list(stats), rows, pct_keys, pct_rows


def resolve_run_dir(run: Path) -> Path:
    """Accept either 'runs/001', '001', or an absolute path."""
    candidates = [run, Path("runs") / run]
    for c in candidates:
        if c.exists():
            return c
    # default to the first; downstream reporting will note it is empty
    return run


# ---------------------------------------------------------------------------
# extraction: each returns a Section (or None if the artifact is absent)
# ---------------------------------------------------------------------------

BUFFERS = [
    ("cv_buffers.z", ["cv_a", "cv_b", "morph", "voct_sweep"]),
    ("capture_buffers.z", ["morph_out", "ch1", "ch2", "triangle"]),
    (
        "model_data.z",
        ["x0_triangle", "x1_cv_a", "x2_cv_b", "x3_morph", "y_true_morph_out"],
    ),
]

SECTION_ORDER = ["cv_samples.npy", "losses.tsv", "audio_stats.pkl"] + [
    n for n, _ in BUFFERS
]


def extract_cv_samples(run_dir: Path):
    path = run_dir / "cv_samples.npy"
    if not path.exists():
        return None
    samples = np.load(path)
    if samples.ndim == 2 and samples.shape[1] == 4:
        labels = ["cv_a", "cv_b", "morph", "amplitude"]
    else:
        labels = [
            f"col_{i}" for i in range(samples.shape[1] if samples.ndim == 2 else 1)
        ]
    stats, rows, pct_stats, pct_rows = _col_stats(samples, labels)
    header = (
        f"shape {samples.shape}  dtype {samples.dtype}  "
        f"({_fmt_bytes(path.stat().st_size)})"
    )
    return Section(
        "cv_samples.npy", header, stats, rows, pct_stats=pct_stats, pct_rows=pct_rows
    )


def extract_losses(run_dir: Path):
    path = run_dir / "losses.tsv"
    if not path.exists():
        return None
    rows_raw = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if header is None and any(not _is_float(p) for p in parts):
                header = parts
                continue
            rows_raw.append([float(p) for p in parts])
    if not rows_raw:
        return Section("losses.tsv", "(empty)")
    data = np.array(rows_raw)
    if header is None or len(header) != data.shape[1]:
        header = [f"col_{i}" for i in range(data.shape[1])]
    stats, rows, pct_stats, pct_rows = _col_stats(data, header)
    text = []
    if data.shape[1] > 1:
        corr = np.corrcoef(data, rowvar=False)
        width = max(len(h) for h in header)
        text.append("correlation:")
        text.append(f"{'':<{width}}  " + " ".join(f"{h:>10}" for h in header))
        for i, h in enumerate(header):
            text.append(
                f"{h:<{width}}  "
                + " ".join(f"{corr[i, j]:>10.3f}" for j in range(len(header)))
            )
    return Section(
        "losses.tsv",
        f"{len(data)} rows (one per capture)",
        stats,
        rows,
        text,
        pct_stats=pct_stats,
        pct_rows=pct_rows,
    )


def extract_audio_stats(run_dir: Path):
    path = run_dir / "audio_stats.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        records = pickle.load(f)
    if not records:
        return Section("audio_stats.pkl", "(empty)")
    keys = list(records[0].keys())
    data = np.array([[r[k] for k in keys] for r in records], dtype=float)
    stats, rows, pct_stats, pct_rows = _col_stats(data, keys)
    return Section(
        "audio_stats.pkl",
        f"{len(records)} records (ch0 of each capture)",
        stats,
        rows,
        pct_stats=pct_stats,
        pct_rows=pct_rows,
    )


def extract_buffer(run_dir: Path, name: str, labels, max_captures: int):
    path = run_dir / name
    if not path.exists():
        return None
    try:
        import zarr
    except ImportError:
        return Section(name, "(zarr not installed)")

    z = zarr.open(str(path), mode="r")
    n_captures = z.nchunks
    per_capture = z.chunks[0]
    if z.shape[1] != len(labels):
        labels = [f"ch_{i}" for i in range(z.shape[1])]

    # streaming over every capture would read GBs; sample evenly instead
    sample_idx = np.unique(
        np.linspace(0, n_captures - 1, min(max_captures, n_captures)).astype(int)
    )
    sampled = np.concatenate([np.asarray(z.blocks[b]) for b in sample_idx], axis=0)
    stats, rows, pct_stats, pct_rows = _col_stats(sampled, labels)

    # clipping is meaningful for audio: percentage of samples at +/-1.0
    clip = np.mean(np.abs(sampled) >= 0.999, axis=0) * 100.0
    stats = list(stats) + ["clip%"]
    rows = [
        (lbl, {**vals, "clip%": float(clip[c])}) for c, (lbl, vals) in enumerate(rows)
    ]
    header = (
        f"shape {z.shape}  dtype {z.dtype}  captures {n_captures}  "
        f"samples/capture {per_capture}  ({_fmt_bytes(_dir_size(path))})  "
        f"[{len(sample_idx)} sampled]"
    )
    return Section(name, header, stats, rows, pct_stats=pct_stats, pct_rows=pct_rows)


def extract_sections(run_dir: Path, max_captures: int) -> dict:
    """Return {title: Section|None} keyed by SECTION_ORDER."""
    secs = {
        "cv_samples.npy": extract_cv_samples(run_dir),
        "losses.tsv": extract_losses(run_dir),
        "audio_stats.pkl": extract_audio_stats(run_dir),
    }
    for name, labels in BUFFERS:
        secs[name] = extract_buffer(run_dir, name, labels, max_captures)
    return secs


def inventory_lines(run_dir: Path) -> list:
    known = [
        "cv_samples.npy",
        "cv_buffers.z",
        "capture_buffers.z",
        "model_data.z",
        "losses.tsv",
        "audio_stats.pkl",
        "candidate_generation_stats.txt",
        "cv_samples.density.jpg",
        "plots",
        "cv_buffers",
        "capture_buffers",
    ]
    present = [k for k in known if (run_dir / k).exists()]
    missing = [k for k in known if k not in present]
    return [
        f"total size: {_fmt_bytes(_dir_size(run_dir))}",
        f"present: {', '.join(present) or '(none)'}",
        f"missing: {', '.join(missing) or '(none)'}",
    ]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _print_grid(stats, rows) -> None:
    width = max([len(lbl) for lbl, _ in rows] + [6])
    print(f"  {'column':<{width}}  " + " ".join(f"{s:>10}" for s in stats))
    for lbl, vals in rows:
        cells = " ".join(
            f"{vals[s]:>10.4f}" if s in vals else f"{'-':>10}" for s in stats
        )
        print(f"  {lbl:<{width}}  {cells}")


def render_single_section(sec) -> None:
    if sec is None:
        return
    _section(sec.title)
    if sec.header:
        print(f"  {sec.header}")
    if sec.rows:
        _print_grid(sec.stats, sec.rows)
    if sec.pct_rows:
        print("  percentiles:")
        _print_grid(sec.pct_stats, sec.pct_rows)
    for line in sec.text:
        print(f"  {line}")


def render_compare_section(a, b, name_a: str, name_b: str) -> None:
    if a is None and b is None:
        return
    title = (a or b).title
    _section(title)
    print(f"  [{name_a}] " + ((a.header if a and a.header else None) or "(absent)"))
    print(f"  [{name_b}] " + ((b.header if b and b.header else None) or "(absent)"))

    # if only one side has data, fall back to a plain grid for that side
    if a is None or b is None:
        present = a or b
        if present and present.rows:
            _print_grid(present.stats, present.rows)
        if present and present.pct_rows:
            print("  percentiles:")
            _print_grid(present.pct_stats, present.pct_rows)
        for line in present.text if present else []:
            print(f"  {line}")
        return

    # union of labels / stats (preserve a's order, append b-only extras)
    labels = [lbl for lbl, _ in a.rows]
    for lbl, _ in b.rows:
        if lbl not in labels:
            labels.append(lbl)
    stats = a.stats if a.stats == b.stats else list(dict.fromkeys(a.stats + b.stats))

    a_map = dict(a.rows)
    b_map = dict(b.rows)
    col_w = max(12, len(name_a), len(name_b))
    metric_w = max([len(f"{lbl}.{s}") for lbl in labels for s in stats] + [6])
    print(
        f"  {'metric':<{metric_w}}  "
        f"{name_a:>{col_w}} {name_b:>{col_w}} {'delta':>{col_w}}"
    )
    for lbl in labels:
        for s in stats:
            va = a_map.get(lbl, {}).get(s)
            vb = b_map.get(lbl, {}).get(s)
            sa = f"{va:>{col_w}.4f}" if va is not None else f"{'-':>{col_w}}"
            sb = f"{vb:>{col_w}.4f}" if vb is not None else f"{'-':>{col_w}}"
            if va is not None and vb is not None:
                sd = f"{vb - va:>+{col_w}.4f}"
            else:
                sd = f"{'-':>{col_w}}"
            print(f"  {lbl + '.' + s:<{metric_w}}  {sa} {sb} {sd}")

    # percentile comparison
    pct_labels = (
        [lbl for lbl, _ in a.pct_rows]
        if a and a.pct_rows
        else ([lbl for lbl, _ in b.pct_rows] if b and b.pct_rows else [])
    )
    if pct_labels:
        pct_stats = (a or b).pct_stats
        a_pct_map = dict(a.pct_rows) if a and a.pct_rows else {}
        b_pct_map = dict(b.pct_rows) if b and b.pct_rows else {}
        for lbl, _ in (b.pct_rows if b and b.pct_rows else []):
            if lbl not in pct_labels:
                pct_labels.append(lbl)
        metric_w_pct = max(
            [len(f"{lbl}.{s}") for lbl in pct_labels for s in pct_stats] + [6]
        )
        print(f"\n  percentiles:")
        print(
            f"  {'metric':<{metric_w_pct}}  "
            f"{name_a:>{col_w}} {name_b:>{col_w}} {'delta':>{col_w}}"
        )
        for lbl in pct_labels:
            for s in pct_stats:
                va = a_pct_map.get(lbl, {}).get(s)
                vb = b_pct_map.get(lbl, {}).get(s)
                sa = f"{va:>{col_w}.4f}" if va is not None else f"{'-':>{col_w}}"
                sb = f"{vb:>{col_w}.4f}" if vb is not None else f"{'-':>{col_w}}"
                sd = (
                    f"{vb - va:>+{col_w}.4f}"
                    if va is not None and vb is not None
                    else f"{'-':>{col_w}}"
                )
                print(f"  {lbl + '.' + s:<{metric_w_pct}}  {sa} {sb} {sd}")

    # correlation (and any other freeform text) is shown per run, labeled
    for run_name, sec in ((name_a, a), (name_b, b)):
        if sec.text:
            print(f"  correlation [{run_name}]:")
            for line in sec.text[1:]:  # skip the "correlation:" label line
                print(f"    {line}")


def report_misc(run_dir: Path) -> None:
    # unpacked per-capture buffers (older format)
    for name in ("cv_buffers", "capture_buffers"):
        d = run_dir / name
        if d.is_dir():
            n = len(list(d.glob("*.npy")))
            _section(f"{name}/ (unpacked)")
            print(f"  {n} per-capture .npy files ({_fmt_bytes(_dir_size(d))})")

    plots = run_dir / "plots"
    if plots.is_dir():
        jpgs = list(plots.glob("*.jpg"))
        _section("plots/")
        kinds = {}
        for p in jpgs:
            kind = ".".join(p.name.split(".")[1:-1]) or "other"
            kinds[kind] = kinds.get(kind, 0) + 1
        print(f"  {len(jpgs)} jpg files ({_fmt_bytes(_dir_size(plots))})")
        for kind, count in sorted(kinds.items()):
            print(f"    {kind}: {count}")

    cand = run_dir / "candidate_generation_stats.txt"
    if cand.exists():
        _section("candidate_generation_stats.txt")
        text = cand.read_text()
        # first couple of lines carry the opts + source runs
        for line in text.splitlines()[:3]:
            print(f"  {line}")
        print(
            f"  ... ({len(text.splitlines())} lines, {_fmt_bytes(cand.stat().st_size)})"
        )


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def render_single(run_dir: Path, max_captures: int) -> None:
    print(f"================ run stats: {run_dir} ================")
    if not run_dir.exists():
        print("  (directory does not exist)")
        return
    for line in inventory_lines(run_dir):
        print(f"  {line}")

    secs = extract_sections(run_dir, max_captures)
    for title in SECTION_ORDER:
        render_single_section(secs[title])
    report_misc(run_dir)


def render_compare(run_a: Path, run_b: Path, max_captures: int) -> None:
    name_a, name_b = str(run_a), str(run_b)
    print(f"================ run comparison: {name_a}  vs  {name_b} ================")

    for run_dir, name in ((run_a, name_a), (run_b, name_b)):
        label = f"[{name}]"
        print(f"  {label}")
        if not run_dir.exists():
            print("    (directory does not exist)")
            continue
        for line in inventory_lines(run_dir):
            print(f"    {line}")

    secs_a = extract_sections(run_a, max_captures) if run_a.exists() else {}
    secs_b = extract_sections(run_b, max_captures) if run_b.exists() else {}
    for title in SECTION_ORDER:
        render_compare_section(secs_a.get(title), secs_b.get(title), name_a, name_b)

    for run_dir, name in ((run_a, name_a), (run_b, name_b)):
        if run_dir.exists():
            print(f"\n######## misc: {name} ########")
            report_misc(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        nargs="+",
        help="runs dirs to check",
    )
    parser.add_argument(
        "--max-captures",
        type=int,
        default=64,
        help="number of captures sampled when summarising large zarr buffers",
    )
    opts = parser.parse_args()

    if len(opts.run) not in (1, 2):
        parser.error("--run takes one or two run dirs")

    run_dirs = [resolve_run_dir(r) for r in opts.run]
    if len(run_dirs) == 1:
        render_single(run_dirs[0], opts.max_captures)
    else:
        render_compare(run_dirs[0], run_dirs[1], opts.max_captures)


if __name__ == "__main__":
    main()
