"""Walk-forward benchmark: naive vs LSTM (price) vs LSTM (returns) vs Chronos-2.

python -m smp.benchmark                 # full run, writes results/ and docs/
python -m smp.benchmark --no-chronos    # skip the Chronos-2 download
python -m smp.benchmark --end today     # score the latest data instead of 2020-2022
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data import FEATURES, TARGETS, Series, build_features, load_ohlcv
from .evaluate import fold_starts, prepare_fold, score, to_price
from .importance import permutation_importance
from .model import pick_device, predict, train_lstm

MODELS = ["Naive", "LSTM (price)", "LSTM (returns)", "Chronos-2"]


@dataclass
class Config:
    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "AMZN", "TSLA"])
    start: str = "2020-01-01"
    end: str = "2023-01-01"
    folds: int = 8
    test_days: int = 30
    seq_len: int = 5
    epochs: int = 150
    batch_size: int = 64
    lr: float = 1e-3
    hidden_size: int = 100
    seed: int = 0
    importance_repeats: int = 10
    chronos_model: str | None = "amazon/chronos-2"


def run_benchmark(
    frames: dict[str, pd.DataFrame], cfg: Config, chronos=None, log=print
) -> dict:
    """Score every model on every (ticker, fold); returns plain data for JSON."""
    rows: list[dict] = []
    importance: dict[str, list[list[float]]] = {}
    timings: dict[str, float] = {}
    last_fold: dict[str, dict[str, list[float]]] = {t: {} for t in cfg.tickers}
    device = pick_device()

    def record(model, series: Series, k, test_rows, pred, with_direction=True):
        actual, prev = series.close[test_rows], series.prev_close[test_rows]
        rows.append(
            {
                "model": model,
                "ticker": series.ticker,
                "fold": k,
                "test_from": str(series.dates[test_rows[0]].date()),
                "test_to": str(series.dates[test_rows[-1]].date()),
                **score(actual, pred, prev, with_direction),
            }
        )
        if k == cfg.folds - 1:
            last_fold[series.ticker][model] = [float(v) for v in pred]
            last_fold[series.ticker]["Actual"] = [float(v) for v in actual]
            last_fold[series.ticker]["dates"] = [
                str(series.dates[r].date()) for r in test_rows
            ]

    up_days: list[dict] = []
    for target in TARGETS:
        name = f"LSTM ({'price' if target == 'price' else 'returns'})"
        series = {t: build_features(frames[t], target, t) for t in cfg.tickers}
        # Each ticker's last fold ends on its own last row, so folds share dates
        starts = {
            t: fold_starts(len(s.close), cfg.folds, cfg.test_days, cfg.seq_len)
            for t, s in series.items()
        }
        importance[name] = []
        t0 = time.perf_counter()
        for k in range(cfg.folds):
            folds = {
                t: prepare_fold(s, starts[t][k], cfg.test_days, cfg.seq_len, target)
                for t, s in series.items()
            }
            x_train = np.concatenate([f.x_train for f in folds.values()])
            y_train = np.concatenate([f.y_train for f in folds.values()])
            model = train_lstm(
                x_train,
                y_train,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                hidden_size=cfg.hidden_size,
                seed=cfg.seed + k,
                device=device,
            )
            for t, f in folds.items():
                s = series[t]
                pred = to_price(
                    predict(model, f.x_test), f.scaler, s, f.test_rows, target
                )
                record(name, s, k, f.test_rows, pred)
                if target == "price":
                    prev = s.prev_close[f.test_rows]
                    record("Naive", s, k, f.test_rows, prev, with_direction=False)
                    up = float(np.mean(s.close[f.test_rows] > prev) * 100)
                    up_days.append({"ticker": t, "fold": k, "always_up": up})
            x_test = np.concatenate([f.x_test for f in folds.values()])
            y_test = np.concatenate([f.y_test for f in folds.values()])
            imp = permutation_importance(
                lambda x, m=model: predict(m, x),
                x_test,
                y_test,
                repeats=cfg.importance_repeats,
                seed=cfg.seed + k,
            )
            importance[name].append([float(v) for v in imp])
            log(
                f"{name} fold {k + 1}/{cfg.folds} done ({len(x_train)} training windows)"
            )
        timings[name] = time.perf_counter() - t0

        if target == "price" and chronos is not None:
            from .chronos_baseline import forecast_next_close

            t0 = time.perf_counter()
            for t, s in series.items():
                blocks = [np.arange(a, a + cfg.test_days) for a in starts[t]]
                # Context for day j is every close strictly before day j
                contexts = [
                    np.concatenate([s.prev_close[:1], s.close[:j]])
                    for rows_ in blocks
                    for j in rows_
                ]
                preds = forecast_next_close(chronos, contexts).reshape(cfg.folds, -1)
                for k, rows_ in enumerate(blocks):
                    record("Chronos-2", s, k, rows_, preds[k])
            timings["Chronos-2"] = time.perf_counter() - t0
            log("Chronos-2 done")

    return {
        "config": asdict(cfg),
        "rows": rows,
        "always_up": up_days,
        "importance": importance,
        "timings_s": timings,
        "last_fold": last_fold,
        "env": {
            "smp": __version__,
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "device": str(device),
            "run_date": str(date.today()),
        },
    }


def summarise(result: dict) -> pd.DataFrame:
    df = pd.DataFrame(result["rows"])
    agg = df.groupby(["ticker", "model"])[["rmse", "mape", "direction"]].agg(
        ["mean", "std"]
    )
    return agg


def _pm(mean: float, std: float, fmt: str) -> str:
    if np.isnan(mean):
        return "n/a"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def render_markdown(result: dict) -> str:
    cfg = result["config"]
    df = pd.DataFrame(result["rows"])
    models = [m for m in MODELS if m in set(df["model"])]
    tickers = cfg["tickers"]
    first = df[df["fold"] == 0]["test_from"].min()
    last = df[df["fold"] == cfg["folds"] - 1]["test_to"].max()
    lines = [
        "# Walk-forward benchmark results",
        "",
        (
            f"Generated by `python -m smp.benchmark` on {result['env']['run_date']} "
            f"(smp {result['env']['smp']}, Python {result['env']['python']}, "
            f"torch {result['env']['torch']}, {result['env']['device']})."
        ),
        "",
        (
            f"{cfg['folds']} expanding-window folds of {cfg['test_days']} trading days "
            f"per ticker, test days {first} to {last}. Each LSTM fold trains from "
            f"scratch for {cfg['epochs']} epochs (batch {cfg['batch_size']}, seed "
            f"{cfg['seed']} + fold) on all rows before its test block. "
            "Values are mean ± std across folds."
        ),
        "",
    ]

    def table(metric: str, title: str, fmt: str, unit: str, extra=None):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Stock | "
                + " | ".join(models + ([extra[0]] if extra else []))
                + " |",
            ]
        )
        lines.append("| --- |" + " ---: |" * (len(models) + (1 if extra else 0)))
        for t in tickers + ["All"]:
            sub = df if t == "All" else df[df["ticker"] == t]
            cells = []
            for m in models:
                vals = sub[sub["model"] == m][metric]
                cells.append(
                    _pm(vals.mean(), vals.std(), fmt)
                    + (unit if not np.isnan(vals.mean()) else "")
                )
            if extra:
                vals = extra[1](t)
                cells.append(_pm(np.mean(vals), np.std(vals, ddof=1), fmt) + unit)
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
        lines.append("")

    up = pd.DataFrame(result["always_up"])
    table("rmse", "RMSE (dollars, lower is better)", ".2f", "")
    table("mape", "MAPE (percent, lower is better)", ".2f", "%")
    table(
        "direction",
        "Direction accuracy (percent of days up/not-up called right, higher is better)",
        ".1f",
        "%",
        (
            "Always up",
            lambda t: (up if t == "All" else up[up["ticker"] == t])[
                "always_up"
            ].to_numpy(),
        ),
    )

    naive = df[df["model"] == "Naive"].set_index(["ticker", "fold"])["rmse"]
    total = len(naive)
    lines.extend(
        [
            "## Head to head against naive",
            "",
            "| Model | Stock-folds with lower RMSE than naive |",
            "| --- | ---: |",
        ]
    )
    for m in models:
        if m == "Naive":
            continue
        other = df[df["model"] == m].set_index(["ticker", "fold"])["rmse"]
        wins = int((other < naive.reindex(other.index)).sum())
        lines.append(f"| {m} | {wins} of {total} |")
    lines.append("")

    lines.extend(
        [
            "## Permutation importance",
            "",
            (
                "Rise in held-out MSE (scaled target units) when one input feature "
                f"is shuffled across test windows, {cfg['importance_repeats']} "
                "shuffles per fold, mean over folds. Higher means the model leans "
                "on that feature more."
            ),
            "",
            "| Feature | " + " | ".join(result["importance"]) + " |",
            "| --- |" + " ---: |" * len(result["importance"]),
        ]
    )
    imp = {k: np.mean(np.array(v), axis=0) for k, v in result["importance"].items()}
    for i, feat in enumerate(FEATURES):
        lines.append(
            f"| {feat} | " + " | ".join(f"{imp[k][i]:.4f}" for k in imp) + " |"
        )
    lines.append("")

    lines.extend(
        [
            "## Run time",
            "",
            "| Model | Seconds (all folds, all tickers) |",
            "| --- | ---: |",
        ]
    )
    for k, v in result["timings_s"].items():
        lines.append(f"| {k} | {v:.0f} |")
    lines.append("")
    return "\n".join(lines)


def render_charts(result: dict, docs: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(result["rows"])
    tickers = result["config"]["tickers"]
    models = [m for m in MODELS if m in set(df["model"])]
    colors = {
        "Naive": "#9aa0a6",
        "LSTM (price)": "#3b82f6",
        "LSTM (returns)": "#f59e0b",
        "Chronos-2": "#10b981",
    }
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=110)
    width = 0.8 / len(models)
    x = np.arange(len(tickers))
    for i, m in enumerate(models):
        sub = df[df["model"] == m].groupby("ticker")["rmse"]
        axes[0].bar(
            x + i * width,
            sub.mean().reindex(tickers),
            width,
            yerr=sub.std().reindex(tickers),
            label=m,
            color=colors[m],
            capsize=2,
        )
    axes[0].set_xticks(x + width * (len(models) - 1) / 2, tickers)
    axes[0].set_ylabel("RMSE ($), mean ± std over folds")
    axes[0].set_title("Next-day close error (lower is better)")
    axes[0].legend(frameon=False, fontsize=8)

    dir_models = [m for m in models if m != "Naive"]
    up = (
        pd.DataFrame(result["always_up"])
        .groupby("ticker")["always_up"]
        .mean()
        .reindex(tickers)
    )
    width = 0.8 / (len(dir_models) + 1)
    for i, m in enumerate(dir_models):
        sub = (
            df[df["model"] == m].groupby("ticker")["direction"].mean().reindex(tickers)
        )
        axes[1].bar(x + i * width, sub, width, label=m, color=colors[m])
    axes[1].bar(
        x + len(dir_models) * width, up, width, label="Always up", color="#d1d5db"
    )
    axes[1].axhline(50, color="#111827", lw=1, ls="--", label="Coin flip")
    axes[1].set_xticks(x + width * len(dir_models) / 2, tickers)
    axes[1].set_ylabel("Direction accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Up or not up tomorrow (higher is better)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout()
    paths.append(docs / "benchmark.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=110)
    for ax, t in zip(axes.ravel(), tickers, strict=False):
        lf = result["last_fold"][t]
        dates = pd.to_datetime(lf["dates"])
        ax.plot(dates, lf["Actual"], color="#111827", lw=2, label="Actual")
        for m in models:
            ax.plot(dates, lf[m], color=colors[m], lw=1.3, label=m)
        ax.set_title(f"{t}, last fold")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(docs / "last-fold.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    imp = {k: np.mean(np.array(v), axis=0) for k, v in result["importance"].items()}
    width = 0.8 / len(imp)
    x = np.arange(len(FEATURES))
    for i, (k, v) in enumerate(imp.items()):
        share = v / v.sum() * 100 if v.sum() > 0 else v
        ax.bar(x + i * width, share, width, label=k, color=colors[k])
    ax.set_xticks(x + width * (len(imp) - 1) / 2, FEATURES)
    ax.set_ylabel("Share of total MSE rise (%)")
    ax.set_title("Permutation importance on held-out windows")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths.append(docs / "feature-importance.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    d = Config()
    p.add_argument("--tickers", nargs="+", default=d.tickers)
    p.add_argument("--start", default=d.start)
    p.add_argument("--end", default=d.end, help="YYYY-MM-DD (exclusive) or 'today'")
    p.add_argument("--folds", type=int, default=d.folds)
    p.add_argument("--test-days", type=int, default=d.test_days)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--chronos-model", default=d.chronos_model)
    p.add_argument(
        "--no-chronos", action="store_true", help="skip the Chronos-2 baseline"
    )
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument(
        "--docs", type=Path, default=Path("docs"), help="where the PNG charts go"
    )
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument(
        "--from-json",
        type=Path,
        help="re-render benchmark.md and the charts from a saved benchmark.json",
    )
    a = p.parse_args(argv)

    if a.from_json:
        result = json.loads(a.from_json.read_text(encoding="utf-8"))
        write_reports(result, a.out, a.docs)
        return

    end = str(date.today()) if a.end == "today" else a.end
    cfg = Config(
        tickers=a.tickers,
        start=a.start,
        end=end,
        folds=a.folds,
        test_days=a.test_days,
        epochs=a.epochs,
        batch_size=a.batch_size,
        seed=a.seed,
        chronos_model=None if a.no_chronos else a.chronos_model,
    )
    frames = {t: load_ohlcv(t, cfg.start, cfg.end, a.cache) for t in cfg.tickers}
    chronos = None
    if cfg.chronos_model:
        from .chronos_baseline import load_pipeline

        chronos = load_pipeline(
            cfg.chronos_model, "cuda" if pick_device().type == "cuda" else "cpu"
        )

    result = run_benchmark(frames, cfg, chronos, log=lambda m: print(m, flush=True))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "benchmark.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    write_reports(result, a.out, a.docs)


def write_reports(result: dict, out: Path, docs: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(result)
    (out / "benchmark.md").write_text(md, encoding="utf-8", newline="\n")
    for path in render_charts(result, docs):
        print(f"wrote {path}")
    print(f"wrote {out / 'benchmark.md'}\n")
    print(md)


if __name__ == "__main__":
    main()
