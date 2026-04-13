#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pstats
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS / "optimization_report_20260413"


@dataclass(frozen=True)
class Attempt:
    key: str
    label: str
    status: str
    timing_file: str | None
    summary: str
    detail: list[str]
    verification_note: str


ATTEMPTS: list[Attempt] = [
    Attempt(
        key="baseline",
        label="Baseline",
        status="reference",
        timing_file="full_case_timings.json",
        summary="Deterministic full-scale seeded baseline before optimization.",
        detail=[
            "Official workload is the six full figure runs used by the baseline generator.",
            "Acceptance criterion for every later optimization was exact seeded numerical regression preservation, not visual similarity.",
            "PNG figures remained human-readable artifacts only; official regression stayed on numerical history arrays stored in versioned NPZ files.",
        ],
        verification_note="Used as the reference timing and numerical baseline.",
    ),
    Attempt(
        key="opt1",
        label="Opt1",
        status="accepted",
        timing_file="full_case_timings_post_opt1.json",
        summary="Remove repeated data motion and clean up velocity assignment paths.",
        detail=[
            "Added Particle.assign_vel() and Particle.update_moments() to separate raw velocity assignment from derived moment refresh.",
            "Removed repeated concatenate-based assembly in the collision update path and wrote into a preallocated vstar_a buffer.",
            "Replaced argsort-based row restoration with restore_rows_from_map(), which uses direct scatter into a preallocated output array.",
            "Cached probabilistic acceptance factors once per get_vstar() call instead of recomputing them inside every subcycle.",
        ],
        verification_note="Fast suite and full verification both passed after the patch.",
    ),
    Attempt(
        key="opt2",
        label="Opt2",
        status="accepted",
        timing_file="full_case_timings_post_opt2.json",
        summary="Reuse self-collision workspaces and remove avoidable fancy indexing.",
        detail=[
            "Introduced _get_like_workspace() so like-collision temporary Particle and Collision objects are reused instead of rebuilt every call.",
            "Replaced np.arange-based subcycle indexing with simple slice objects where the collision schedule is contiguous.",
            "Kept the seeded numerical path intact by reusing the same RNG stream and the same collision ordering semantics.",
        ],
        verification_note="Fast suite and full verification both passed after the patch.",
    ),
    Attempt(
        key="opt3",
        label="Opt3",
        status="rejected",
        timing_file="full_case_timings_post_opt3.json",
        summary="Selective small-s and large-s branch handling in evaluate_cosChi()/get_A().",
        detail=[
            "Attempted to reduce unnecessary transcendental work by handling small-s and large-s cases more selectively.",
            "The candidate preserved the seeded numerical baseline but improved less than the best retained branch and increased total runtime versus later retained states.",
            "The path was therefore not kept.",
        ],
        verification_note="Numerical regression held, but official workload timing was not competitive.",
    ),
    Attempt(
        key="opt3b",
        label="Opt3b",
        status="rejected",
        timing_file="full_case_timings_post_opt3b.json",
        summary="Micro-optimization of get_h() vector assembly.",
        detail=[
            "Restructured orthogonal-vector assembly in get_h() to reduce temporary work in the hot path.",
            "This also preserved the seeded numerical baseline.",
            "End-to-end timing still stayed slower than the best retained branch, so it was reverted.",
        ],
        verification_note="Numerical regression held, but official workload timing was not competitive.",
    ),
    Attempt(
        key="opt4",
        label="Opt4",
        status="accepted-best",
        timing_file="full_case_timings_post_opt4.json",
        summary="Remove a redundant final moment recomputation in get_vstar().",
        detail=[
            "After each subcycle, spb.update_moments() was already executed to keep species-B moments current.",
            "The extra final spb.update_moments() at the end of get_vstar() repeated work that was already done on the final subcycle.",
            "Removing that one redundant full-array scan preserved all seeded baselines and produced the best retained runtime of the campaign.",
        ],
        verification_note="Fast suite and full verification both passed; this became the retained best state.",
    ),
    Attempt(
        key="opt5",
        label="Opt5",
        status="rejected",
        timing_file="full_case_timings_post_opt5_candidate.json",
        summary="In-place row restoration into the original velocity buffer.",
        detail=[
            "The idea was to eliminate one output allocation in restore_rows_from_map() by scattering directly into an existing buffer.",
            "The candidate preserved the seeded baseline.",
            "Measured end-to-end runtime became worse, so it was rejected.",
        ],
        verification_note="Numerical regression held, but official workload timing regressed.",
    ),
    Attempt(
        key="opt6",
        label="Opt6",
        status="rejected",
        timing_file="full_case_timings_post_opt6_candidate.json",
        summary="Simplify scatter term assembly inside get_vPrime().",
        detail=[
            "Refactored the vPrime scattering expression to reduce intermediate arrays in the collision update.",
            "The candidate preserved seeded regression behavior.",
            "Wall time still regressed against the retained branch, so it was dropped.",
        ],
        verification_note="Numerical regression held, but official workload timing regressed.",
    ),
    Attempt(
        key="opt7",
        label="Opt7",
        status="rejected",
        timing_file="full_case_timings_post_opt7_candidate.json",
        summary="Incremental moment updates instead of full-array recomputation.",
        detail=[
            "Maintained sum(v) and sum(v^2) state in the hot loop and updated only the touched slices.",
            "This looked attractive from a complexity perspective because update_moments() was a major hotspot.",
            "In practice, extra Python-level bookkeeping outweighed the saved reductions and total runtime became significantly worse.",
        ],
        verification_note="Numerical regression held, but official workload timing regressed strongly.",
    ),
    Attempt(
        key="opt8",
        label="Opt8",
        status="rejected",
        timing_file="full_case_timings_post_opt8_candidate.json",
        summary="Hot-path velocity access simplification.",
        detail=[
            "Tested a more aggressive hot-path simplification around velocity access and intermediate handling.",
            "The candidate improved over the original baseline, but it still stayed slower than Opt4.",
            "Because the optimization goal was best retained end-to-end runtime under exact regression preservation, it was not kept.",
        ],
        verification_note="Numerical regression held, but official workload timing remained slower than Opt4.",
    ),
]


TEXT_ONLY_REJECTS = [
    {
        "name": "Prefactor precompute in Collision.__init__",
        "reason": "A representative full-scale figure case showed seeded baseline drift, so the idea was rejected before full campaign timing.",
    },
    {
        "name": "Shared g_norm reuse between get_h() and evaluate_s_ab()",
        "reason": "The change altered the numerical path enough to break exact seeded regression, so it was reverted before being promoted to a timed candidate.",
    },
]


HOTSPOT_FUNCS = [
    ("get_vstar", "collision.py"),
    ("get_vPrime", "collision.py"),
    ("like_collision_update", "collision.py"),
    ("unlike_collision_update", "collision.py"),
    ("evaluate_cosChi", "collision.py"),
    ("update_moments", "particle.py"),
    ("get_A", "collision.py"),
    ("evaluate_s_ab", "collision.py"),
    ("get_h", "collision.py"),
    ("shuffle_rows_with_map", "collision.py"),
    ("restore_rows_from_map", "collision.py"),
]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    result = []
    for char in text:
        result.append(replacements.get(char, char))
    return "".join(result)


def load_timing_file(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_all_timings() -> dict[str, list[dict]]:
    timings: dict[str, list[dict]] = {}
    for attempt in ATTEMPTS:
        if attempt.timing_file is not None:
            timings[attempt.key] = load_timing_file(ARTIFACTS / attempt.timing_file)
    return timings


def total_seconds(entries: list[dict]) -> float:
    return float(sum(item["seconds"] for item in entries))


def case_map(entries: list[dict]) -> dict[str, dict]:
    return {item["name"]: item for item in entries}


def current_git_identity() -> tuple[str, str, str]:
    import subprocess

    def run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = run(["git", "rev-parse", "--short", "HEAD"])
    describe = run(["git", "describe", "--always", "--dirty"])
    return branch, head, describe


def extract_hotspots(stats_path: Path) -> tuple[float, list[dict]]:
    stats = pstats.Stats(str(stats_path))
    rows = []
    for func, stat in stats.stats.items():
        filename, line, name = func
        if not (filename.endswith("collision.py") or filename.endswith("particle.py")):
            continue
        for target_name, target_file in HOTSPOT_FUNCS:
            if name == target_name and filename.endswith(target_file):
                cc, nc, tt, ct, callers = stat
                rows.append(
                    {
                        "name": name,
                        "file": Path(filename).name,
                        "line": line,
                        "ncalls": nc,
                        "tottime": float(tt),
                        "cumtime": float(ct),
                    }
                )
                break
    rows.sort(key=lambda item: item["cumtime"], reverse=True)
    return float(stats.total_tt), rows


def build_total_timing_plot(timings: dict[str, list[dict]], out_path: Path) -> None:
    labels = []
    totals = []
    colors = []
    for attempt in ATTEMPTS:
        if attempt.timing_file is None:
            continue
        labels.append(attempt.label)
        totals.append(total_seconds(timings[attempt.key]))
        if "accepted" in attempt.status:
            colors.append("#1f77b4")
        elif attempt.status == "reference":
            colors.append("#444444")
        else:
            colors.append("#d95f02")

    plt.figure(figsize=(10.5, 4.8))
    xpos = np.arange(len(labels))
    bars = plt.bar(xpos, totals, color=colors)
    plt.xticks(xpos, labels, rotation=30, ha="right")
    plt.ylabel("Total wall time (s)")
    plt.title("Full official workload timing by attempt")
    for bar, total in zip(bars, totals):
        plt.text(bar.get_x() + bar.get_width() / 2.0, total + 4.0, f"{total:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_case_breakdown_plot(baseline: list[dict], best: list[dict], out_path: Path) -> None:
    base_map = case_map(baseline)
    best_map = case_map(best)
    names = list(base_map.keys())
    base_vals = [base_map[name]["seconds"] for name in names]
    best_vals = [best_map[name]["seconds"] for name in names]

    plt.figure(figsize=(10.8, 5.0))
    xpos = np.arange(len(names))
    width = 0.38
    plt.bar(xpos - width / 2, base_vals, width=width, label="Baseline", color="#666666")
    plt.bar(xpos + width / 2, best_vals, width=width, label="Opt4 best", color="#1f77b4")
    plt.xticks(xpos, names, rotation=25, ha="right")
    plt.ylabel("Seconds")
    plt.title("Per-case timing: baseline vs final retained best state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_hotspot_plot(total_tt: float, rows: list[dict], out_path: Path) -> None:
    top = rows[:8]
    labels = [row["name"] for row in reversed(top)]
    values = [row["cumtime"] for row in reversed(top)]
    pct = [100.0 * value / total_tt for value in values]

    plt.figure(figsize=(9.5, 5.2))
    ypos = np.arange(len(labels))
    plt.barh(ypos, values, color="#4c78a8")
    plt.yticks(ypos, labels)
    plt.xlabel("Cumulative time (s)")
    plt.title("Dominant hotspots from diagnostic cProfile snapshot")
    for y, value, share in zip(ypos, values, pct):
        plt.text(value + 2.0, y, f"{value:.1f}s ({share:.1f}%)", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def paragraph(lines: list[str]) -> str:
    return "\n\n".join(lines)


def format_attempt_row(attempt: Attempt, timings: dict[str, list[dict]], base_total: float) -> str:
    if attempt.timing_file is None:
        total = "-"
        gain = "-"
    else:
        total_value = total_seconds(timings[attempt.key])
        total = f"{total_value:.3f}"
        gain = f"{(base_total - total_value) / base_total * 100.0:.2f}\\%"
    return (
        f"{latex_escape(attempt.label)} & "
        f"{latex_escape(attempt.status)} & "
        f"{total} & "
        f"{gain} & "
        f"{latex_escape(attempt.summary)} \\\\"
    )


def build_tex(timings: dict[str, list[dict]], hotspot_total_tt: float, hotspots: list[dict]) -> str:
    branch, head, describe = current_git_identity()
    baseline = timings["baseline"]
    best = timings["opt4"]
    baseline_total = total_seconds(baseline)
    best_total = total_seconds(best)
    improvement_pct = (baseline_total - best_total) / baseline_total * 100.0
    baseline_map = case_map(baseline)
    best_map = case_map(best)

    official_workload_rows = []
    for item in baseline:
        official_workload_rows.append(
            f"{latex_escape(item['name'])} & {item['markers_total']:,} & {item['iterations']} & {item['seconds']:.3f} \\\\"
        )

    attempt_rows = [format_attempt_row(attempt, timings, baseline_total) for attempt in ATTEMPTS]

    per_case_rows = []
    for name, base_item in baseline_map.items():
        best_item = best_map[name]
        saved = base_item["seconds"] - best_item["seconds"]
        pct = 100.0 * saved / base_item["seconds"]
        per_case_rows.append(
            f"{latex_escape(name)} & {base_item['seconds']:.3f} & {best_item['seconds']:.3f} & {saved:.3f} & {pct:.2f}\\% \\\\"
        )

    hotspot_rows = []
    for row in hotspots:
        share = 100.0 * row["cumtime"] / hotspot_total_tt
        hotspot_rows.append(
            f"{latex_escape(row['name'])} & {latex_escape(row['file'])}:{row['line']} & {row['ncalls']:,} & {row['cumtime']:.3f} & {share:.2f}\\% \\\\"
        )

    detail_sections = []
    for attempt in ATTEMPTS[1:]:
        if attempt.timing_file is not None:
            total = total_seconds(timings[attempt.key])
            saved = baseline_total - total
            pct = (saved / baseline_total) * 100.0
            metrics = f"Official workload total: {total:.3f}s. Improvement vs baseline: {saved:.3f}s ({pct:.2f}\\%)."
        else:
            metrics = "No retained timing artifact."
        bullets = "\n".join([f"\\item {latex_escape(line)}" for line in attempt.detail])
        detail_sections.append(
            "\n".join(
                [
                    f"\\subsection*{{{latex_escape(attempt.label)}: {latex_escape(attempt.summary)}}}",
                    latex_escape(metrics),
                    "",
                    f"Verification note: {latex_escape(attempt.verification_note)}",
                    "",
                    "\\begin{itemize}",
                    bullets,
                    "\\end{itemize}",
                ]
            )
        )

    text_only_rows = "\n".join(
        f"\\item \\textbf{{{latex_escape(item['name'])}}}: {latex_escape(item['reason'])}"
        for item in TEXT_ONLY_REJECTS
    )

    fig6_baseline = baseline_map["fig6_equal"]["seconds"] + baseline_map["fig6_weighted"]["seconds"]
    values = {
        "__BRANCH__": latex_escape(branch),
        "__DESCRIBE__": latex_escape(describe),
        "__BASELINE_TOTAL__": f"{baseline_total:.3f}",
        "__BEST_TOTAL__": f"{best_total:.3f}",
        "__IMPROVEMENT_PCT__": f"{improvement_pct:.2f}",
        "__OFFICIAL_WORKLOAD_ROWS__": "\n".join(official_workload_rows),
        "__FIG6_TOTAL__": f"{fig6_baseline:.3f}",
        "__FIG6_SHARE__": f"{fig6_baseline / baseline_total * 100.0:.2f}",
        "__HOTSPOT_TOTAL_TT__": f"{hotspot_total_tt:.3f}",
        "__HOTSPOT_ROWS__": "\n".join(hotspot_rows),
        "__ATTEMPT_ROWS__": "\n".join(attempt_rows),
        "__DETAIL_SECTIONS__": "\n".join(detail_sections),
        "__TEXT_ONLY_ROWS__": text_only_rows,
        "__TOTAL_SAVED__": f"{baseline_total - best_total:.3f}",
        "__PER_CASE_ROWS__": "\n".join(per_case_rows),
    }

    template = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\usepackage{float}
\usepackage{titlesec}
\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}
\titleformat{\section}{\large\bfseries}{\thesection}{0.5em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.5em}{}

\begin{document}

\begin{center}
{\LARGE binary-coulomb-collision Optimization Campaign Report}\\[0.5em]
{\large Full-detail summary of all optimization attempts and final retained state}\\[0.8em]
Generated on 2026-04-13 for branch \texttt{__BRANCH__}, git identity \texttt{__DESCRIBE__}
\end{center}

\section*{Executive Summary}
This report summarizes the full optimization campaign performed on the \texttt{binary-coulomb-collision} codebase while keeping the numerical behavior fixed under the seeded full-scale baseline test. The campaign goal was simple: improve runtime without changing the official seeded numerical relaxation histories used for regression. The best retained state combines three accepted changesets, here called Opt1, Opt2, and Opt4. The retained best total official workload time dropped from __BASELINE_TOTAL__s to __BEST_TOTAL__s, a reduction of __IMPROVEMENT_PCT__\%.

The official benchmark was the six full-size figure cases used to generate the baseline relaxation histories. Acceptance required exact seeded numerical regression preservation on the official verification suite. Human-readable PNG figures remained inspection artifacts only and were not treated as the formal regression criterion.

\section{Benchmark Definition and Acceptance Rules}
The optimization campaign used the same canonical six full-scale cases that underpin the numerical verification baseline. Each timing file in \texttt{artifacts/} records the exact per-case wall time for one full run of those cases.

\subsection{Official workload}
\begin{table}[H]
\centering
\small
\begin{tabular}{lrrr}
\toprule
Case & Total markers & Iterations & Baseline wall time (s) \\
\midrule
__OFFICIAL_WORKLOAD_ROWS__
\bottomrule
\end{tabular}
\end{table}

The total baseline time across all six cases was __BASELINE_TOTAL__s. The workload is dominated by the Fig.6 cases; in the baseline timing they consumed __FIG6_TOTAL__s, which is __FIG6_SHARE__\% of the total.

\subsection{Acceptance rules}
\begin{itemize}
\item The formal correctness criterion was exact seeded numerical reproduction of the baseline NPZ histories.
\item The fast gate remained \texttt{pytest -q -m "not verification"}.
\item The formal regression gate remained \texttt{pytest -q -m verification}.
\item Performance was judged by the official six-case workload totals stored in versioned timing JSON artifacts.
\item A candidate was kept only if it preserved seeded regression and improved the official workload total relative to the best retained state.
\end{itemize}

\section{Profiling Evidence}
The most informative diagnostic profile snapshot used during the campaign is \texttt{artifacts/full\_baseline\_profile\_refresh\_20260411.pstats}. It corresponds to the post-Opt2 code path that directly guided the later Opt4 cleanup. Its total profiled time was __HOTSPOT_TOTAL_TT__s.

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{hotspots_diagnostic.png}
\caption{Dominant hotspots from the diagnostic cProfile snapshot used to guide the remaining work.}
\end{figure}

\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrc}
\toprule
Function & Location & Calls & Cum. time (s) & Share of profile \\
\midrule
__HOTSPOT_ROWS__
\bottomrule
\end{tabular}
\end{table}

The profile made the high-level structure clear. Most of the time remained inside \texttt{get\_vstar()}, \texttt{get\_vPrime()}, seeded scattering-angle evaluation, repeated moment recomputation, and row shuffle/restore data motion. That is why the campaign focused first on structural overhead removal rather than trying to micro-optimize isolated transcendental expressions.

\section{Attempt Summary}
\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{timing_totals.png}
\caption{Total official workload time by attempt. Blue bars were retained at least temporarily. Orange bars were evaluated and rejected.}
\end{figure}

\begin{table}[H]
\centering
\small
\begin{tabular}{>{\raggedright\arraybackslash}p{1.0in}>{\raggedright\arraybackslash}p{1.0in}rr>{\raggedright\arraybackslash}p{2.7in}}
\toprule
Attempt & Status & Total (s) & Gain vs baseline & Summary \\
\midrule
__ATTEMPT_ROWS__
\bottomrule
\end{tabular}
\end{table}

\section{Detailed Attempt Log}
__DETAIL_SECTIONS__

\subsection*{Pre-screened ideas rejected before full timing promotion}
\begin{itemize}
__TEXT_ONLY_ROWS__
\end{itemize}

\section{Final Retained State}
The final retained branch state is the composition of Opt1, Opt2, and Opt4:
\begin{itemize}
\item Opt1 removed repeated data motion and clarified the raw velocity assignment path.
\item Opt2 reused like-collision scratch workspaces and replaced avoidable fancy indexing with slices.
\item Opt4 removed a redundant final full-array moment recomputation at the end of \texttt{get\_vstar()}.
\end{itemize}

Together these changes reduced the official workload total from __BASELINE_TOTAL__s to __BEST_TOTAL__s, saving __TOTAL_SAVED__s overall. This is a __IMPROVEMENT_PCT__\% reduction in wall time while preserving the seeded numerical baseline.

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{case_breakdown_opt4.png}
\caption{Per-case timing comparison between the original baseline and the final retained best state.}
\end{figure}

\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Case & Baseline (s) & Opt4 best (s) & Saved (s) & Improvement \\
\midrule
__PER_CASE_ROWS__
\bottomrule
\end{tabular}
\end{table}

The largest absolute savings came from the Fig.6 cases, especially \texttt{fig6\_equal}, because those cases dominate the official workload. Structural overhead removal therefore mattered more than local algebraic rewrites.

\section{Key Lessons from the Campaign}
\begin{itemize}
\item The winning changes were structural: less allocation, less object churn, less repeated whole-array work.
\item Several mathematically reasonable micro-optimizations preserved seeded results but still made the end-to-end workload slower.
\item Reducing Python-level orchestration overhead in self-collision handling was materially more valuable than rewriting individual vector formulas.
\item Exact seeded numerical regression was a strong filter. Multiple ideas that looked attractive on paper were rejected because they either drifted numerically or failed to beat the retained best state.
\end{itemize}

\section{Current Bottlenecks and Rational Next Steps}
Even after the retained optimizations, the dominant costs are still concentrated in the collision hot path: \texttt{get\_vstar()}, \texttt{get\_vPrime()}, scattering-angle evaluation, moment recomputation, and row shuffle/restore movement. If more work is done later, the next rational steps are:
\begin{itemize}
\item Re-profile the current Opt4 best state directly and confirm the remaining shares on the final retained code.
\item Focus on whole-array work that still scales with every subcycle, especially moment refresh and shuffle/restore traffic.
\item Treat expression-level rewrites as secondary unless they produce a clear end-to-end gain on the full official workload.
\item Keep the seeded numerical baseline as the non-negotiable acceptance gate for any future performance work.
\end{itemize}

\section{Evidence Files Used for This Report}
\begin{itemize}
\item Timing artifacts: \texttt{artifacts/full\_case\_timings*.json}
\item Diagnostic profile: \texttt{artifacts/full\_baseline\_profile\_refresh\_20260411.pstats}
\item Current best-state code paths: \texttt{binary\_collision/collision.py} and \texttt{binary\_collision/particle.py}
\item Baseline and verification infrastructure: \texttt{scripts/generate\_baselines.py} and \texttt{tests/test\_relaxation\_figures\_verification.py}
\end{itemize}

\end{document}
"""
    for key, value in values.items():
        template = template.replace(key, value)
    return template


def write_report() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings = load_all_timings()
    hotspot_total_tt, hotspots = extract_hotspots(ARTIFACTS / "full_baseline_profile_refresh_20260411.pstats")

    build_total_timing_plot(timings, OUTPUT_DIR / "timing_totals.png")
    build_case_breakdown_plot(timings["baseline"], timings["opt4"], OUTPUT_DIR / "case_breakdown_opt4.png")
    build_hotspot_plot(hotspot_total_tt, hotspots, OUTPUT_DIR / "hotspots_diagnostic.png")

    tex = build_tex(timings, hotspot_total_tt, hotspots)
    tex_path = OUTPUT_DIR / "optimization_report.tex"
    tex_path.write_text(tex)
    return tex_path


def main() -> None:
    tex_path = write_report()
    print(tex_path)


if __name__ == "__main__":
    main()
