import argparse
import csv
import os
from datetime import datetime
from typing import Dict, Tuple, List


def merge_results(
    framing_csv: str | None,
    coverage_csv: str | None,
    faithfulness_csv: str | None,
    output_csv: str,
):
    framing_by_key: Dict[Tuple[str, str, str], dict] = {}
    coverage_by_key: Dict[Tuple[str, str, str], dict] = {}
    faithfulness_by_key: Dict[Tuple[str, str, str], dict] = {}

    if framing_csv and os.path.exists(framing_csv):
        with open(framing_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    (row.get("Genre") or "").strip(),
                    (row.get("Article Name") or "").strip(),
                    (row.get("Model") or "").strip(),
                )
                framing_by_key[key] = row

    if coverage_csv and os.path.exists(coverage_csv):
        with open(coverage_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    (row.get("Genre") or "").strip(),
                    (row.get("Article Name") or "").strip(),
                    (row.get("Model") or "").strip(),
                )
                coverage_by_key[key] = row

    if faithfulness_csv and os.path.exists(faithfulness_csv):
        with open(faithfulness_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    (row.get("Genre") or "").strip(),
                    (row.get("Article Name") or "").strip(),
                    (row.get("Model") or "").strip(),
                )
                faithfulness_by_key[key] = row

    all_keys = sorted(set(framing_by_key.keys()) | set(coverage_by_key.keys()) | set(faithfulness_by_key.keys()))

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Genre",
            "Article Name",
            "Model",
            "coverage_score",
            "covered_points",
            "partial_points",
            "missing_points",
            "total_key_points",
            "sentiment_shift",
            "keyword_overlap",
            "similarity",
            "framing_label",
            "missing_important_topics",
            "faithfulness_score",
            "supported",
            "contradictions",
            "neutral",
            "total_summary_sentences",
            "contradiction_rate",
            "sentence_level_nli",
            "point_results",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in all_keys:
            genre, article_name, model = key
            f_row = framing_by_key.get(key, {})
            c_row = coverage_by_key.get(key, {})
            h_row = faithfulness_by_key.get(key, {})

            writer.writerow(
                {
                    "Genre": genre,
                    "Article Name": article_name,
                    "Model": model,
                    "coverage_score": c_row.get("coverage_score", ""),
                    "covered_points": c_row.get("covered_points", ""),
                    "partial_points": c_row.get("partial_points", ""),
                    "missing_points": c_row.get("missing_points", ""),
                    "total_key_points": c_row.get("total_key_points", ""),
                    "sentiment_shift": f_row.get("sentiment_shift", ""),
                    "keyword_overlap": f_row.get("keyword_overlap", ""),
                    "similarity": f_row.get("similarity", ""),
                    "framing_label": f_row.get("framing_label", ""),
                    "missing_important_topics": f_row.get("missing_important_topics", ""),
                    "faithfulness_score": h_row.get("faithfulness_score", ""),
                    "supported": h_row.get("supported", ""),
                    "contradictions": h_row.get("contradictions", ""),
                    "neutral": h_row.get("neutral", ""),
                    "total_summary_sentences": h_row.get("total_summary_sentences", ""),
                    "contradiction_rate": h_row.get("contradiction_rate", ""),
                    "sentence_level_nli": h_row.get("sentence_level_nli", ""),
                    "point_results": c_row.get("point_results", ""),
                    "note": c_row.get("note", ""),
                }
            )

    print(f"Done. Wrote merged comparison to {output_csv}")


def filter_input_rows(
    input_csv: str,
    output_csv: str,
    article_name: str = "",
    max_articles: int = 0,
) -> int:
    with open(input_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        rows: List[dict] = list(reader)
        filtered = rows

        if article_name.strip():
            target = article_name.strip().lower()
            filtered = [r for r in filtered if (r.get("Article Name") or "").strip().lower() == target]
            if not filtered:
                raise ValueError(f"No rows found for article name: '{article_name}'")

        if max_articles > 0:
            filtered = filtered[:max_articles]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    return len(filtered)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full summarization + evaluation pipeline.")
    parser.add_argument(
        "--input",
        default="manually-labeling-news-articles.csv",
        help="Input CSV with article links and manual key points",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Output directory. Default creates timestamped folder under runs/",
    )
    parser.add_argument(
        "--covered-threshold",
        type=float,
        default=0.7,
        help="Coverage threshold for 'Covered'",
    )
    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=0.5,
        help="Coverage threshold for 'Partial'",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=220,
        help="Words per article chunk for faithfulness NLI premise selection",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=60,
        help="Word overlap between faithfulness premise chunks",
    )
    parser.add_argument(
        "--skip-framing",
        action="store_true",
        help="Skip framing analysis step",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip coverage analysis step",
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="Skip faithfulness analysis step",
    )
    parser.add_argument(
        "--article-name",
        default="",
        help="Run only one article by exact title match from 'Article Name'",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="Optional cap on number of input rows to run (0 = all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from summarize_articles_pipeline import run as run_summarization

    if args.outdir.strip():
        outdir = args.outdir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("runs", f"run_{stamp}")

    os.makedirs(outdir, exist_ok=True)

    summaries_csv = os.path.join(outdir, "article_summaries.csv")
    framing_csv = os.path.join(outdir, "framing_metrics.csv")
    coverage_csv = os.path.join(outdir, "coverage_metrics.csv")
    faithfulness_csv = os.path.join(outdir, "faithfulness_metrics.csv")
    merged_csv = os.path.join(outdir, "comparison_metrics.csv")
    pipeline_input_csv = args.input

    if args.article_name.strip() or args.max_articles > 0:
        pipeline_input_csv = os.path.join(outdir, "_filtered_input.csv")
        selected_count = filter_input_rows(
            input_csv=args.input,
            output_csv=pipeline_input_csv,
            article_name=args.article_name,
            max_articles=args.max_articles,
        )
        print(f"Selected {selected_count} article row(s) for this run.")

    print("Step 1/5: Summarization")
    run_summarization(input_csv=pipeline_input_csv, output_csv=summaries_csv)

    produced_framing_csv = framing_csv
    produced_coverage_csv = coverage_csv
    produced_faithfulness_csv = faithfulness_csv

    if args.skip_framing:
        produced_framing_csv = None
        print("Step 2/5: Framing analysis (skipped)")
    else:
        print("Step 2/5: Framing analysis")
        from framing_analysis import run as run_framing

        run_framing(input_csv=summaries_csv, output_csv=framing_csv)

    if args.skip_coverage:
        produced_coverage_csv = None
        print("Step 3/5: Coverage analysis (skipped)")
    else:
        print("Step 3/5: Coverage analysis")
        from coverage_analysis import run as run_coverage

        run_coverage(
            summaries_csv=summaries_csv,
            keypoints_csv=args.input,
            output_csv=coverage_csv,
            covered_threshold=args.covered_threshold,
            partial_threshold=args.partial_threshold,
        )

    if args.skip_faithfulness:
        produced_faithfulness_csv = None
        print("Step 4/5: Faithfulness analysis (skipped)")
    else:
        print("Step 4/5: Faithfulness analysis")
        from faithfulness_analysis import run as run_faithfulness

        run_faithfulness(
            input_csv=summaries_csv,
            output_csv=faithfulness_csv,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )

    print("Step 5/5: Merge comparison outputs")
    merge_results(
        framing_csv=produced_framing_csv,
        coverage_csv=produced_coverage_csv,
        faithfulness_csv=produced_faithfulness_csv,
        output_csv=merged_csv,
    )

    print("Pipeline complete. Outputs:")
    print(f"- {summaries_csv}")
    if produced_framing_csv:
        print(f"- {framing_csv}")
    if produced_coverage_csv:
        print(f"- {coverage_csv}")
    if produced_faithfulness_csv:
        print(f"- {faithfulness_csv}")
    print(f"- {merged_csv}")


if __name__ == "__main__":
    main()
