import argparse
import csv
import json
import re
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def detect_summary_columns(fieldnames: List[str]) -> Tuple[str, List[str], str]:
    normalized = {normalize_column(c): c for c in fieldnames}

    article_name_col = normalized.get("article_name") or normalized.get("article")
    if not article_name_col:
        raise ValueError("Could not find article name column (expected 'Article Name').")

    genre_col = normalized.get("genre", "Genre")

    summary_cols = []
    for norm_name, original in normalized.items():
        if "summary" in norm_name:
            summary_cols.append(original)

    if not summary_cols:
        raise ValueError("No summary columns found. Expected columns like 'GPT Summary', 'Gemini Summary'.")

    return article_name_col, summary_cols, genre_col


def parse_key_points(raw: str) -> List[str]:
    if not raw:
        return []

    # Accept either " | " separated points or newline/semicolon separated lists.
    chunks = []
    if "|" in raw:
        chunks = [p.strip() for p in raw.split("|")]
    elif "\n" in raw:
        chunks = [p.strip() for p in raw.splitlines()]
    elif ";" in raw:
        chunks = [p.strip() for p in raw.split(";")]
    else:
        chunks = re.split(r"(?<=[.!?])\s+", raw.strip())

    return [p for p in chunks if p]


def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    # Lightweight sentence splitter without extra runtime downloads.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def label_similarity(score: float, covered_threshold: float, partial_threshold: float) -> str:
    if score >= covered_threshold:
        return "Covered"
    if score >= partial_threshold:
        return "Partial"
    return "Missing"


def coverage_score(labels: List[str]) -> float:
    if not labels:
        return 0.0

    points = 0.0
    for label in labels:
        if label == "Covered":
            points += 1.0
        elif label == "Partial":
            points += 0.5

    return points / len(labels)


def load_keypoints_csv(path: str) -> Dict[Tuple[str, str], List[str]]:
    keypoints: Dict[Tuple[str, str], List[str]] = {}

    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Key points CSV has no header.")

        normalized = {normalize_column(c): c for c in reader.fieldnames}
        article_col = normalized.get("article_name")
        if not article_col:
            raise ValueError("Key points CSV needs an 'Article Name' column.")

        genre_col = normalized.get("genre")
        if not genre_col:
            for norm_name, original in normalized.items():
                if norm_name.startswith("genre"):
                    genre_col = original
                    break
        keypoints_col = (
            normalized.get("key_points")
            or normalized.get("keypoints")
            or normalized.get("points")
            or normalized.get("key_summary,_notes,_and_takeaways")
        )
        if not keypoints_col:
            # Fallback: any column containing both "key" and "summary".
            for norm_name, original in normalized.items():
                if "key" in norm_name and "summary" in norm_name:
                    keypoints_col = original
                    break

        if not keypoints_col:
            raise ValueError(
                "Key points CSV needs a key-point column (e.g., 'Key Points' or 'Key Summary, Notes, and Takeaways')."
            )

        for row in reader:
            article_name = (row.get(article_col) or "").strip()
            genre = (row.get(genre_col) or "").strip() if genre_col else ""
            points = parse_key_points(row.get(keypoints_col, ""))
            key = (genre.lower(), article_name.lower())
            keypoints[key] = points

    return keypoints


def run(
    summaries_csv: str,
    keypoints_csv: str,
    output_csv: str,
    covered_threshold: float,
    partial_threshold: float,
):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    keypoint_map = load_keypoints_csv(keypoints_csv)

    output_rows = []

    with open(summaries_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Summaries CSV has no header.")

        article_col, summary_cols, genre_col = detect_summary_columns(reader.fieldnames)

        for row in reader:
            article_name = (row.get(article_col) or "").strip()
            genre = (row.get(genre_col) or "").strip() if genre_col in row else ""

            key = (genre.lower(), article_name.lower())
            key_points = keypoint_map.get(key)

            if key_points is None:
                # fallback: article name only
                key_points = keypoint_map.get(("", article_name.lower()), [])

            for summary_col in summary_cols:
                summary_text = row.get(summary_col, "") or ""
                summary_sentences = split_sentences(summary_text)

                if not key_points:
                    output_rows.append(
                        {
                            "Genre": genre,
                            "Article Name": article_name,
                            "Model": summary_col.replace(" Summary", ""),
                            "coverage_score": "",
                            "covered_points": "",
                            "partial_points": "",
                            "missing_points": "",
                            "total_key_points": 0,
                            "point_results": json.dumps([], ensure_ascii=True),
                            "note": "No key points found for this article",
                        }
                    )
                    continue

                if not summary_sentences:
                    labels = ["Missing"] * len(key_points)
                    point_results = [
                        {
                            "key_point": kp,
                            "best_sentence": "",
                            "max_similarity": 0.0,
                            "status": "Missing",
                        }
                        for kp in key_points
                    ]
                else:
                    key_emb = model.encode(key_points)
                    sent_emb = model.encode(summary_sentences)
                    sim_matrix = cosine_similarity(key_emb, sent_emb)

                    labels = []
                    point_results = []
                    for i, key_point in enumerate(key_points):
                        row_sim = sim_matrix[i]
                        best_idx = int(row_sim.argmax())
                        best_sim = float(row_sim[best_idx])
                        status = label_similarity(best_sim, covered_threshold, partial_threshold)
                        labels.append(status)

                        point_results.append(
                            {
                                "key_point": key_point,
                                "best_sentence": summary_sentences[best_idx],
                                "max_similarity": round(best_sim, 4),
                                "status": status,
                            }
                        )

                covered = sum(1 for l in labels if l == "Covered")
                partial = sum(1 for l in labels if l == "Partial")
                missing = sum(1 for l in labels if l == "Missing")

                output_rows.append(
                    {
                        "Genre": genre,
                        "Article Name": article_name,
                        "Model": summary_col.replace(" Summary", ""),
                        "coverage_score": round(coverage_score(labels), 4),
                        "covered_points": covered,
                        "partial_points": partial,
                        "missing_points": missing,
                        "total_key_points": len(key_points),
                        "point_results": json.dumps(point_results, ensure_ascii=True),
                        "note": "",
                    }
                )

            print(f"Coverage analyzed: {article_name}")

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
            "point_results",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Done. Wrote {len(output_rows)} rows to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Coverage evaluation using sentence embeddings and cosine similarity")
    parser.add_argument("--summaries", default="article_summaries.csv", help="CSV produced by summarize_articles_pipeline.py")
    parser.add_argument(
        "--keypoints",
        default="manually-labeling-news-articles.csv",
        help="CSV containing article key points (defaults to your manually-labeled source CSV)",
    )
    parser.add_argument("--output", default="coverage_metrics.csv", help="Output CSV path")
    parser.add_argument("--covered-threshold", type=float, default=0.7, help="Similarity threshold for Covered")
    parser.add_argument("--partial-threshold", type=float, default=0.5, help="Similarity threshold for Partial")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        summaries_csv=args.summaries,
        keypoints_csv=args.keypoints,
        output_csv=args.output,
        covered_threshold=args.covered_threshold,
        partial_threshold=args.partial_threshold,
    )
