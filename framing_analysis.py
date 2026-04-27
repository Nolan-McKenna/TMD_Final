import argparse
import csv
from typing import Dict, List, Tuple

import nltk
from keybert import KeyBERT
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class NeutralSentimentAnalyzer:
    def polarity_scores(self, _text: str) -> Dict[str, float]:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}


def get_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        # Try a best-effort download, then gracefully degrade if SSL/network blocks it.
        try:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as exc:
            print(
                "Warning: VADER lexicon unavailable; sentiment shift will default to neutral (0.0). "
                f"Reason: {exc}"
            )
            return NeutralSentimentAnalyzer()


def normalize_column(name: str) -> str:
    return "_".join(name.strip().lower().split())


def extract_keywords(kw_model: KeyBERT, text: str, top_n: int = 10) -> List[str]:
    if not text or not text.strip():
        return []

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n,
    )
    return [kw for kw, _score in keywords]


def keyword_overlap(article_keywords: List[str], summary_keywords: List[str]) -> Tuple[float, List[str]]:
    article_set = {k.lower() for k in article_keywords}
    summary_set = {k.lower() for k in summary_keywords}

    if not article_set:
        return 0.0, []

    overlap = len(article_set.intersection(summary_set)) / len(article_set)
    missing = sorted(list(article_set.difference(summary_set)))
    return overlap, missing


def framing_label(sentiment_shift: float, overlap: float) -> str:
    abs_shift = abs(sentiment_shift)

    if abs_shift <= 0.10 and overlap >= 0.70:
        return "No shift"
    if abs_shift <= 0.25 and overlap >= 0.40:
        return "Slight shift"
    return "Major shift"


def detect_columns(fieldnames: List[str]) -> Dict[str, str]:
    normalized = {normalize_column(c): c for c in fieldnames}

    article_text_col = None
    for candidate in ["article_text", "article"]:
        if candidate in normalized:
            article_text_col = normalized[candidate]
            break

    if not article_text_col:
        raise ValueError("Could not find article text column (expected 'Article Text' or 'article_text').")

    summary_cols = []
    for norm_name, original in normalized.items():
        if "summary" in norm_name:
            summary_cols.append(original)

    if not summary_cols:
        raise ValueError("No summary columns found. Expected columns like 'GPT Summary', 'Gemini Summary'.")

    return {"article_text": article_text_col, "summary_cols": summary_cols}


def run(input_csv: str, output_csv: str):
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    sentiment_analyzer = get_sentiment_analyzer()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    output_rows = []

    with open(input_csv, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")

        col_config = detect_columns(reader.fieldnames)
        article_col = col_config["article_text"]
        summary_cols = col_config["summary_cols"]

        for row in reader:
            article_text = row.get(article_col, "")
            article_name = row.get("Article Name", row.get("article_name", ""))
            genre = row.get("Genre", row.get("genre", ""))

            article_keywords = extract_keywords(kw_model, article_text, top_n=10)
            article_sentiment = sentiment_analyzer.polarity_scores(article_text)["compound"]
            article_embedding = embed_model.encode(article_text)

            for summary_col in summary_cols:
                summary_text = row.get(summary_col, "")
                summary_keywords = extract_keywords(kw_model, summary_text, top_n=10)
                summary_sentiment = sentiment_analyzer.polarity_scores(summary_text)["compound"]
                summary_embedding = embed_model.encode(summary_text)

                overlap, missing_topics = keyword_overlap(article_keywords, summary_keywords)
                sentiment_shift = summary_sentiment - article_sentiment
                similarity = cos_sim(article_embedding, summary_embedding).item()
                label = framing_label(sentiment_shift, overlap)

                output_rows.append(
                    {
                        "Genre": genre,
                        "Article Name": article_name,
                        "Model": summary_col.replace(" Summary", ""),
                        "sentiment_shift": round(sentiment_shift, 4),
                        "keyword_overlap": round(overlap, 4),
                        "similarity": round(float(similarity), 4),
                        "framing_label": label,
                        "article_keywords": " | ".join(article_keywords),
                        "summary_keywords": " | ".join(summary_keywords),
                        "missing_important_topics": " | ".join(missing_topics),
                    }
                )

            print(f"Analyzed framing: {article_name}")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "Genre",
            "Article Name",
            "Model",
            "sentiment_shift",
            "keyword_overlap",
            "similarity",
            "framing_label",
            "article_keywords",
            "summary_keywords",
            "missing_important_topics",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Done. Wrote {len(output_rows)} rows to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Framing/emphasis analysis for article summaries.")
    parser.add_argument("--input", default="article_summaries.csv", help="Input CSV with article + summary columns")
    parser.add_argument("--output", default="framing_metrics.csv", help="Output CSV for framing metrics")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
