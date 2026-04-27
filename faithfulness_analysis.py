import argparse
import csv
import json
import re
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


def normalize_column(name: str) -> str:
    return "_".join(name.strip().lower().split())


def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_article_chunks(text: str, chunk_words: int = 220, overlap_words: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk_words_slice = words[start:end]
        if not chunk_words_slice:
            continue
        chunks.append(" ".join(chunk_words_slice))
        if end >= len(words):
            break
    return chunks


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


def pick_best_chunk(
    summary_sentence: str,
    article_chunks: List[str],
    embed_model: SentenceTransformer,
) -> str:
    if not article_chunks:
        return ""

    if len(article_chunks) == 1:
        return article_chunks[0]

    sent_emb = embed_model.encode([summary_sentence])
    chunk_emb = embed_model.encode(article_chunks)
    sims = cosine_similarity(sent_emb, chunk_emb)[0]
    best_idx = int(sims.argmax())
    return article_chunks[best_idx]


def normalize_nli_label(raw_label: str) -> str:
    label = (raw_label or "").upper()

    if "ENTAIL" in label or label in {"LABEL_2", "2"}:
        return "ENTAILMENT"
    if "CONTRADI" in label or label in {"LABEL_0", "0"}:
        return "CONTRADICTION"
    if "NEUTRAL" in label or label in {"LABEL_1", "1"}:
        return "NEUTRAL"

    return "NEUTRAL"


def faithfulness_score(supported: int, neutral: int, contradictions: int) -> float:
    total = supported + neutral + contradictions
    if total == 0:
        return 0.0

    # Weighted score for quick comparison across models.
    # Supported counts fully, neutral partially, contradictions count as 0.
    return (supported + 0.5 * neutral) / total


def run(
    input_csv: str,
    output_csv: str,
    chunk_words: int,
    overlap_words: int,
):
    nli = pipeline("text-classification", model="roberta-large-mnli")
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
            article_text = row.get(article_col, "") or ""
            article_name = row.get("Article Name", row.get("article_name", ""))
            genre = row.get("Genre", row.get("genre", ""))

            article_chunks = build_article_chunks(
                text=article_text,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )

            for summary_col in summary_cols:
                summary_text = row.get(summary_col, "") or ""
                summary_sentences = split_sentences(summary_text)

                supported = 0
                contradictions = 0
                neutral = 0
                sentence_results = []

                for sentence in summary_sentences:
                    premise = pick_best_chunk(sentence, article_chunks, embed_model)
                    if not premise:
                        label = "NEUTRAL"
                        conf = 0.0
                    else:
                        result = nli(
                            {
                                "text": premise,
                                "text_pair": sentence,
                            },
                            truncation=True,
                            max_length=512,
                        )
                        # HF pipeline output shape can vary by version:
                        # - dict: {"label": "...", "score": ...}
                        # - list[dict]: [{"label": "...", "score": ...}]
                        if isinstance(result, list):
                            top = result[0] if result else {"label": "NEUTRAL", "score": 0.0}
                        elif isinstance(result, dict):
                            top = result
                        else:
                            top = {"label": "NEUTRAL", "score": 0.0}
                        label = normalize_nli_label(top.get("label", ""))
                        conf = float(top.get("score", 0.0))

                    if label == "ENTAILMENT":
                        supported += 1
                    elif label == "CONTRADICTION":
                        contradictions += 1
                    else:
                        neutral += 1

                    sentence_results.append(
                        {
                            "summary_sentence": sentence,
                            "label": label,
                            "confidence": round(conf, 4),
                        }
                    )

                total = supported + contradictions + neutral
                score = faithfulness_score(supported, neutral, contradictions)

                output_rows.append(
                    {
                        "Genre": genre,
                        "Article Name": article_name,
                        "Model": summary_col.replace(" Summary", ""),
                        "supported": supported,
                        "contradictions": contradictions,
                        "neutral": neutral,
                        "total_summary_sentences": total,
                        "faithfulness_score": round(score, 4),
                        "contradiction_rate": round((contradictions / total), 4) if total else 0.0,
                        "sentence_level_nli": json.dumps(sentence_results, ensure_ascii=True),
                    }
                )

            print(f"Faithfulness analyzed: {article_name}")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "Genre",
            "Article Name",
            "Model",
            "supported",
            "contradictions",
            "neutral",
            "total_summary_sentences",
            "faithfulness_score",
            "contradiction_rate",
            "sentence_level_nli",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Done. Wrote {len(output_rows)} rows to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Faithfulness evaluation via NLI using roberta-large-mnli")
    parser.add_argument("--input", default="article_summaries.csv", help="Input CSV with article + summary columns")
    parser.add_argument("--output", default="faithfulness_metrics.csv", help="Output CSV path")
    parser.add_argument("--chunk-words", type=int, default=220, help="Words per article chunk for premise selection")
    parser.add_argument("--overlap-words", type=int, default=60, help="Overlap words between chunks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_csv=args.input,
        output_csv=args.output,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
