import csv
import os
import argparse
import re
import time
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_CSV = "manually-labeling-news-articles.csv"
OUTPUT_CSV = "article_summaries.csv"
SYSTEM_PROMPT = "You are a helpful, general-purpose assistant."
PROMPT_PREFIX = "Please summarize the following article in a paragraph."
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = [5, 10]


def normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())


def detect_input_columns(fieldnames: List[str]) -> Dict[str, str]:
    norm_map = {normalize_header(c): c for c in fieldnames}

    article_col = norm_map.get("articlename")
    if not article_col:
        raise ValueError("Could not find article title column (expected something like 'Article Name').")

    genre_col = None
    article_text_col = None

    for normalized, original in norm_map.items():
        if genre_col is None and normalized.startswith("genre"):
            genre_col = original
        if article_text_col is None and normalized == "articletext":
            article_text_col = original

    if not article_text_col:
        raise ValueError("Could not find article text column (expected 'Article Text').")

    return {
        "article_name": article_col,
        "genre": genre_col or "",
        "article_text": article_text_col,
    }


def build_clients():
    clients = {}

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    xai_key = os.getenv("XAI_API_KEY", "").strip()

    if openai_key:
        clients["gpt"] = {
            "client": OpenAI(api_key=openai_key),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        }

    if gemini_key:
        clients["gemini"] = {
            "client": OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        }

    if xai_key:
        clients["grok"] = {
            "client": OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1"),
            "model": os.getenv("GROK_MODEL", "grok-4.20"),
        }

    return clients


def load_articles(csv_path: str) -> Dict[str, List[Tuple[str, str, str]]]:
    article_texts: Dict[str, List[Tuple[str, str, str]]] = {}

    with open(csv_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {csv_path}")

        cols = detect_input_columns(reader.fieldnames)
        for row in reader:
            title = (row.get(cols["article_name"]) or "").strip()
            genre = (row.get(cols["genre"]) or "").strip() if cols["genre"] else ""
            genre = genre or "Unknown"
            text = (row.get(cols["article_text"]) or "").strip()
            text_source = "provided_article_text" if text else "missing_article_text"

            article_texts.setdefault(genre, []).append((title, text, text_source))

    return article_texts


def summarize(client: OpenAI, model: str, article_name: str, article_text: str) -> str:
    if not article_text.strip():
        return ""

    user_prompt = (
        f"{PROMPT_PREFIX}\n\n"
        f"Article Name: {article_name.strip() or 'Untitled'}\n\n"
        f"Article Content:\n{article_text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def summarize_with_retries(
    client: OpenAI,
    model: str,
    article_text: str,
    article_name: str,
    provider: str,
    max_retries: int = MAX_RETRIES,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            return summarize(client, model, article_name, article_text)
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Provider '{provider}' failed for '{article_name}' after {max_retries} attempts: {exc}"
                ) from exc
            sleep_seconds = RETRY_DELAYS_SECONDS[min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)]
            print(
                f"Retry {attempt}/{max_retries} for provider '{provider}' on '{article_name}' "
                f"after error: {exc}. Waiting {sleep_seconds}s before retry."
            )
            time.sleep(sleep_seconds)


def run(input_csv: str = INPUT_CSV, output_csv: str = OUTPUT_CSV):
    clients = build_clients()
    if not clients:
        raise ValueError(
            "No API keys found. Add at least one of OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY to .env"
        )

    article_texts = load_articles(input_csv)

    rows = []
    for genre, items in article_texts.items():
        for article_name, article_text, text_source in items:
            result = {
                "Genre": genre,
                "Article Name": article_name,
                "Article Text": article_text,
                "Article Text Source": text_source,
                "GPT Summary": "",
                "Gemini Summary": "",
                "Grok Summary": "",
            }

            for provider, config in clients.items():
                summary = summarize_with_retries(
                    client=config["client"],
                    model=config["model"],
                    article_text=article_text,
                    article_name=article_name,
                    provider=provider,
                    max_retries=MAX_RETRIES,
                )

                if provider == "gpt":
                    result["GPT Summary"] = summary
                elif provider == "gemini":
                    result["Gemini Summary"] = summary
                elif provider == "grok":
                    result["Grok Summary"] = summary

            rows.append(result)
            print(f"Summarized: {article_name}")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "Genre",
            "Article Name",
            "Article Text",
            "Article Text Source",
            "GPT Summary",
            "Gemini Summary",
            "Grok Summary",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize provided article text using available LLM providers.")
    parser.add_argument("--input", default=INPUT_CSV, help="Input CSV with article metadata and Article Text")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV with model summaries")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(input_csv=args.input, output_csv=args.output)
