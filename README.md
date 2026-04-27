# TMD Final Pipeline

This project runs a full article summarization + evaluation workflow:
1. Pull article text from links in `manually-labeling-news-articles.csv`
2. Generate summaries (GPT / Gemini / Grok, based on keys in `.env`)
3. Run framing analysis
4. Run coverage analysis
5. Run faithfulness analysis (NLI with `roberta-large-mnli`)
6. Merge results for comparison

## Files
- `summarize_articles_pipeline.py`: creates `article_summaries.csv`
- `framing_analysis.py`: framing/emphasis metrics
- `coverage_analysis.py`: key-point coverage metrics
- `faithfulness_analysis.py`: hallucination/faithfulness metrics
- `run_full_evaluation.py`: orchestration script for all steps

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (or copy from `.env.example`) and add at least one key:

```bash
cp .env.example .env
```

Then edit `.env` and set at least `GEMINI_API_KEY` (or other provider keys).

## 2) Run Everything (All Articles)

```bash
python3 run_full_evaluation.py --input manually-labeling-news-articles.csv
```

Output goes to a timestamped folder like:
- `runs/run_YYYYMMDD_HHMMSS/article_summaries.csv`
- `runs/run_YYYYMMDD_HHMMSS/framing_metrics.csv`
- `runs/run_YYYYMMDD_HHMMSS/coverage_metrics.csv`
- `runs/run_YYYYMMDD_HHMMSS/faithfulness_metrics.csv`
- `runs/run_YYYYMMDD_HHMMSS/comparison_metrics.csv`

## 3) Run Just One Article

Use exact title match from the CSV `Article Name` column:

```bash
python3 run_full_evaluation.py \
  --input manually-labeling-news-articles.csv \
  --article-name "Exact Article Title Here"
```

## 4) Quick Test Run (First N Articles)

```bash
python3 run_full_evaluation.py --input manually-labeling-news-articles.csv --max-articles 1
```

## 5) Optional: List Article Titles First

If you want to copy an exact title for `--article-name`:

```bash
python3 - <<'PY'
import csv
with open('manually-labeling-news-articles.csv', newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        print(row['Article Name'])
PY
```

## 6) Run Individual Steps Manually

### Summaries only
```bash
python3 summarize_articles_pipeline.py \
  --input manually-labeling-news-articles.csv \
  --output article_summaries.csv
```

### Framing only
```bash
python3 framing_analysis.py --input article_summaries.csv --output framing_metrics.csv
```

### Coverage only
```bash
python3 coverage_analysis.py --summaries article_summaries.csv --output coverage_metrics.csv
```

### Faithfulness only
```bash
python3 faithfulness_analysis.py --input article_summaries.csv --output faithfulness_metrics.csv
```

## 7) Compare Models in Notebook

Open the notebook:

```bash
jupyter notebook model_comparison_notebook.ipynb
```

Inside the notebook:
1. Set `RUN_FOLDER_NAME` (example: `run_20260427_084418`)
2. Run all cells
3. It will output:
   1. Side-by-side model comparison table
   2. Aggregate and composite ranking
   3. Charts
   4. `model_ranking_summary.csv` saved in that run folder

## Notes
- `--article-name` requires exact match (case-insensitive).
- First faithfulness run can be slower because `roberta-large-mnli` downloads from Hugging Face.
- If a provider API key is missing, that provider summary column remains blank.

## Troubleshooting (Bus Error / Native Lib Crash)

If you see `zsh: bus error python3 ...`, use a clean Python 3.11 virtualenv (recommended for `torch` / `sentence-transformers` / `transformers` stability):

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you only want to run summarization while debugging ML dependencies:

```bash
python3 run_full_evaluation.py \
  --input manually-labeling-news-articles.csv \
  --article-name "Exact Article Title Here" \
  --skip-framing --skip-coverage --skip-faithfulness
```
