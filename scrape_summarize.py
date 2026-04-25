from newspaper import Article 
import csv
import openai
from dotenv import load_dotenv
import anthropic 


# Load keys
load_dotenv()

article_texts = {}
# key=Genre, value=[(title, text, gpt_summary, claude_summary, grok_summary), ...]
article_summaries = {}

# Parse links, construct link dict ({key=Genre: value=(title, text)})
with open('manually-labeling-news-articles.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row["Genre"] not in article_texts:
            # Download and parse text 
            article = Article(row["Link to Article"])
            article.download()
            article.parse()

            article_texts[row["Genre"]] = [(row["Article Name"], article.text)]
        else:
            # Download and parse text 
            article = Article(row["Link to Article"])
            article.download()
            article.parse()
            
            article_texts[row["Genre"]] += [(row["Article Name"], article.text)]

# Query LLMs for each link
prompt = "Please summarize the following article: "

# Clients
client_gpt = openai.OpenAI()
client_claude = anthropic.Anthropic()

# System prompt 
system_prompt = "You are a helpful, general-purpose assistant."

# Loop through articles and summarize
for genre in article_texts:

    for article_name, article_text in article_texts[genre]:
        
        # ChatGPT response
        response = client_gpt.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt + article_text}
            ]
        )

        gpt_response = response.choices[0].message.content

        # Claude response 
        response = client_claude.message.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt + article_text}
            ]
        )
        
        claude_response = response.content[0].text

        # Grok response
        response = client_gpt.chat.completions.create(                          # grok OpenAI compatible
            model="grok-beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt + article_text}
            ]
        )

        grok_response = response.choices[0].message.content

        # Populate article_summaries
        if genre not in article_summaries:
            article[genre] = [(article_name, article_text, gpt_response, claude_response, grok_response)]
        else:
            article[genre] += [(article_name, article_text, gpt_response, claude_response, grok_response)]
