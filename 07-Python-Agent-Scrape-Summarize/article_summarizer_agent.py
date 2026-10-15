import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


os.makedirs("output", exist_ok=True)


API_KEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
URL = os.getenv("WATSONX_URL", "https://eu-de.ml.cloud.ibm.com")
MODEL_ID = "ibm/granite-4-h-small"

MAX_ARTICLE_CHARS = 6000

ARTICLES = [
    "https://news.mit.edu/2023/explained-generative-ai-1109",
    "https://www.ibm.com/think/topics/generative-ai"
]

if not API_KEY or not PROJECT_ID:
    print("Please set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.")
    sys.exit(1)


model = ModelInference(
    model_id=MODEL_ID,
    credentials=Credentials(url=URL, api_key=API_KEY),
    project_id=PROJECT_ID
)

print("Connected to IBM watsonx.ai")
print("Model:", MODEL_ID)


# Tool: web scraper
def scrape_article(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else "No title"

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p.split()) > 8]
    text = "\n".join(paragraphs)

    print(f"   [scrape_article] Title: {title}")
    print(f"   [scrape_article] Paragraphs: {len(paragraphs)}, Words: {len(text.split())}")

    return {
        "title": title,
        "word_count": len(text.split()),
        "text": text[:MAX_ARTICLE_CHARS]
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scrape_article",
            "description": "Download a web page and return its title and main article text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the article"}
                },
                "required": ["url"]
            }
        }
    }
]

AVAILABLE_TOOLS = {"scrape_article": scrape_article}

SYSTEM_PROMPT = (
    "You are an article summarizer agent. When the user gives a URL, call the "
    "scrape_article tool to get the article text. Then write:\n"
    "1. Title of the article\n"
    "2. A summary in 4-5 sentences\n"
    "3. 3 key points as bullet points\n"
    "Use only information from the scraped article."
)


def parse_arguments(raw_arguments):
    arguments = json.loads(raw_arguments)
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return arguments


# Agent loop: LLM decides -> tool runs -> LLM reads result -> final answer
def run_agent(user_request, max_steps=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]
    tool_log = []

    for step in range(1, max_steps + 1):
        result = model.chat(
            messages=messages,
            tools=TOOLS,
            params={"temperature": 0, "max_tokens": 600}
        )
        message = result["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            print(f"   Step {step}: Agent wrote the final answer.")
            return message["content"].strip(), tool_log

        messages.append(message)

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            arguments = parse_arguments(tool_call["function"]["arguments"])
            print(f"   Step {step}: Agent called tool -> {name}({arguments})")

            tool_output = AVAILABLE_TOOLS[name](**arguments)
            tool_log.append({
                "tool": name,
                "arguments": arguments,
                "title": tool_output["title"],
                "word_count": tool_output["word_count"]
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_output)
            })

    return "Agent stopped: too many steps.", tool_log


results_file = open("output/article_summaries.md", "w", encoding="utf-8")
results_file.write("# Article Summaries by Python Agent\n\n")
results_file.write(f"Model: `{MODEL_ID}`\n\n")

for number, article_url in enumerate(ARTICLES, start=1):
    user_request = "Summarize this article: " + article_url

    print("\n" + "=" * 70)
    print(f"Article {number}")
    print("=" * 70)
    print("User request:", user_request)

    summary, tool_log = run_agent(user_request)

    print("\nAgent Response:\n")
    print(summary)

    results_file.write(f"## Article {number}\n\n")
    results_file.write(f"**User request:** {user_request}\n\n")
    for entry in tool_log:
        results_file.write(
            f"**Tool called:** `{entry['tool']}({entry['arguments']})` - "
            f"scraped \"{entry['title']}\" ({entry['word_count']} words)\n\n"
        )
    results_file.write("**Agent response:**\n\n" + summary + "\n\n")

results_file.close()

print("\nAll articles summarized successfully.")
print("Results saved in output/article_summaries.md")
