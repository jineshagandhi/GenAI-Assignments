# Lab 7: Python Agent to Scrape and Summarize Articles

**Script:** `article_summarizer_agent.py`
**LLM:** IBM watsonx.ai, model `ibm/granite-4-h-small` (Frankfurt region)
**Libraries:** `requests`, `beautifulsoup4`, `ibm-watsonx-ai`

## Block diagram

```
+-----------------+      +-------------------------------------+
|  User request   | ---> |              AGENT                  |
| "Summarize this |      |  LLM: ibm/granite-4-h-small         |
|  article: URL"  |      |  System prompt (role + output form) |
+-----------------+      +-------------------------------------+
                            |  (1) decides to call the tool  ^
                            v                                |  (3) article text
                   +-------------------------+               |
                   |  Tool: scrape_article   | --------------+
                   |  requests -> download   |
                   |  BeautifulSoup -> text  |
                   +-------------------------+
                            |
                    (2) web page (HTML)
                            |
                   +-------------------------+
                   |   Article website       |
                   +-------------------------+

  (4) LLM writes: Title + 4-5 sentence summary + 3 key points  --->  output/article_summaries.md
```

## Basic components of the agent
| Component | In this script |
|---|---|
| **LLM (brain)** | Granite 4 on IBM watsonx.ai decides when to call the tool and writes the summary |
| **System prompt** | Defines the agent's role and the output format (title, summary, key points) |
| **Tool** | `scrape_article(url)`, a Python function the LLM can call |
| **Tool description (schema)** | `TOOLS` list: tells the LLM the tool's name, purpose and `url` parameter |
| **Agent loop** | `run_agent()`: LLM → tool call → tool result back to LLM → final answer |
| **Output** | Printed in the terminal and saved to `output/article_summaries.md` |

## Tools used
| Tool / library | Purpose |
|---|---|
| `requests` | Downloads the web page HTML (with a browser User-Agent header) |
| `BeautifulSoup` | Parses the HTML, removes scripts, menus and footers, and extracts `<p>` paragraph text and the page title |
| `ibm-watsonx-ai` (`ModelInference.chat` with `tools`) | Sends messages to Granite and receives tool calls or the final answer |
| IBM watsonx.ai Granite 4 | Large language model that does the tool calling and summarization |

## Explanation of the Python script
1. **Setup:** reads `WATSONX_APIKEY` and `WATSONX_PROJECT_ID` from environment variables and connects to `ibm/granite-4-h-small` with `ModelInference`.
2. **`scrape_article(url)`** (the tool):
   - downloads the page with `requests.get()`
   - removes `script`, `style`, `nav`, `header`, `footer`, `aside` and `form` tags
   - keeps paragraphs longer than 8 words (skips menu text and captions)
   - returns the title, word count and the first 6000 characters of text (to stay within the model's input limit)
3. **`TOOLS`:** a JSON schema that describes the tool to the LLM, so the model knows it can call `scrape_article` with a `url`.
4. **`SYSTEM_PROMPT`:** tells the agent to call the tool for a URL and reply with a title, a 4–5 sentence summary and 3 key points, using only the scraped text.
5. **`run_agent()`:** the agent loop:
   - sends the messages and tools to the model
   - if the model returns a **tool call**, runs the Python function, adds the result as a `tool` message and asks the model again
   - if the model returns **text**, that is the final answer
   - stops after 5 steps at most, so it can't loop forever
6. **Main part:** runs the agent on 2 article URLs, prints each step and saves everything to `output/article_summaries.md`.

## Steps to run
```powershell
cd "C:\Users\JINESHA GANDHI\Downloads\GenAI-Assignments\07-Python-Agent-Scrape-Summarize"
$env:WATSONX_APIKEY="your-api-key"
$env:WATSONX_PROJECT_ID="your-project-id"
cls
python article_summarizer_agent.py
```

## Prompts and responses
| Article | Tool call | Scraped | Result |
|---|---|---|---|
| MIT News: "Explained: Generative AI" | `scrape_article(url)` | 33 paragraphs, 1497 words | Title, summary (history, GANs, diffusion, transformers) and 3 key points |
| IBM: "What is Generative AI?" | `scrape_article(url)` | 94 paragraphs, 3724 words | Title, summary (foundation models, fine-tuning, RLHF, RAG) and 3 key points |

The full responses are in `output/article_summaries.md`.

## Observations
- The agent decided by itself to call `scrape_article`. The URL was not passed to the function by our code; the LLM chose the tool and filled in its argument.
- Each article needed 2 steps: (1) tool call, (2) final summary.
- The summaries followed the requested format and matched the article content.

## Limitations
- Only the first 6000 characters are sent to the model, so for long articles (IBM, 3724 words) later sections are not covered by the summary.
- Websites that load text with JavaScript, block bots or need a login cannot be scraped with `requests`.
- Generated summaries should always be checked against the source, because the LLM can leave out or misstate details.
- Scraping must respect each website's terms of use and robots.txt.

## Result and Conclusion
Studied how to build a Python AI agent that uses tool calling. The agent scrapes an article from a URL with requests and BeautifulSoup, and IBM watsonx.ai Granite summarizes it into a title, a summary and key points.
