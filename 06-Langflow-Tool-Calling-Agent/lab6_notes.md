# Lab 6: Simple LLM Interface and Basic Tool Calling Agent (Langflow)

**Tool:** IBM Langflow Desktop 1.12.0
**LLM:** IBM watsonx.ai, model `ibm/granite-4-h-small` (Frankfurt region, `https://eu-de.ml.cloud.ibm.com`)

## Part A: Simple LLM Interface

### Block diagram

```
+------------+      +-------------------------------+      +-------------+
| Chat Input | ---> |        Language Model         | ---> | Chat Output |
|  (user     |      |  IBM watsonx.ai               |      |  (reply     |
|  question) |      |  ibm/granite-4-h-small        |      |  shown)     |
+------------+      |  + System Message             |      +-------------+
                    +-------------------------------+
```

### Steps
1. Open Langflow Desktop and click **Create first flow**, then choose a blank flow.
2. Rename the flow to **Simple LLM Interface**.
3. From **Input & Output**, drag **Chat Input** and **Chat Output** onto the canvas.
4. From **Models & Agents**, drag **Language Model** onto the canvas.
5. In Language Model, select the provider **IBM watsonx.ai** (API key, project ID and endpoint) and the model **ibm/granite-4-h-small**.
6. Set the System Message: *You are a helpful assistant. Answer briefly and clearly.*
7. Connect Chat Input → Language Model *Input*, and Language Model *Model Response* → Chat Output.
8. Open the **Playground** and ask a question.

### Output
| Screenshot | Description |
|---|---|
| `output/01_simple_llm_flow.png` | Flow with Chat Input → Language Model → Chat Output |
| `output/02_simple_llm_chat.png` | Question "What is Generative AI? Explain in 3 lines." and the model's reply |

## Part B: Basic Tool Calling Agent

### Block diagram

```
                          +-----------------------------------+
+------------+            |              AGENT                |            +-------------+
| Chat Input | ---------> |  LLM: ibm/granite-4-h-small       | ---------> | Chat Output |
+------------+   Input    |  Agent Instructions (prompt)      |  Response  +-------------+
                          |  Reasoning: which tool to call?   |
                          +-----------------------------------+
                                     ^      ^      ^
                               Tools |      |      |
                     +---------------+      |      +------------------+
                     |                      |                         |
             +---------------+     +-----------------+     +-------------------+
             |  Calculator   |     |  Current Date   |     | DuckDuckGo Search |
             |  (maths)      |     |  (date / time)  |     |  (web search)     |
             +---------------+     +-----------------+     +-------------------+
```

### Basic components of an AI agent
| Component | Role in this flow |
|---|---|
| **LLM (brain)** | Granite 4 understands the question and decides whether a tool is needed |
| **Instructions / prompt** | Tells the agent its role and when to use each tool |
| **Tools** | Calculator, Current Date and DuckDuckGo Search let the agent act beyond its own knowledge |
| **Input / Output** | Chat Input receives the user's question and Chat Output shows the final answer |
| **Reasoning loop** | Think → call a tool → read the tool result → answer (shown as tool steps in the Playground) |

### Tools used
| Tool | Action name | What it does |
|---|---|---|
| Calculator | `EVALUATE_EXPRESSION` | Evaluates arithmetic expressions exactly |
| Current Date | `GET_CURRENT_DATE` | Returns the current date and time for a timezone |
| DuckDuckGo Search | `FETCH_CONTENT_DATAFRAME` | Searches the web and returns result snippets |

### Steps to create the agent in Langflow
1. Create a new blank flow named **Tool Calling Agent**.
2. Drag **Chat Input**, **Agent** and **Chat Output** onto the canvas.
3. Drag the tool components **Calculator**, **Current Date** and **DuckDuckGo Search** onto the canvas.
4. Turn on **Tool Mode** for each tool, so its output becomes **Toolset**.
5. In the Agent, select the language model **ibm/granite-4-h-small**.
6. Write the Agent Instructions: *You are a helpful assistant. Use the Calculator tool for any maths, the Current Date tool for date or time questions, and DuckDuckGo Search for current events or facts you don't know. Always use a tool when one fits.*
7. Connect Chat Input → Agent *Input*, connect each tool's *Toolset* → Agent *Tools*, and connect Agent *Response* → Chat Output.
8. Open the **Playground**, ask questions and expand the tool step to see which tool the agent called.

### Prompts and responses
| Prompt | Tool called | Response |
|---|---|---|
| What is 15% of 2340 plus 125? | Calculator (`EVALUATE_EXPRESSION`) | 15% of 2,340 is 351; adding 125 gives **476** ✅ |
| What is today's date and day? | Current Date (`GET_CURRENT_DATE`) | October 15, 2026 (Friday). The date is correct but the weekday is wrong: it is a Thursday ⚠️ |
| Who is the current CEO of Google? | DuckDuckGo Search (`FETCH_CONTENT_DATAFRAME`) | **Sundar Pichai** ✅. The tenure dates given are inaccurate ⚠️ |
| What is the capital of France? | No tool (answered from model knowledge) | **Paris** ✅ |

| Screenshot | Description |
|---|---|
| `output/03_tool_calling_agent_flow.png` | Agent flow with 3 tools connected to Tools |
| `output/04_agent_calculator_and_date.png` | Calculator and Current Date tool calls |
| `output/05_agent_web_search.png` | DuckDuckGo Search tool call |
| `output/06_agent_no_tool.png` | Question answered without any tool |

Exported flows (can be imported back into Langflow): `output/simple_llm_interface_flow.json`, `output/tool_calling_agent_flow.json`.

## Observations
- The agent chose the correct tool for every question and used no tool when none was needed.
- Tool calls are fast (8–14 ms for Calculator and Current Date); the web search took about 13 seconds.
- The simple LLM flow can only answer from the model's training data. The agent can calculate exactly, know today's date and look up information online.

## Limitations
- The LLM can still make mistakes after a correct tool call. The Current Date tool returned the right date, but the model worked out the weekday itself and said Friday instead of Thursday.
- Web search results can be mixed up. The agent named the correct CEO but gave wrong dates for when he took the role, so answers should be checked against the source.
- DuckDuckGo search may be rate-limited if used too often.
- The agent depends on the watsonx API (internet access, API key, usage limits).

## Result and Conclusion
Studied the Langflow framework for AI agents. Built a simple LLM chat interface and a basic tool-calling agent using IBM watsonx.ai Granite with Calculator, Current Date and DuckDuckGo Search tools, and observed how the agent decides when to call a tool.
