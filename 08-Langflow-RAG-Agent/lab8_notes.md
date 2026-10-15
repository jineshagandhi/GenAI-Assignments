# Lab 8: RAG Agent using Langflow

**Tool:** IBM Langflow Desktop 1.12.0
**Knowledge source:** the Generative AI Lab Manual PDF (`MITWPU_DCET_Generative_AI_Lab_Manual_Updated_Lab_Titles_2023_2027 (1).pdf`)
**Embedding model:** IBM watsonx.ai `ibm/slate-125m-english-rtrvr-v2`
**Vector store:** Langflow Knowledge Base (local Chroma)
**LLM:** IBM watsonx.ai `ibm/granite-4-h-small`

## Block diagram

```
 STAGE 1: INDEXING (flow "RAG Ingestion" - run once)

+------------+     +-------------------+     +---------------------------------------+
| Read File  | --> |    Split Text     | --> |  Knowledge (Ingest)                   |
| (lab       |     | chunk size 1000   |     |  embeddings: ibm/slate-125m-rtrvr-v2  |
|  manual    |     | overlap 200       |     |  store: Chroma  ->  "lab_manual"      |
|  PDF)      |     | -> 22 chunks      |     |  (22 chunks as vectors)               |
+------------+     +-------------------+     +---------------------------------------+
                                                               |
 STAGE 2: RETRIEVAL + GENERATION (flow "RAG Agent")           | stored vectors
                                                               v
+------------+  question  +---------------------------+  top-5  +----------+
| Chat Input | ---------> |  Knowledge (Retrieve)     | ------> |  Parser  |
+------------+            |  similarity search        | chunks  | chunks ->|
      |                   +---------------------------+         |  text    |
      |                                                         +----------+
      |                                                              | context
      |  question       +-----------------------------------+        v
      +---------------> |  Prompt: {context} + {question}   | <------+
                        +-----------------------------------+
                                        |
                                        v
                        +-----------------------------------+      +-------------+
                        |  Agent (ibm/granite-4-h-small)    | ---> | Chat Output |
                        |  "answer ONLY from the context"   |      +-------------+
                        +-----------------------------------+
```

## Basic components of the RAG agent
| Component | Role |
|---|---|
| **Read File** | Loads the PDF and extracts its text |
| **Split Text** | Splits the text into 1000-character chunks with a 200-character overlap, so each chunk is small enough to search and to fit in the prompt |
| **Embedding model** | `ibm/slate-125m-english-rtrvr-v2` turns each chunk (and each question) into a vector of numbers that captures its meaning |
| **Knowledge (vector store)** | Stores the chunk vectors (Ingest mode) and finds the chunks most similar to the question (Retrieve mode, top 5) |
| **Parser** | Converts the retrieved chunks into plain text |
| **Prompt** | Combines the retrieved context and the user's question into one prompt |
| **Agent (LLM)** | Granite 4 writes the answer using only the retrieved context |
| **Chat Input / Chat Output** | Take the user's question and show the answer in the Playground |

## Tools used in this assignment
| Tool | Purpose |
|---|---|
| Langflow Desktop | Visual drag-and-drop builder for the ingestion and RAG flows |
| IBM watsonx.ai embeddings (`slate-125m-english-rtrvr-v2`) | Creates the embeddings for retrieval |
| Chroma (Langflow Knowledge Base) | Local vector database, no cloud account needed |
| IBM watsonx.ai Granite 4 (`granite-4-h-small`) | Generates the final answer |

## Retrieval and generation stages
1. **Indexing (done once):** PDF → text → 22 chunks → embeddings → stored in the `lab_manual` knowledge base. Result: 3,158 words, 20,940 characters, 22 chunks (average 952 characters), status *ready*.
2. **Retrieval:** the user's question is converted into an embedding, and the knowledge base returns the 5 chunks whose embeddings are most similar (semantic search, not keyword matching).
3. **Augmentation:** the Prompt component places those chunks as `{context}` above the `{question}`.
4. **Generation:** the Agent (Granite) reads the prompt and writes an answer, instructed to use only the retrieved context and to say so when the answer is not there.

## Steps to create the RAG agent in Langflow
**Flow 1: RAG Ingestion**
1. Create a blank flow named **RAG Ingestion**.
2. Add **Read File** and upload the lab manual PDF.
3. Add **Split Text** with Chunk Size 1000 and Chunk Overlap 200.
4. Add **Knowledge** and set Mode to **Ingest**. Create a new knowledge base `lab_manual` with the embedding model `ibm/slate-125m-english-rtrvr-v2` and the local Chroma provider.
5. Connect Read File (*Raw Content*) → Split Text (*Input*), and Split Text (*Chunks*) → Knowledge (*Input*).
6. Run the Knowledge component (▷). Read File took 426 ms, Split Text 34 ms and Knowledge (embedding + storing) 14.6 s.

**Flow 2: RAG Agent**
1. Create a new flow from the **Vector Store RAG** template and name it **RAG Agent**.
2. In **Knowledge**, set Mode to **Retrieve** and Knowledge to **lab_manual**.
3. In **Agent**, set Language Model to **ibm/granite-4-h-small**.
4. The template connects Chat Input → Knowledge → Parser → Prompt (*context*), Chat Input → Prompt (*question*), Prompt → Agent → Chat Output.
5. Open the **Playground** and ask questions about the document.

## Prompts and responses
| Question | Agent answer | Grounded in source? |
|---|---|---|
| What is Lab 5 about? | Prompt engineering using IBM Watson AI, with its objectives (protocols, parameter tuning, text generation, RAG, summarization, extraction, comparison, search, classification) and theory (tokens, one-shot prompting, parameters) | ✅ Matches Lab 5, page 9 |
| How many marks is LCA2 and which assignments does it cover? | 5 marks; Assignments 4, 5 and 6 | ✅ Matches the Lab Assessment Rubric, page 13 |
| Which COs are mapped to Lab 8? | CO2, CO4 and CO5 (from the CO–Lab Mapping table) | ✅ Matches page 13 |
| What tools and resources are used in this lab course? | Python, Keras, TensorFlow, Hugging Face, Transformers, Langflow, LangChain, IBM watsonx, MNIST, Fashion-MNIST, Jupyter Notebook | ✅ Matches page 2 |
| Who won the IPL in 2024? | "I don't have enough information in the provided context…" | ✅ Correctly refused; the answer is not in the document |

## Output
| File | Description |
|---|---|
| `output/01_ingestion_flow.png` | Ingestion flow after running (Read File → Split Text → Knowledge Ingest) |
| `output/02_rag_agent_flow.png` | RAG Agent flow (Knowledge Retrieve → Parser → Prompt → Agent) |
| `output/03_rag_answer_lab5.png` | Answer about Lab 5 |
| `output/04_rag_answer_lca2_and_cos.png` | Answers about LCA2 marks and the COs for Lab 8 |
| `output/05_rag_answer_tools_and_not_in_document.png` | Tools and resources answer, plus the out-of-document question |
| `output/rag_ingestion_flow.json`, `output/rag_agent_flow.json` | Exported flows (can be imported back into Langflow) |

## Evaluation: are the responses grounded in the source?
All 4 in-document answers were checked against the lab manual and are correct. For the IPL question, the agent did not use its general knowledge and said the context does not contain the answer. This shows the agent is grounded in the supplied document.

## Limitations
- Answers can only be as good as the retrieved chunks. If the relevant text is split across chunks or not in the top 5, the answer may be incomplete.
- Tables in the PDF (rubric, CO mapping) are read as plain text, which can lose row and column structure.
- The knowledge base must be re-ingested when the document changes.
- It needs the IBM watsonx API for both embeddings and generation (internet access and an API key).

## Result and Conclusion
Studied the Langflow framework for RAG agents. Built an ingestion flow that stores the lab manual in a Chroma knowledge base using IBM watsonx embeddings, and a RAG agent that retrieves the relevant chunks and answers questions with IBM Granite, grounded only in the supplied document.
