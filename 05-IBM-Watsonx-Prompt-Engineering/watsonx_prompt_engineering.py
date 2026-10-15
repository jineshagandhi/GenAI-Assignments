import os
import sys
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


os.makedirs("output", exist_ok=True)


API_KEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
URL = os.getenv("WATSONX_URL", "https://eu-de.ml.cloud.ibm.com")
MODEL_ID = "ibm/granite-4-h-small"

if not API_KEY or not PROJECT_ID:
    print("Please set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.")
    sys.exit(1)


credentials = Credentials(url=URL, api_key=API_KEY)

model = ModelInference(
    model_id=MODEL_ID,
    credentials=credentials,
    project_id=PROJECT_ID
)

print("Connected to IBM watsonx.ai")
print("Model:", MODEL_ID)
print("Region URL:", URL)


GREEDY = {
    "temperature": 0,
    "max_tokens": 200
}

results_file = open("output/prompt_results.md", "w", encoding="utf-8")
results_file.write("# IBM watsonx.ai Prompt Engineering Results\n\n")
results_file.write(f"Model: `{MODEL_ID}`\n\n")


def run_prompt(number, title, prompt, params=GREEDY, note=""):
    messages = [{"role": "user", "content": prompt}]
    result = model.chat(messages=messages, params=params)

    response = result["choices"][0]["message"]["content"].strip()
    tokens_used = result["usage"]["total_tokens"]

    print("\n" + "=" * 70)
    print(f"Exercise {number}: {title}")
    print("=" * 70)
    print("Parameters:", params)
    print("\nPrompt:\n" + prompt)
    print("\nResponse:\n" + response)
    print("\nTotal tokens used:", tokens_used)

    results_file.write(f"## Exercise {number}: {title}\n\n")
    if note != "":
        results_file.write(note + "\n\n")
    results_file.write(f"**Parameters:** `{params}`\n\n")
    results_file.write("**Prompt:**\n\n```\n" + prompt + "\n```\n\n")
    results_file.write("**Response:**\n\n```\n" + response + "\n```\n\n")
    results_file.write(f"Total tokens used: {tokens_used}\n\n")

    return response


# Exercise 1: Tokens in a prompt
token_prompt = "Generative AI models read text as tokens, not words."
tokenized = model.tokenize(prompt=token_prompt, return_tokens=True)["result"]

print("\n" + "=" * 70)
print("Exercise 1: Tokens in a Prompt")
print("=" * 70)
print("Text        :", token_prompt)
print("Word count  :", len(token_prompt.split()))
print("Token count :", tokenized["token_count"])
print("Tokens      :", tokenized["tokens"])

results_file.write("## Exercise 1: Tokens in a Prompt\n\n")
results_file.write(f"Text: `{token_prompt}`\n\n")
results_file.write(f"Word count: {len(token_prompt.split())}\n\n")
results_file.write(f"Token count: {tokenized['token_count']}\n\n")
results_file.write(f"Tokens: `{tokenized['tokens']}`\n\n")


# Exercise 2: Zero-shot text generation
run_prompt(
    2,
    "Text Generation (Zero-Shot)",
    "Write a short paragraph of 3 sentences explaining what Generative AI is "
    "for a first-year engineering student.",
    note="No example is given to the model (zero-shot)."
)


# Exercise 3: One-shot prompting
run_prompt(
    3,
    "One-Shot Prompting",
    "Convert the product name into a catchy marketing slogan. "
    "Reply with only the slogan.\n\n"
    "Product: Solar power bank\n"
    "Slogan: Charge your phone with the power of the sun!\n\n"
    "Product: Noise cancelling headphones\n"
    "Slogan:",
    params={"temperature": 0, "max_tokens": 30},
    note="One example (product -> slogan) is given before the actual input."
)


# Exercise 4: Parameter tuning - temperature
creative_prompt = "Write a two-line poem about the monsoon in Pune."

run_prompt(
    4,
    "Parameter Tuning - Low Temperature (Greedy)",
    creative_prompt,
    params={"temperature": 0, "max_tokens": 60},
    note="Temperature 0 always picks the most likely next token, so the output is fixed and repeatable."
)

run_prompt(
    "4b",
    "Parameter Tuning - High Temperature (Sampling)",
    creative_prompt,
    params={"temperature": 1.2, "top_p": 0.9, "max_tokens": 60, "seed": 42},
    note="A high temperature with top-p sampling gives more creative and varied output."
)


# Exercise 5: Parameter tuning - max tokens and stop sequences
run_prompt(
    5,
    "Parameter Tuning - Max Tokens and Stop Sequence",
    "List 6 advantages of cloud computing as a numbered list, one line each.",
    params={"temperature": 0, "max_tokens": 150, "stop": ["4."]},
    note="Max tokens sets the upper limit on output length and the stop sequence '4.' "
         "stops the list after 3 points even though 6 were asked for."
)


# Exercise 6: Summarization
article = (
    "India's Chandrayaan-3 mission successfully landed near the south pole of "
    "the Moon on 23 August 2023, making India the first country to land in "
    "this region. The mission consisted of the Vikram lander and the Pragyan "
    "rover. The rover travelled more than 100 metres on the lunar surface and "
    "confirmed the presence of sulphur and other elements in the soil. The "
    "mission was developed by ISRO at a cost of about 615 crore rupees, which "
    "is much lower than similar missions by other countries."
)

run_prompt(
    6,
    "Summarization",
    "Summarize the following article in 2 sentences.\n\n"
    "Article: " + article,
    params={"temperature": 0, "max_tokens": 200}
)


# Exercise 7: Extraction
run_prompt(
    7,
    "Information Extraction",
    "Extract the following details from the text and return only JSON "
    "with keys: mission, organization, landing_date, lander, rover, cost.\n\n"
    "Text: " + article,
    params={"temperature": 0, "max_tokens": 150}
)


# Exercise 8: Comparison
run_prompt(
    8,
    "Compare Two Texts",
    "Compare the two laptops below. Give 3 key differences as one-line bullet "
    "points and recommend one for a computer engineering student.\n\n"
    "Laptop A: Intel i5, 8 GB RAM, 512 GB SSD, integrated graphics, 1.4 kg, "
    "battery 12 hours, price 55,000 rupees.\n"
    "Laptop B: AMD Ryzen 7, 16 GB RAM, 1 TB SSD, RTX 3050 graphics, 2.2 kg, "
    "battery 5 hours, price 75,000 rupees.",
    params={"temperature": 0, "max_tokens": 400}
)


# Exercise 9: Search
documents = [
    "Doc 1: The library is open from 8 AM to 8 PM on weekdays and closed on Sundays.",
    "Doc 2: Students must carry their ID cards to enter the computer lab.",
    "Doc 3: The canteen serves breakfast from 7:30 AM and lunch from 12:30 PM.",
    "Doc 4: Fees for the next semester must be paid before 30 June.",
    "Doc 5: The placement cell conducts mock interviews every Friday."
]

run_prompt(
    9,
    "Search",
    "Here are some documents:\n" + "\n".join(documents) + "\n\n"
    "Which document answers the question: 'When can I practise for job "
    "interviews?' Reply with the document number and the answer.",
    params={"temperature": 0, "max_tokens": 60}
)


# Exercise 10: Classification (few-shot)
run_prompt(
    10,
    "Classification (Few-Shot)",
    "Classify the customer message as Complaint, Query or Praise. "
    "Reply with only the class.\n\n"
    "Message: My order arrived broken and nobody is answering my calls.\n"
    "Class: Complaint\n\n"
    "Message: Do you deliver to Nagpur?\n"
    "Class: Query\n\n"
    "Message: The delivery was super fast, thank you!\n"
    "Class: Praise\n\n"
    "Message: I was charged twice for the same item.\n"
    "Class:",
    params={"temperature": 0, "max_tokens": 5},
    note="Few-shot prompting: three labelled examples are given before the new message."
)


# Exercise 11: RAG (Retrieval Augmented Generation)
knowledge_base = [
    "The Generative AI course code at MIT-WPU is AID4PE54A and it has 4 credits.",
    "The lab assessment has three components LCA1, LCA2 and LCA3, each of 5 marks.",
    "LCA1 covers assignments 1, 2 and 3. LCA2 covers assignments 4, 5 and 6.",
    "LCA3 covers assignments 7 and 8 along with the lab exam.",
    "IBM watsonx.ai is used in the lab for the prompt engineering assignment."
]

question = "Which assignments are covered in LCA2 and how many marks is it worth?"

STOP_WORDS = {"which", "what", "are", "is", "it", "in", "and", "how", "many",
              "the", "a", "of", "to", "for", "with", "has"}


def retrieve(query, documents, top_k=2):
    query_words = set(query.lower().replace("?", "").split()) - STOP_WORDS
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().replace(".", "").replace(",", "").split())
        scores.append(len(query_words & doc_words))
    ranked = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
    return [doc for score, doc in ranked[:top_k]]


retrieved = retrieve(question, knowledge_base)

print("\nRetrieved context for RAG:")
for doc in retrieved:
    print("  -", doc)

run_prompt(
    11,
    "RAG (Retrieval Augmented Generation)",
    "Answer the question using only the context below. If the answer is not "
    "in the context, say 'I don't know'.\n\n"
    "Context:\n" + "\n".join(retrieved) + "\n\n"
    "Question: " + question,
    params={"temperature": 0, "max_tokens": 80},
    note="Retrieval step: the 2 most relevant sentences are selected from a small knowledge base "
         "by keyword overlap. Generation step: the model answers using only that context."
)


results_file.close()

print("\nAll exercises completed successfully.")
print("Results saved in output/prompt_results.md")
