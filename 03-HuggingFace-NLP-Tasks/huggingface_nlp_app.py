import os
os.environ["USE_TF"] = "0"

import gradio as gr
from transformers import pipeline


os.makedirs("output", exist_ok=True)


HF_TASKS = {
    "NLP": [
        "Text Classification (Sentiment Analysis)",
        "Token Classification (Named Entity Recognition)",
        "Question Answering",
        "Summarization",
        "Translation",
        "Text Generation",
        "Fill-Mask",
        "Zero-Shot Classification",
        "Feature Extraction (Embeddings)"
    ],
    "Computer Vision": [
        "Image Classification",
        "Object Detection",
        "Image Segmentation",
        "Depth Estimation",
        "Text-to-Image"
    ],
    "Audio": [
        "Automatic Speech Recognition",
        "Text-to-Speech",
        "Audio Classification"
    ],
    "Multimodal": [
        "Image-to-Text (Captioning)",
        "Visual Question Answering",
        "Document Question Answering"
    ]
}

print("Hugging Face Tasks:")
for category, tasks in HF_TASKS.items():
    print("\n" + category)
    for task in tasks:
        print("  -", task)


print("\nLoading Hugging Face pipelines...")

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

ner_tagger = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

question_answerer = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

summarizer = pipeline(
    "summarization",
    model="Falconsai/text_summarization"
)

translator = pipeline(
    "translation_en_to_fr",
    model="Helsinki-NLP/opus-mt-en-fr"
)

text_generator = pipeline(
    "text-generation",
    model="distilbert/distilgpt2"
)

mask_filler = pipeline(
    "fill-mask",
    model="distilbert/distilbert-base-uncased"
)

print("All pipelines loaded.")


def sentiment_analysis(text):
    result = sentiment_analyzer(text)[0]
    return f"Label: {result['label']}\nConfidence: {result['score']:.4f}"


def named_entity_recognition(text):
    entities = ner_tagger(text)

    if len(entities) == 0:
        return "No named entities found."

    lines = []
    for entity in entities:
        lines.append(
            f"{entity['word']} -> {entity['entity_group']} "
            f"({entity['score']:.4f})"
        )
    return "\n".join(lines)


def question_answering(context, question):
    if question.strip() == "":
        return "Please enter a question for Question Answering."

    result = question_answerer(question=question, context=context)
    return f"Answer: {result['answer']}\nConfidence: {result['score']:.4f}"


def summarization(text):
    result = summarizer(
        text,
        max_new_tokens=100,
        min_length=20,
        do_sample=False
    )[0]
    return result["summary_text"]


def translation(text):
    result = translator(text)[0]
    return "French: " + result["translation_text"]


def text_generation(text):
    result = text_generator(
        text,
        max_new_tokens=40,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        pad_token_id=50256
    )[0]
    return result["generated_text"].strip()


def fill_mask(text):
    if "[MASK]" not in text:
        return "Please include the [MASK] token in the sentence."

    predictions = mask_filler(text)

    lines = []
    for prediction in predictions:
        lines.append(
            f"{prediction['sequence']} ({prediction['score']:.4f})"
        )
    return "\n".join(lines)


def run_task(task, text, question=""):
    if text.strip() == "":
        return "Please enter some text."

    if task == "Sentiment Analysis":
        return sentiment_analysis(text)
    elif task == "Named Entity Recognition":
        return named_entity_recognition(text)
    elif task == "Question Answering":
        return question_answering(text, question)
    elif task == "Summarization":
        return summarization(text)
    elif task == "Translation (English to French)":
        return translation(text)
    elif task == "Text Generation":
        return text_generation(text)
    elif task == "Fill-Mask":
        return fill_mask(text)
    else:
        return "Unknown task."


TASK_NAMES = [
    "Sentiment Analysis",
    "Named Entity Recognition",
    "Question Answering",
    "Summarization",
    "Translation (English to French)",
    "Text Generation",
    "Fill-Mask"
]

SAMPLE_TEXT = (
    "Hugging Face is a company based in New York City that develops tools "
    "for building machine learning applications. Its Transformers library "
    "provides thousands of pretrained models for tasks such as text "
    "classification, question answering, summarization and translation. "
    "Clement Delangue, Julien Chaumond and Thomas Wolf founded the company "
    "in 2016. Today, researchers and developers around the world use the "
    "Hugging Face Hub to share models, datasets and demo applications."
)

SAMPLES = [
    ("Sentiment Analysis",
     "I really enjoyed this lab, the Hugging Face pipelines are very easy to use!",
     ""),
    ("Named Entity Recognition", SAMPLE_TEXT, ""),
    ("Question Answering", SAMPLE_TEXT, "Who founded Hugging Face?"),
    ("Summarization", SAMPLE_TEXT, ""),
    ("Translation (English to French)",
     "Generative AI can create new text, images and music.",
     ""),
    ("Text Generation", "Generative Artificial Intelligence is", ""),
    ("Fill-Mask", "Hugging Face provides [MASK] models for NLP tasks.", "")
]


print("\nPipeline Example:")
print('classifier = pipeline("sentiment-analysis")')
print('classifier("I love Generative AI")')
print(sentiment_analyzer("I love Generative AI"))


print("\nRunning sample inputs for all NLP tasks...")

with open("output/nlp_results.txt", "w", encoding="utf-8") as file:
    for task, text, question in SAMPLES:
        output = run_task(task, text, question)

        record = "=" * 60 + "\n"
        record += "Task: " + task + "\n"
        record += "Input: " + text + "\n"
        if question != "":
            record += "Question: " + question + "\n"
        record += "Output:\n" + output + "\n"

        print(record)
        file.write(record + "\n")

print("Sample results saved in output/nlp_results.txt")


app = gr.Interface(
    fn=run_task,
    inputs=[
        gr.Dropdown(
            choices=TASK_NAMES,
            value="Sentiment Analysis",
            label="Select NLP Task"
        ),
        gr.Textbox(
            lines=6,
            label="Input Text / Context",
            placeholder="Enter your text here..."
        ),
        gr.Textbox(
            lines=1,
            label="Question (only for Question Answering)"
        )
    ],
    outputs=gr.Textbox(lines=8, label="Output"),
    title="NLP Subtasks using Hugging Face Transformers",
    description=(
        "Select an NLP task, enter text and click Submit. "
        "For Fill-Mask, use [MASK] in the sentence."
    ),
    examples=[list(sample) for sample in SAMPLES],
    flagging_mode="never"
)

app.launch()
