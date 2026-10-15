import os
os.environ["USE_TF"] = "0"

import csv
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import pipeline


os.makedirs("output", exist_ok=True)


APPLICATIONS = [
    {
        "category": "Synthetic Data Generation",
        "space": "Intel/synthetic-data-generator",
        "input": "Description of the dataset (domain, fields, number of rows)",
        "output": "Synthetic tabular/text dataset (CSV)",
        "architecture": "Decoder-only Transformer LLM (Llama 3.2 / SmolLM2 / Gemma 3)",
        "local_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "uses": "Training data for ML models, testing software, privacy-safe data sharing"
    },
    {
        "category": "Code Generation",
        "space": "Qwen/Qwen3-Coder-WebDev",
        "input": "Natural language description of the program",
        "output": "Source code (Python / HTML / JavaScript)",
        "architecture": "Decoder-only Transformer LLM trained on code (Qwen Coder)",
        "local_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "uses": "Coding assistants, auto-completion, code explanation, prototyping"
    },
    {
        "category": "Depth Estimation",
        "space": "depth-anything/Depth-Anything-V2",
        "input": "Single RGB image",
        "output": "Depth map image (near = bright, far = dark)",
        "architecture": "Vision Transformer (DINOv2 ViT) encoder + DPT decoder",
        "local_model": "depth-anything/Depth-Anything-V2-Small-hf",
        "uses": "Robotics, autonomous driving, AR/VR, 3D reconstruction, photo effects"
    },
    {
        "category": "Text Summarization",
        "space": "pszemraj/summarize-long-text",
        "input": "Long text / article / document",
        "output": "Short summary text",
        "architecture": "Encoder-Decoder Transformer (LongT5 / T5)",
        "local_model": "Falconsai/text_summarization",
        "uses": "News summaries, meeting notes, research papers, legal documents"
    },
    {
        "category": "Video Processing",
        "space": "depth-anything/Video-Depth-Anything",
        "input": "Video file (MP4)",
        "output": "Depth video (depth map for every frame)",
        "architecture": "Depth Anything V2 ViT encoder + spatio-temporal head",
        "local_model": "depth-anything/Depth-Anything-V2-Small-hf (frame by frame)",
        "uses": "Video editing, 3D video, robotics navigation, visual effects"
    }
]


def print_study_table():
    print("=" * 70)
    print("Study of GenAI Applications from Hugging Face Spaces")
    print("=" * 70)

    for app in APPLICATIONS:
        print("\nCategory     :", app["category"])
        print("Space        : https://huggingface.co/spaces/" + app["space"])
        print("Input        :", app["input"])
        print("Output       :", app["output"])
        print("Architecture :", app["architecture"])
        print("Local Model  :", app["local_model"])
        print("Real-time Use:", app["uses"])

    with open("output/spaces_study.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=APPLICATIONS[0].keys())
        writer.writeheader()
        writer.writerows(APPLICATIONS)


def chat(generator, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    result = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    return result[0]["generated_text"][-1]["content"].strip()


def synthetic_data_generation():
    print("\n" + "=" * 70)
    print("1. Synthetic Data Generation")
    print("=" * 70)

    generator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct"
    )

    products = ["Laptop", "Headphones", "Smartphone", "Coffee Maker", "Running Shoes"]
    sentiments = ["positive", "negative"]

    rows = []
    for product in products:
        for sentiment in sentiments:
            prompt = (
                f"Write one short {sentiment} customer review for a {product}. "
                "Reply with only the review in under 30 words."
            )
            review = chat(generator, prompt, 80).replace("\n", " ")
            rows.append([product, sentiment, review])
            print(f"{product} | {sentiment} | {review}")

    with open("output/synthetic_reviews.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["product", "sentiment", "review"])
        writer.writerows(rows)

    print("Saved: output/synthetic_reviews.csv")


def code_generation():
    print("\n" + "=" * 70)
    print("2. Code Generation")
    print("=" * 70)

    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-Coder-0.5B-Instruct"
    )

    prompt = (
        "Write a Python function is_prime(n) that checks whether a number "
        "is prime, and print all prime numbers from 1 to 50. "
        "Reply with only the Python code."
    )
    code = chat(generator, prompt, 300)

    print("Prompt:", prompt)
    print("\nGenerated Code:\n")
    print(code)

    with open("output/generated_code.md", "w", encoding="utf-8") as file:
        file.write("Prompt: " + prompt + "\n\n" + code + "\n")

    print("Saved: output/generated_code.md")


def depth_estimation(depth_estimator):
    print("\n" + "=" * 70)
    print("3. Depth Estimation")
    print("=" * 70)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    result = depth_estimator(image)
    depth = result["depth"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap="inferno")
    plt.title("Generated Depth Map")
    plt.axis("off")

    plt.suptitle("Depth Estimation - Depth Anything V2")
    plt.tight_layout()
    plt.savefig("output/depth_estimation.png")
    plt.show()
    plt.close()

    print("Input image size:", image.size)
    print("Depth map size  :", depth.size)
    print("Saved: output/depth_estimation.png")


def text_summarization():
    print("\n" + "=" * 70)
    print("4. Text Summarization")
    print("=" * 70)

    summarizer = pipeline(
        "summarization",
        model="Falconsai/text_summarization"
    )

    article = (
        "Generative Artificial Intelligence refers to models that can create "
        "new content such as text, images, audio, code and video. Unlike "
        "traditional AI systems that only classify or predict, generative "
        "models learn the patterns of their training data and use them to "
        "produce new samples. Popular examples include large language models "
        "like GPT and Llama, image generators like Stable Diffusion, and "
        "code assistants. Hugging Face Spaces allows developers to host "
        "demos of these models so that anyone can try them in a web browser "
        "without installing anything. While generative AI improves "
        "productivity in education, healthcare and software development, it "
        "also raises concerns about misinformation, bias and copyright, so "
        "its outputs must be checked carefully before use."
    )

    summary = summarizer(
        article,
        max_new_tokens=100,
        min_length=25,
        do_sample=False
    )[0]["summary_text"]

    print("Input article length :", len(article.split()), "words")
    print("Summary length       :", len(summary.split()), "words")
    print("\nSummary:\n" + summary)

    with open("output/summary.txt", "w", encoding="utf-8") as file:
        file.write("Article:\n" + article + "\n\nSummary:\n" + summary + "\n")

    print("Saved: output/summary.txt")


def video_processing(depth_estimator):
    print("\n" + "=" * 70)
    print("5. Video Processing")
    print("=" * 70)

    video_path = hf_hub_download(
        repo_id="depth-anything/Video-Depth-Anything",
        filename="assets/example_videos/davis_rollercoaster.mp4",
        repo_type="space"
    )

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    frames = []
    depth_frames = []

    while len(frames) < 40:
        success, frame = video.read()
        if not success:
            break

        frame = cv2.resize(frame, (480, 270))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        depth = depth_estimator(Image.fromarray(rgb_frame))["depth"]
        depth = np.array(depth.resize((480, 270)))
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        frames.append(rgb_frame)
        depth_frames.append(depth_color)

    video.release()

    writer = cv2.VideoWriter(
        "output/depth_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (960, 270)
    )
    for frame, depth_color in zip(frames, depth_frames):
        combined = np.hstack([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), depth_color])
        writer.write(combined)
    writer.release()

    plt.figure(figsize=(12, 5))
    indexes = np.linspace(0, len(frames) - 1, 4).astype(int)

    for i, index in enumerate(indexes):
        plt.subplot(2, 4, i + 1)
        plt.imshow(frames[index])
        plt.title(f"Frame {index}")
        plt.axis("off")

        plt.subplot(2, 4, i + 5)
        plt.imshow(cv2.cvtColor(depth_frames[index], cv2.COLOR_BGR2RGB))
        plt.title(f"Depth {index}")
        plt.axis("off")

    plt.suptitle("Video Processing - Frame-wise Depth Estimation")
    plt.tight_layout()
    plt.savefig("output/video_depth_frames.png")
    plt.show()
    plt.close()

    print("Video FPS        :", fps)
    print("Frames processed :", len(frames))
    print("Saved: output/depth_video.mp4")
    print("Saved: output/video_depth_frames.png")


print_study_table()

synthetic_data_generation()
code_generation()

depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)
depth_estimation(depth_estimator)

text_summarization()
video_processing(depth_estimator)

print("\nStudy completed successfully.")
print("Results saved in the output folder.")
