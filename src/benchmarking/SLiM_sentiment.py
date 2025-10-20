import json
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config.config import get_config
from src.model import SLiMedNet

config = get_config()
checkpoint = "./resources/checkpoints/sentiment_best.pth"

# params and directories
save_dir = "/home/raj/thesis/SLiM/SLiM/resources/evaluation_results"
prompts_setting = "sentiment"
display = True

steer_state = "positive"
mode_name = "gpt2"
method = "slim"

sample_n = 1000
SEED = 0

latencies = []
peak_mems = []
throughput = []

device = "cuda" if torch.cuda.is_available() else "cpu"


def truncate_to_32_tokens(text):
    tokens = tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
    truncated_text = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
    return truncated_text


def generate_control_text(
    prompt,
    model,
):
    if steer_state is not None:
        state_tensor = get_state_tensor(steer_state, device)
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        state_tensor=state_tensor,
    )
    return output


def write_eval_output_file(
    outputs,
    save_dir,
    note,
):
    """Writes eval output to a file"""

    def convert(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError

    if not os.path.exists(f"{save_dir}/{prompts_setting}"):
        os.makedirs(f"{save_dir}/{prompts_setting}")

    filename = f"{save_dir}/{prompts_setting}/{note}.jsonl"
    with open(filename, "w") as f:
        print(f"Saved outputs to {filename}")
        json.dump(outputs, f, default=convert)


def generate_text_eval(
    prompts_requested_sampled,
    model,
    note,
    display=False,
):
    """Generates completions for the eval set and computes all metrics (tox, perp, rel)"""

    # model used for relevance - embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_rel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)

    # model used for success metric
    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        device=device,
    )

    generations = []
    outputs = []
    generated_text_all = []

    for elem in tqdm(prompts_requested_sampled):
        if len(elem["text"]) < 3:
            continue

        generated_text = generate_control_text(
            prompt=elem["text"],
            model=model,
        )

        # Success
        continuation_sentiment_analysis = sentiment_analysis(
            generated_text[len(elem["text"]) :].strip()
        )

        if continuation_sentiment_analysis[0]["label"] == "POSITIVE":
            continuation_label = 1
        elif continuation_sentiment_analysis[0]["label"] == "NEGATIVE":
            continuation_label = 0
        else:
            continuation_label = "check_again"

        # Relevance
        embeddings_prompt = model_rel.encode(elem["text"])
        embeddings_continuation = model_rel.encode(
            generated_text[len(elem["text"]) :].strip()
        )
        similarity = cosine_similarity(
            embeddings_prompt.reshape(1, -1), embeddings_continuation.reshape(1, -1)
        )[0][0]

        if display:
            print("Prompt:", elem["text"], "\n")
            print(
                f"Generated Text by {method}:",
                generated_text[len(elem["text"]) :].strip(),
                "\n",
            )
            print(
                f"Cont Sent: {continuation_label}, Prompt Sent(label):{elem['label']}, Relevance: {similarity}"
                "\n\n=====\n"
            )

        generations.append(generated_text)

        generated_text_all.append(generated_text[len(elem["text"]) :].strip())

        outputs.append(
            {
                "content": generated_text,
                "prompt": elem["text"],
                "continuation": generated_text[len(elem["text"]) :].strip(),
                "prompt_label": elem["label"],
                "continuation_label": continuation_label,
                "continuation_sentiment_analysis": continuation_sentiment_analysis,
                "relevance_similarity": similarity,
            }
        )
    write_eval_output_file(
        outputs,
        save_dir,
        note,
    )

    return generated_text_all, outputs


def load_model_and_tokenizer(path):
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    model = SLiMedNet(state_embed_dim=config["num_states"], model=model)
    checkpoint = torch.load(path, map_location=torch.device(config["device"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_text(model, tokenizer, prompt, state_tensor=None):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(config["device"])
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            state_tensor=state_tensor,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=config["max_new_tokens"],
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=config["top_k"],
            top_p=config["top_p"],
            temperature=config["temperature"],
            do_sample=True,
        )
    torch.cuda.synchronize()
    end = time.time()

    num_new_tokens = output.shape[-1] - input_ids.shape[-1]
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    latency_ms = (end - start) / num_new_tokens * 1000
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    tokens_per_sec = num_new_tokens / (end - start)
    latencies.append(latency_ms)
    peak_mems.append(peak_mem)
    throughput.append(tokens_per_sec)
    print(
        f"Latency: {latency_ms:.2f} ms/token, Peak memory: {peak_mem:.2f} GB, Throughput: {tokens_per_sec:.2f} tokens/sec"
    )
    return output


def get_state_tensor(state, device):
    state_mapping = config["sentiment_mapping"]
    if state in state_mapping:
        state_vector = state_mapping[state]
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
        return state_tensor


def analyze_sentiment(outputs):
    results_list = []
    all_success = []
    all_rel = []
    for fname in outputs:
        with open(fname, "r") as f:
            lines = json.load(f)

        prompt_label = []
        cont_label = []
        rel = []
        total = len(lines)
        for line in lines:
            prompt_label.append(line["prompt_label"])
            cont_label.append(line["continuation_label"])
            rel.append(line["relevance_similarity"])

        label_agree_count = sum(
            1 for prompt, cont in zip(prompt_label, cont_label) if prompt != cont
        )
        success = label_agree_count / total if total > 0 else 0
        avg_rel = sum(rel) / total

        results_list.append(
            {
                "Filename": fname,
                "Sample Size": total,
                "Success": success,
                "Average Relevance of Continuations": avg_rel,
            }
        )

        all_success.append(success)
        all_rel.append(avg_rel)
        print("Statistics of", fname)
        print(f"    Sample size: {total}")
        print(f"    Success: {success:.3f}")
        print(f"    Average Relevance: {avg_rel:.3f}\n")

    # Compute averages across all files
    avg_success = sum(all_success) / len(all_success) if all_success else 0
    avg_rel = sum(all_rel) / len(all_rel) if all_rel else 0
    print("=== Average over all files ===")
    print(f"Average Success: {avg_success:.3f}")
    print(f"Average Relevance: {avg_rel:.3f}\n")

    valid_latencies = [
        l for l in latencies[1:] if l > 0
    ]  # skip the first iteration because it is the warmup
    valid_throughput = [
        t for t in throughput[1:] if t > 0
    ]  # skip the first iteration because it is the warmup
    valid_mems = peak_mems[1:]  # skip the first iteration because it is the warmup

    avg_latency = np.mean(valid_latencies)
    avg_mem = np.mean(valid_mems)
    peak_mem_overall = np.max(valid_mems)
    avg_throughput = np.mean(valid_throughput)
    print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"Average Latency: {avg_latency:.2f} ms/token")
    print(f"Average Peak Memory: {avg_mem:.2f} GB")
    print(f"Peak Memory (overall): {peak_mem_overall:.2f} GB")

    return results_list


if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("stanfordnlp/imdb")
    merged_dataset = dataset["test"]
    print("Merged dataset has", merged_dataset.num_rows, "rows")

    # Filter dataset
    pos_dataset = merged_dataset.filter(lambda example: example["label"] == 1)
    neg_dataset = merged_dataset.filter(lambda example: example["label"] == 0)

    if steer_state == "positive":
        dataset = neg_dataset
    else:
        dataset = pos_dataset

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint)
    model.to(device)

    # Truncate dataset
    dataset = dataset.map(
        lambda example: {"text": truncate_to_32_tokens(example["text"])}
    )
    dataset.to_json(
        f"{save_dir}/{prompts_setting}/{steer_state}_dataset_truncated32.jsonl"
    )

    # Sample dataset
    sampled_dataset_indices = random.sample(range(len(dataset)), sample_n)
    sampled_dataset = dataset.select(sampled_dataset_indices)
    sampled_dataset.to_json(
        f"{save_dir}/{prompts_setting}/{steer_state}_dataset_{sample_n}.jsonl"
    )
    sentiment_samples_paths = [
        f"{save_dir}/{prompts_setting}/{steer_state}_dataset_{sample_n}.jsonl"
    ]

    # Prepare dataset
    prompts_requested_sampled = {}
    for i, path in enumerate(sentiment_samples_paths):
        dataset_num = i + 1
        filename = sentiment_samples_paths[dataset_num - 1]
        print(filename)
        note = f"dataset{dataset_num}"
        data_random = []
        with open(filename, "r") as f:
            for line in f:
                data_random.append(json.loads(line))
        prompts_requested_sampled[dataset_num] = data_random
        print(f"First lines of {note}: {prompts_requested_sampled[dataset_num][:3]}")

    # Run completions
    n = sample_n
    for iter in range(5):
        for i, dataset in enumerate(prompts_requested_sampled):
            dataset_num = i + 1
            note = steer_state + "_" + str(iter)
            print(
                note,
                len(prompts_requested_sampled[dataset][:n]),
                prompts_requested_sampled[dataset][:n],
            )
            generations, outputs = generate_text_eval(
                prompts_requested_sampled=prompts_requested_sampled[dataset][:n],
                model=model,
                note=note,
                display=display,
            )

    # Analyze results
    array_filename = [f"{steer_state}_{iter}.jsonl" for iter in range(5)]
    array_fullpath = [f"{save_dir}/{prompts_setting}/" + fn for fn in array_filename]
    get_sent_results = analyze_sentiment(array_fullpath)
