import json
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from googleapiclient import discovery
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from config.config import get_config
from src.model import SLiMedNet

config = get_config()
checkpoint = os.getenv("CHECKPOINT")

# params and directories
save_dir = os.getenv("SAVE_DIR")
prompts_setting = "toxicity"

type = "random"
if type == "random":
    steer_state = "0.0_random"
elif type == "toxic":
    steer_state = "0.0_top"
else:
    steer_state = "0.0_base"
display = True

mode_name = "gpt2"
method = "slim"

sample_n = 1000
SEED = 0

latencies = []
peak_mems = []
throughput = []

PERSPECTIVE_API_KEY = os.environ["PERSPECTIVE_API_KEY"]
PERSPECTIVE_API_ATTRIBUTES = ("TOXICITY",)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Alternatives to Perspective API to compute some toxicity score
model_checkpoint1 = "cointegrated/rubert-tiny-toxicity"
tokenizer1 = AutoTokenizer.from_pretrained(model_checkpoint1)
model1 = AutoModelForSequenceClassification.from_pretrained(model_checkpoint1)
if torch.cuda.is_available():
    model1.cuda()

# Alternatives to Perspective API to compute some toxicity score
model2 = pipeline(
    "text-classification", model="s-nlp/roberta_toxicity_classifier", device=device
)


def top_toxicity_dataset(num_sentences=1000, display=False):
    """
    Extracts the top toxic sentences from the real-toxicity-prompts dataset
    based on Perspective API attribute scores.
    """
    # Load dataset
    dataset = load_dataset("allenai/real-toxicity-prompts")["train"]

    # Merge prompt + continuation text, and take max toxicity attributes
    def merge_text_and_scores(example):
        merged_text = (example["prompt"]["text"] or "") + (
            example["continuation"]["text"] or ""
        )
        merged_scores = {}
        for attr in ["toxicity"]:
            p_score = example["prompt"].get(attr)
            c_score = example["continuation"].get(attr)
            if p_score is not None and c_score is not None:
                merged_scores[attr] = max(p_score, c_score)
            elif p_score is not None:
                merged_scores[attr] = p_score
            elif c_score is not None:
                merged_scores[attr] = c_score
            else:
                merged_scores[attr] = None
        return {"text": merged_text, **merged_scores}

    dataset = dataset.map(merge_text_and_scores)

    # Collect top toxic sentences
    top_sentences = set()
    for attr in ["toxicity"]:
        scored_data = dataset.filter(lambda x: x[attr] is not None)
        scored_data = scored_data.sort(attr, reverse=True)
        top_rows = scored_data.select(range(min(num_sentences, len(scored_data))))
        top_sentences.update([r["text"].strip() for r in top_rows])

        if display:
            print(f"\nTop 10 sentences according to '{attr}' score:\n")
            for s in top_rows["text"][:10]:
                print(s.strip())
            print("\n===\n")

    # Create a dataset of unique top sentences
    top_sentences = list(top_sentences)
    print(f"Collected {len(top_sentences)} unique high-toxicity sentences.")

    return top_sentences


def truncate_to_32_tokens(text):
    tokens = tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
    truncated_text = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
    return truncated_text


def unpack_scores(response):
    """Extracts Perspective API scores from request response"""

    if not response:
        return None

    attribute_scores = response["attributeScores"].items()

    summary_scores = {}
    span_scores = {}
    for attribute, scores in attribute_scores:
        assert scores["summaryScore"]["type"] == "PROBABILITY"
        summary_scores[attribute] = scores["summaryScore"]["value"]

        for span_score_dict in scores["spanScores"]:
            assert span_score_dict["score"]["type"] == "PROBABILITY"
            span = (span_score_dict["begin"], span_score_dict["end"])
            span_scores.setdefault(span, {})[attribute] = span_score_dict["score"][
                "value"
            ]

    return summary_scores, span_scores


def get_perspective_api_scores(content, display=False):
    """Computes Toxicity score using the Perspective API"""
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        "comment": {"text": content},
        "requestedAttributes": dict(
            [(attribute, {}) for attribute in PERSPECTIVE_API_ATTRIBUTES]
        ),
        "languages": ["en"],
    }

    response = client.comments().analyze(body=analyze_request).execute()

    summary_scores, span_scores = unpack_scores(response)

    return summary_scores, span_scores


def text2tox_rubert_tiny(text, aggregate=True):
    with torch.no_grad():
        inputs = tokenizer1(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(model1.device)
        prob = torch.sigmoid(model1(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        prob = prob[0]
    if aggregate:
        return 1 - prob.T[0]
    return prob


def text2tox_roberta_onnx(text, aggregate=True):
    output = model2(text)
    if output[0]["label"] == "neutral":
        return 1 - output[0]["score"]
    else:
        return output[0]["score"]


def generate_control_text(
    prompt,
    model,
):
    if steer_state == "0.0_base":
        state_tensor = None
    else:
        state_tensor = torch.FloatTensor([0.0]).unsqueeze(0).to(device)
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

    generations = []
    outputs = []
    generated_text_all = []

    for prompt in tqdm(prompts_requested_sampled):
        if len(prompt) < 3:
            continue
        time.sleep(1.1)
        generated_text = generate_control_text(
            prompt=prompt,
            model=model,
        )
        print("Prompt:", prompt, "\n")
        print(
            f"Generated Text by {method}:",
            generated_text[len(prompt) :].strip(),
            "\n",
        )
        # Rubert and Roberta scores - Toxicity
        # summary_tox_perspective, _ = get_perspective_api_scores(generated_text, display)
        if len(generated_text[len(prompt) :].strip()) < 3:
            continue
        continuation_tox_perspective, _ = get_perspective_api_scores(
            generated_text[len(prompt) :].strip(), display
        )
        summary_tox_rubert = text2tox_rubert_tiny(generated_text)
        summary_tox_roberta = text2tox_roberta_onnx(generated_text)
        summary_scores = {"rubert": summary_tox_rubert, "roberta": summary_tox_roberta}
        continuation_tox_rubert = text2tox_rubert_tiny(
            generated_text[len(prompt) :].strip()
        )
        continuation_tox_roberta = text2tox_roberta_onnx(
            generated_text[len(prompt) :].strip()
        )
        # Toxcity scores

        summary_scores = {
            "perspective": 0,
            "rubert": summary_tox_rubert,
            "roberta": summary_tox_roberta,
        }
        continuation_scores = {
            "perspective": continuation_tox_perspective["TOXICITY"],
            "rubert": continuation_tox_rubert,
            "roberta": continuation_tox_roberta,
        }

        # Relevance
        embeddings_prompt = model_rel.encode(prompt)
        embeddings_continuation = model_rel.encode(
            generated_text[len(prompt) :].strip()
        )
        similarity = cosine_similarity(
            embeddings_prompt.reshape(1, -1), embeddings_continuation.reshape(1, -1)
        )[0][0]

        if display:
            print(
                f"Continuation Toxicity: {continuation_scores}, Relevance: {similarity}"
                "\n\n=====\n"
            )

        generations.append(generated_text)

        generated_text_all.append(generated_text[len(prompt) :].strip())

        outputs.append(
            {
                "prompt": prompt,
                "content": generated_text,
                "api_scores": dict(sorted(summary_scores.items())),
                "continuation_api_scores": dict(sorted(continuation_scores.items())),
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


def generate_text(model, tokenizer, prompt, state_tensor=None, max_new_tokens=32):
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
            repetition_penalty=1.0,
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
    state_mapping = config["toxicity_mapping"]
    if state in state_mapping:
        state_vector = state_mapping[state]
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
        return state_tensor


def analyze_sentiment(outputs):
    results_list = []
    all_rubert_cont_tox = []
    all_perspective_cont_tox = []
    all_rel = []
    all_roberta_cont_tox = []
    all_std_rubert_cont_tox = []
    all_std_roberta_cont_tox = []
    all_std_perspective_cont_tox = []
    for fname in outputs:
        with open(fname, "r") as f:
            lines = json.load(f)

        roberta_cont_tox = []
        rubert_cont_tox = []
        perspective_cont_tox = []
        rel = []
        total = len(lines)
        for line in lines:
            rel.append(line["relevance_similarity"])
            perspective_cont_tox.append(line["continuation_api_scores"]["perspective"])
            rubert_cont_tox.append(line["continuation_api_scores"]["rubert"])
            roberta_cont_tox.append(line["continuation_api_scores"]["roberta"])
        rubert_cont_tox = sum(rubert_cont_tox) / total
        perspective_cont_tox = sum(perspective_cont_tox) / total
        roberta_cont_tox = sum(roberta_cont_tox) / total
        std_rubert_cont_tox = np.std(rubert_cont_tox)
        std_roberta_cont_tox = np.std(roberta_cont_tox)
        std_perspective_cont_tox = np.std(perspective_cont_tox)
        std_perspective_cont_tox = np.std(perspective_cont_tox)
        avg_rel = sum(rel) / total

        all_perspective_cont_tox.append(perspective_cont_tox)
        all_rubert_cont_tox.append(rubert_cont_tox)
        all_roberta_cont_tox.append(roberta_cont_tox)
        all_std_rubert_cont_tox.append(std_rubert_cont_tox)
        all_std_perspective_cont_tox.append(std_perspective_cont_tox)
        all_std_roberta_cont_tox.append(std_roberta_cont_tox)
        all_rel.append(avg_rel)

        print("Statistics of", fname)
        print(f"    Sample size: {total}")
        print(
            f"    Average Perspective toxicity of continuations: {perspective_cont_tox}"
        )
        print(f"    Average Rubert toxicity of continuations: {rubert_cont_tox}")
        print(f"    Average Roberta toxicity of continuations: {roberta_cont_tox}")
        print(f"    Std toxicity of continuations: {std_rubert_cont_tox}")
        print(
            f"    Std Perspective toxicity of continuations: {std_perspective_cont_tox}"
        )
        print(f"    Std Roberta toxicity of continuations: {std_roberta_cont_tox}")
        print(f"    Average relevance of continuations: {avg_rel}\n")

        results_list.append(
            {
                "Filename": fname,
                "Sample Size": total,
                "Average Perspective toxicity of continuations": perspective_cont_tox,
                "Average Rubert toxicity of continuations": rubert_cont_tox,
                "Average Roberta toxicity of continuations": roberta_cont_tox,
                "Std Toxicity of Continuations": std_rubert_cont_tox,
                "Average Relevance of Continuations": avg_rel,
            }
        )

    avg_perspective_cont_tox = (
        sum(all_perspective_cont_tox) / len(all_perspective_cont_tox)
        if all_perspective_cont_tox
        else 0
    )
    avg_std_perspective_cont_tox = (
        sum(all_std_perspective_cont_tox) / len(all_std_perspective_cont_tox)
        if all_std_perspective_cont_tox
        else 0
    )
    # Compute averages across all files
    avg_rubert_cont_tox = (
        sum(all_rubert_cont_tox) / len(all_rubert_cont_tox)
        if all_rubert_cont_tox
        else 0
    )
    avg_roberta_cont_tox = (
        sum(all_roberta_cont_tox) / len(all_roberta_cont_tox)
        if all_roberta_cont_tox
        else 0
    )
    avg_std_rubert_cont_tox = (
        sum(all_std_rubert_cont_tox) / len(all_std_rubert_cont_tox)
        if all_std_rubert_cont_tox
        else 0
    )
    avg_std_roberta_cont_tox = (
        sum(all_std_roberta_cont_tox) / len(all_std_roberta_cont_tox)
        if all_std_roberta_cont_tox
        else 0
    )
    avg_rel = sum(all_rel) / len(all_rel) if all_rel else 0
    print("=== Average over all files ===")
    print(
        f"Average Perspective toxicity of continuations: {avg_perspective_cont_tox:.3f}"
    )
    print(f"Average Rubert toxicity of continuations: {avg_rubert_cont_tox:.3f}")
    print(f"Average Roberta toxicity of continuations: {avg_roberta_cont_tox:.3f}")
    print(f"Std Toxicity of Continuations: {avg_std_rubert_cont_tox:.3f}")
    print(
        f"Std Perspective toxicity of continuations: {avg_std_perspective_cont_tox:.3f}"
    )
    print(f"Std Roberta toxicity of continuations: {avg_std_roberta_cont_tox:.3f}")
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
    if type == "random":
        dataset = load_dataset("allenai/real-toxicity-prompts")["train"]
        sampled_tox_dataset_indices = random.sample(range(len(dataset)), sample_n)
        sampled_tox_dataset = dataset.select(sampled_tox_dataset_indices)
        filename = f"{save_dir}/{prompts_setting}/{method}_random_{sample_n}.jsonl"
        sampled_tox_dataset.to_json(
            f"{save_dir}/{prompts_setting}/tox_prompts_random_{sample_n}.jsonl"
        )  # save dataset

        prompts_requested_sampled_1 = [d["prompt"]["text"] for d in sampled_tox_dataset]
        prompts_requested_sampled = [prompts_requested_sampled_1]
    else:
        prompts_requested_sampled = [top_toxicity_dataset()]

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint)
    model.to(device)

    # Prepare dataset
    sentiment_samples_paths = [
        f"{save_dir}/{prompts_setting}/{steer_state}_dataset_{sample_n}.jsonl"
    ]
    for i, path in enumerate(prompts_requested_sampled):
        dataset_num = i + 1
        note = f"dataset{dataset_num}"
        print(
            note, len(prompts_requested_sampled[i][:]), prompts_requested_sampled[i][:n]
        )

    # Run completions
    n = sample_n
    for iter in range(5):
        for i, dataset in enumerate(prompts_requested_sampled):
            print(dataset[i])
            dataset_num = i + 1
            note = steer_state + "_" + str(iter)
            print(
                note,
                len(prompts_requested_sampled[i][:n]),
                prompts_requested_sampled[i][:n],
            )
            generations, outputs = generate_text_eval(
                prompts_requested_sampled=prompts_requested_sampled[i][:n],
                model=model,
                note=note,
                display=display,
            )

    # Analyze results
    array_filename = [f"{steer_state}_{iter}.jsonl" for iter in range(5)]
    array_fullpath = [f"{save_dir}/{prompts_setting}/" + fn for fn in array_filename]
    get_sent_results = analyze_sentiment(array_fullpath)
