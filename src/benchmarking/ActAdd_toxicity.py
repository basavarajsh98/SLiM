import json
import os
import random
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import transformer_lens
from datasets import load_dataset
from googleapiclient import discovery
from optimum.onnxruntime import ORTModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import NamesFilter
from transformer_lens.utils import Slice, SliceInput
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from config.config import get_config

config = get_config()

PERSPECTIVE_API_KEY = os.environ["PERSPECTIVE_API_KEY"]

PERSPECTIVE_API_ATTRIBUTES = ("TOXICITY",)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)

# params and directories
save_dir = os.getenv("SAVE_DIR")
prompts_setting = "toxicity"
display = True

SEED = 0
prompt_add, prompt_sub = "Love", "Hate"
sample_n = 1000
method = "slim"
type = "random"
if type == "random":
    steer_state = "0.0_random"
elif type == "toxic":
    steer_state = "0.0_top"
else:
    steer_state = "0.0_base"

sampling_kwargs = dict(
    temperature=config["temperature"], top_p=config["top_p"], freq_penalty=1.0
)
act_name = 6  # l
coeff = 5  # c

mode_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def initialize_model(
    model_name: str = "meta-llama/Meta-Llama-3-8B", device: str = None
) -> torch.nn.Module:
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Decive being used is {device}")
    model.to(device)
    return model


model = initialize_model(mode_name, device)


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


# ActAddd logic
def prepare_prompts(prompt_add: str, prompt_sub: str, model: torch.nn.Module) -> tuple:
    def tlen(prompt):
        return model.to_tokens(prompt).shape[1]

    def pad_right(prompt, length):
        return prompt + " " * (length - tlen(prompt))

    l = max(tlen(prompt_add), tlen(prompt_sub))
    return pad_right(prompt_add, l), pad_right(prompt_sub, l)


def get_resid_pre(prompt: str, layer: int, model: torch.nn.Module) -> torch.Tensor:
    name = f"blocks.{layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]


def ave_hook(resid_pre, hook, act_diff, coeff):
    if resid_pre.shape[1] == 1:
        return
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) than prompt tokens ({ppos})!"
    resid_pre[:, :apos, :] += coeff * act_diff


def hooked_generate(
    prompt_batch: List[str],
    editing_hooks: list,
    seed: int,
    model: torch.nn.Module,
    **kwargs,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    with model.hooks(fwd_hooks=editing_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            input=tokenized, max_new_tokens=32, do_sample=True, **kwargs
        )
    return result


def generate_actadd(
    model,
    prompts: List[str],
    layer: int,
    prompt_add: str,
    prompt_sub: str,
    coeff: int,
    seed: int,
    sampling_kwargs: Dict[str, Any],
) -> List[str]:
    prompt_add, prompt_sub = prepare_prompts(prompt_add, prompt_sub, model)
    act_add = get_resid_pre(prompt_add, layer, model)
    act_sub = get_resid_pre(prompt_sub, layer, model)
    act_diff = act_add - act_sub
    editing_hooks = [
        (
            f"blocks.{layer}.hook_resid_pre",
            lambda resid_pre, hook: ave_hook(resid_pre, hook, act_diff, coeff),
        )
    ]
    results_tensor = hooked_generate(
        prompts, editing_hooks, seed, model, **sampling_kwargs
    )
    results_str = model.to_string(results_tensor[:, 1:])
    # results_str_only_generated_text = [results_str[0][len(prompts[0]):]]
    return results_str


def generate_control_text(
    method,
    prompt,
    model,
    act_name,
    prompt_add,
    prompt_sub,
    coeff,
    SEED,
    sampling_kwargs,
):
    while True:
        try:
            prompt_lst = [prompt]
            output = generate_actadd(
                model,
                prompt_lst,
                act_name,
                prompt_add,
                prompt_sub,
                coeff,
                SEED,
                sampling_kwargs,
            )[0]
            break
        except Exception as e:
            error_message = str(e)
            print(
                f"Generate control text for {method}: something went wrong. Error: {error_message} Output: {output}. Retrying..."
            )
            break

    else:
        raise NotImplementedError

    return output


# Alternatives to Perspective API to compute some toxicity score
# Alternatives to Perspective API to compute some toxicity score

model_checkpoint1 = "cointegrated/rubert-tiny-toxicity"
tokenizer1 = AutoTokenizer.from_pretrained(model_checkpoint1)
model1 = AutoModelForSequenceClassification.from_pretrained(model_checkpoint1)
if torch.cuda.is_available():
    model1.cuda()


def text2tox_rubert_tiny(text, aggregate=True):
    with torch.no_grad():
        inputs = tokenizer1(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(model1.device)
        prob = torch.sigmoid(model1(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        prob = prob[0]
    if aggregate:
        return 1 - prob.T[0] * (1 - prob.T[-1])
    return prob


model_checkpoint2 = "laiyer/unbiased-toxic-roberta-onnx"
tokenizer2 = AutoTokenizer.from_pretrained(model_checkpoint2)
model2 = ORTModelForSequenceClassification.from_pretrained(
    model_checkpoint2, file_name="model.onnx"
)
classifier2 = pipeline(
    task="text-classification",
    model=model2,
    tokenizer=tokenizer2,
)


def text2tox_roberta_onnx(text, aggregate=True):
    prob = classifier2(text)[0]["score"]
    return prob


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
    method,
    prompts_setting,
    model,
    max_tokens,
    sampling_kwargs,
    act_name,
    prompt_add,
    prompt_sub,
    coeff,
    SEED,
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
            method=method,
            prompt=prompt,
            model=model,
            act_name=act_name,
            prompt_add=prompt_add,
            prompt_sub=prompt_sub,
            coeff=coeff,
            SEED=SEED,
            sampling_kwargs=sampling_kwargs,
        )
        print("Prompt:", prompt, "\n")
        print(
            f"Generated Text by {method}:",
            generated_text[len(prompt) :].strip(),
            "\n",
        )
        if len(generated_text[len(prompt) :].strip()) < 3:
            continue
        continuation_tox_perspective, _ = get_perspective_api_scores(
            generated_text[len(prompt) :].strip(), display
        )
        # Toxcity scores
        continuation_scores = {
            "perspective": continuation_tox_perspective["TOXICITY"],
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


def get_caching_hooks(
    self,
    names_filter: NamesFilter = None,
    incl_bwd: bool = False,
    device=None,
    remove_batch_dim: bool = False,
    cache: Optional[dict] = None,
    pos_slice: Union[Slice, SliceInput] = None,
) -> Tuple[dict, list, list]:
    """Creates hooks to cache activations. Note: It does not add the hooks to the model.

    Args:
        names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
        incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
        device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
        remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
        cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

    Returns:
        cache (dict): The cache where activations will be stored.
        fwd_hooks (list): The forward hooks.
        bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
    """
    if cache is None:
        cache = {}

    if not isinstance(pos_slice, Slice):
        if isinstance(
            pos_slice, int
        ):  # slicing with an int collapses the dimension so this stops the pos dimension from collapsing
            pos_slice = [pos_slice]
        pos_slice = Slice(pos_slice)

    if names_filter is None:
        names_filter = lambda name: True
    elif isinstance(names_filter, str):
        filter_str = names_filter
        names_filter = lambda name: name == filter_str
    elif isinstance(names_filter, list):
        filter_list = names_filter
        names_filter = lambda name: name in filter_list
    self.is_caching = True

    # mypy can't seem to infer this
    names_filter = cast(Callable[[str], bool], names_filter)

    def save_hook(tensor, hook, is_backward=False):
        hook_name = hook.name
        if is_backward:
            hook_name += "_grad"
        resid_stream = tensor.detach().to(device)
        if remove_batch_dim:
            resid_stream = resid_stream[0]

        # for attention heads the pos dimension is the third from last
        if (
            hook.name.endswith("hook_q")
            or hook.name.endswith("hook_k")
            or hook.name.endswith("hook_v")
            or hook.name.endswith("hook_z")
            or hook.name.endswith("hook_result")
        ):
            pos_dim = -3
        else:
            # for all other components the pos dimension is the second from last
            # including the attn scores where the dest token is the second from last
            pos_dim = -2

        if (
            tensor.dim() >= -pos_dim
        ):  # check if the residual stream has a pos dimension before trying to slice
            resid_stream = pos_slice.apply(resid_stream, dim=pos_dim)
        cache[hook_name] = resid_stream

    fwd_hooks = []
    bwd_hooks = []
    for name, hp in self.hook_dict.items():
        if names_filter(name):
            fwd_hooks.append((name, partial(save_hook, is_backward=False)))
            if incl_bwd:
                bwd_hooks.append((name, partial(save_hook, is_backward=True)))

    return cache, fwd_hooks, bwd_hooks


# Replace the original get_caching_hooks function
transformer_lens.hook_points.HookedRootModule.get_caching_hooks = get_caching_hooks


# Run completions
n = sample_n
for i, path in enumerate(prompts_requested_sampled):
    dataset_num = i + 1
    note = f"dataset{dataset_num}"
    print(note, len(prompts_requested_sampled[i][:]), prompts_requested_sampled[i][:n])

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
        method=method,
        prompts_setting=prompts_setting,
        model=model,
        max_tokens=32,  # dummy, toxicity is 32 by default
        sampling_kwargs=sampling_kwargs,
        act_name=act_name,
        prompt_add=prompt_add,
        prompt_sub=prompt_sub,
        coeff=coeff,
        SEED=SEED,
        note=note,
        display=display,
    )


def analyze_sentiment(outputs):
    results_list = []
    all_perspective_cont_tox = []
    all_rel = []
    all_std_perspective_cont_tox = []
    for fname in outputs:
        with open(fname, "r") as f:
            lines = json.load(f)

        perspective_cont_tox = []
        rel = []
        total = len(lines)
        for line in lines:
            rel.append(line["relevance_similarity"])
            perspective_cont_tox.append(line["continuation_api_scores"]["perspective"])
        perspective_cont_tox = sum(perspective_cont_tox) / total
        std_perspective_cont_tox = np.std(perspective_cont_tox)
        avg_rel = sum(rel) / total

        all_perspective_cont_tox.append(perspective_cont_tox)
        all_std_perspective_cont_tox.append(std_perspective_cont_tox)
        all_rel.append(avg_rel)

        print("Statistics of", fname)
        print(f"    Sample size: {total}")
        print(
            f"    Average Perspective toxicity of continuations: {perspective_cont_tox}"
        )
        print(
            f"    Std Perspective toxicity of continuations: {std_perspective_cont_tox}"
        )
        print(f"    Average relevance of continuations: {avg_rel}\n")

        results_list.append(
            {
                "Filename": fname,
                "Sample Size": total,
                "Average Perspective toxicity of continuations": perspective_cont_tox,
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
    avg_rel = sum(all_rel) / len(all_rel) if all_rel else 0
    print("=== Average over all files ===")
    print(
        f"Average Perspective toxicity of continuations: {avg_perspective_cont_tox:.3f}"
    )
    print(
        f"Std Perspective toxicity of continuations: {avg_std_perspective_cont_tox:.3f}"
    )
    print(f"Average Relevance: {avg_rel:.3f}\n")
    return results_list


array_filename = [f"{steer_state}_{iter}.jsonl" for iter in range(5)]
array_fullpath = [f"{save_dir}/{prompts_setting}/" + fn for fn in array_filename]
get_sent_results = analyze_sentiment(array_fullpath)
