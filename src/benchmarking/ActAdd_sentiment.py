import json
import os
import random
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import transformer_lens
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import NamesFilter
from transformer_lens.utils import Slice, SliceInput
from transformers import AutoTokenizer, pipeline

from config.config import get_config

config = get_config()

# params and directories
save_dir = os.getenv("SAVE_DIR")
prompts_setting = "gpt2_sentiment"
display = True
method = "negative"

sample_n = 1000

SEED = 0
sampling_kwargs = dict(
    temperature=config["temperature"], top_p=config["top_p"], freq_penalty=1.0
)
act_name = 6  # l
coeff = 5  # c

mode_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_model(model_name: str = "gpt2", device: str = None) -> torch.nn.Module:
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Decive being used is {device}")
    model.to(device)
    return model


model = initialize_model(mode_name, device)

# sample from IMDb dataset for NegToPos (0 -> 1)
dataset = load_dataset("stanfordnlp/imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]
merged_dataset = concatenate_datasets([train_dataset, test_dataset])
print("Merged dataset has", merged_dataset.num_rows, "rows")

pos_dataset = merged_dataset.filter(lambda example: example["label"] == 1)
neg_dataset = merged_dataset.filter(lambda example: example["label"] == 0)

mode_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(mode_name)


def truncate_to_32_tokens(text):
    tokens = tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
    truncated_text = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
    return truncated_text


pos_dataset = pos_dataset.map(
    lambda example: {"text": truncate_to_32_tokens(example["text"])}
)
neg_dataset = neg_dataset.map(
    lambda example: {"text": truncate_to_32_tokens(example["text"])}
)

pos_dataset.to_json(f"{save_dir}/{prompts_setting}/pos_dataset_truncated32.jsonl")
neg_dataset.to_json(f"{save_dir}/{prompts_setting}/neg_dataset_truncated32.jsonl")

if method == "positive":
    prompt_add, prompt_sub = "Love", "Hate"
    dataset = neg_dataset
elif method == "negative":
    prompt_add, prompt_sub = "Hate", "Love"
    dataset = pos_dataset

sampled_dataset_indices = random.sample(range(len(dataset)), sample_n)
sample_dataset = dataset.select(sampled_dataset_indices)

sample_dataset.to_json(f"{save_dir}/{prompts_setting}/{method}_sample{sample_n}.jsonl")


for i in range(0, sample_n):
    test = sample_dataset[i]["text"]
    test_tokens = model.to_tokens(test).shape[1]
    print(f"{i}, t={test_tokens}, {test}")


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
            input=tokenized, max_new_tokens=64, do_sample=True, **kwargs
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
                f"Generate control text for {method}: something went wrong. Error: {error_message}. Retrying..."
            )
            break

    else:
        raise NotImplementedError

    return output


def write_eval_output_file(
    outputs,
    save_dir,
    prompts_setting,
    method,
    act_name,
    prompt_add,
    prompt_sub,
    coeff,
    num_prompts,
    note,
):
    """Writes eval output to a file"""

    def convert(o):
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError

    def clean_and_truncate(input_str, max_length=6):
        cleaned_str = re.sub("[^A-Za-z0-9]+", "", input_str)
        return cleaned_str[:max_length]

    if not os.path.exists(f"{save_dir}/{prompts_setting}"):
        os.makedirs(f"{save_dir}/{prompts_setting}")

    prefix = "gs_" if num_prompts == 50 else ""
    decode_str = f"l={act_name}_c={coeff}"
    filename = f"{save_dir}/{prompts_setting}/{method}.jsonl"
    with open(filename, "w") as f:
        print(f"c={coeff}, l={act_name}, Saved outputs to {filename}")
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

    # model used for success metric
    sentiment_analysis = pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )

    generations = []
    outputs = []
    generated_text_all = []

    for elem in tqdm(prompts_requested_sampled):
        print(f"For c={coeff}, l={act_name}")
        if len(elem["text"]) < 3:
            continue

        generated_text = generate_control_text(
            method=method,
            prompt=elem["text"],
            model=model,
            act_name=act_name,
            prompt_add=prompt_add,
            prompt_sub=prompt_sub,
            coeff=coeff,
            SEED=SEED,
            sampling_kwargs=sampling_kwargs,
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

        # Fluency
        davinci_continuation_perplexity = 0

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
                f"Cont Sent: {continuation_label}, Prompt Sent(label):{elem['label']}, Fluency:{davinci_continuation_perplexity}, Relevance: {similarity}"
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
                "davinci_continuation_perplexity": davinci_continuation_perplexity,
                "relevance_similarity": similarity,
            }
        )
    if len(prompts_requested_sampled) >= 10:
        num_prompts = len(prompts_requested_sampled)
        write_eval_output_file(
            outputs,
            save_dir,
            prompts_setting,
            method,
            act_name,
            prompt_add,
            prompt_sub,
            coeff,
            num_prompts,
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

sentiment_samples_paths = [
    f"{save_dir}/{prompts_setting}/{method}_sample{sample_n}.jsonl"
]

# dataset
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
for iter in range(4, 5):
    for i, dataset in enumerate(prompts_requested_sampled):
        dataset_num = i + 1
        note = f"dataset{dataset_num}"
        method1 = method + "_" + str(iter)
        print(
            note,
            len(prompts_requested_sampled[dataset][:n]),
            prompts_requested_sampled[dataset][:n],
        )
        while True:
            # try:
            generations, outputs = generate_text_eval(
                prompts_requested_sampled=prompts_requested_sampled[dataset][:n],
                method=method1,
                prompts_setting=prompts_setting,
                model=model,
                max_tokens=32,  # dummy, sentiment is 64 by default
                sampling_kwargs=sampling_kwargs,
                act_name=act_name,
                prompt_add=prompt_add,
                prompt_sub=prompt_sub,
                coeff=coeff,
                SEED=SEED,
                note=note,
                display=display,
            )
            break


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
        all_success.append(success)
        all_rel.append(avg_rel)
        print("Statistics of", fname)
        print(f"    Sample size: {total}")
        print(f"    Success: {success:.3f}")
        print(f"    Average Relevance: {avg_rel:.3f}\n")

        results_list.append(
            {
                "Filename": fname,
                "Sample Size": total,
                "Success": success,
                "Average Relevance of Continuations": avg_rel,
            }
        )

    avg_success = sum(all_success) / len(all_success) if all_success else 0
    avg_rel = sum(all_rel) / len(all_rel) if all_rel else 0
    print("=== Average over all files ===")
    print(f"Average Success: {avg_success:.3f}")
    print(f"Average Relevance: {avg_rel:.3f}\n")
    return results_list


array_filename = [f"{method}_{iter}.jsonl" for iter in range(5)]
array_fullpath = [f"{save_dir}/{prompts_setting}/" + fn for fn in array_filename]

get_sent_results = analyze_sentiment(array_fullpath)
