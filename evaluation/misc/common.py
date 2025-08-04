from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load GPT-2 model and tokenizer
model_name = "NousResearch/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

def get_logprobs(text):
    # Tokenize input text and get input IDs
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    
    # Calculate log probabilities
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    logprobs = []
    
    for i in range(1, input_ids.size(1)):  # Skip the first token
        token_id = input_ids[0, i]
        log_prob = log_probs[0, i-1, token_id].item()
        logprobs.append(log_prob)
    
    return logprobs

def fluency(prompt, generated_text):
    # Get logprobs for prompt and full generated text
    prompt_logprobs = get_logprobs(prompt)
    full_logprobs = get_logprobs(generated_text)

    # Get continuation logprobs by excluding prompt tokens
    continuation_logprobs = full_logprobs[len(prompt_logprobs):]

    # Calculate fluency score
    fluency_score = np.exp(-np.mean(continuation_logprobs))
    return fluency_score

# model used for relevance - embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_rel = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2").to(device)

# model used for success metric
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

generations = []
outputs = []
generated_text_all = []

prompts_requested_sampled = [{'text': 'I feel very furious'}]
for elem in tqdm(prompts_requested_sampled):
    if len(elem['text']) < 3:
        continue

    generated_text = "and I dont want to talk to people now"

    # Success
    continuation_sentiment_analysis = sentiment_analysis(generated_text)

    if continuation_sentiment_analysis[0]['label'] == 'POSITIVE':
        continuation_label = 1
    elif continuation_sentiment_analysis[0]['label'] == 'NEGATIVE':
        continuation_label = 0
    else:
        continuation_label = 'check_again'
    print('continuation_label: ', continuation_label)
    # Fluency
    davinci_continuation_perplexity = fluency(elem['text'], generated_text)

    # Relevance
    embeddings_prompt = model_rel.encode(elem['text'])
    embeddings_continuation = model_rel.encode(generated_text[len(elem['text']):].strip())
    similarity = cosine_similarity(embeddings_prompt.reshape(1, -1), embeddings_continuation.reshape(1, -1))[0][0]

    if True:
        print("Prompt:", elem['text'], "\n")
        # print(f"Generated Text by {method}:", generated_text[len(elem['text']):].strip(), "\n")
        # print(f"Cont Sent: {continuation_label}, Relevance: {similarity}""\n\n=====\n")

        print(f"Cont Sent: {continuation_label}, Fluency:{davinci_continuation_perplexity}, Relevance: {similarity}""\n\n=====\n")

# sentiment_samples_paths = [f"{save_dir}/{prompts_setting}/neg_dataset_sample{sample_n}.jsonl"]

# prompts_requested_sampled = {}
# for i, path in enumerate(sentiment_samples_paths):
#     dataset_num = i+1
#     filename = sentiment_samples_paths[dataset_num-1]
#     print(filename)
#     note = f"dataset{dataset_num}"
#     data_random = []
#     with open(filename, "r") as f:
#         for line in f:
#             data_random.append(json.loads(line))
#     prompts_requested_sampled[dataset_num] = data_random
#     print(f"First lines of {note}: {prompts_requested_sampled[dataset_num][:3]}")


# # Run completions
# n = sample_n
# n = 5

# max_retries = 4
# retry_count = 0

# for i, dataset in enumerate(prompts_requested_sampled):
#     dataset_num = i+1
#     note = f"dataset{dataset_num}"
#     print(note, len(prompts_requested_sampled[dataset][:n]), prompts_requested_sampled[dataset][:n])
#     while True:
#       try:
#         generations, outputs = generate_text_eval(prompts_requested_sampled=prompts_requested_sampled[dataset][:n],
#                                           method=method,
#                                           prompts_setting=prompts_setting,
#                                           model=model_llama,
#                                           max_tokens=32, # dummy, sentiment is 64 by default
#                                           sampling_kwargs=sampling_kwargs,
#                                           act_name=act_name,
#                                           prompt_add=prompt_add,
#                                           prompt_sub=prompt_sub,
#                                           coeff=coeff,
#                                           SEED=SEED,
#                                           note=note,
#                                           display=display)
#         break
#       except Exception as e:
#         print(f"Error communicating with OpenAI: {e}")

#         if retry_count >= max_retries:
#             raise Exception("Maximum number of retries exceeded")

#         time.sleep(5)  # Wait for 5 seconds before retrying


# def analyze_sentiment(outputs):
#     results_list = []
#     for fname in outputs:
#         with open(fname, 'r') as f:
#             lines = json.load(f)

#         match = re.search(r'l=(\d+)_c=(\d+)', fname)
#         if match:
#             l_value, c_value = match.groups()
#         else:
#             print(f"Could not extract l and c values from {fname}")
#             continue

#         prompt_label = []
#         cont_label = []
#         ppl = []
#         rel = []
#         total = len(lines)
#         for line in lines:
#             prompt_label.append(line['prompt_label'])
#             cont_label.append(line['continuation_label'])
#             ppl.append(line['davinci_continuation_perplexity'])
#             rel.append(line['relevance_similarity'])

#         label_agree_count = sum(1 for prompt, cont in zip(prompt_label, cont_label) if prompt != cont)
#         success = label_agree_count / total if total > 0 else 0
#         avg_ppl = sum(ppl) / total
#         avg_rel = sum(rel) / total

#         print("Statistics of", fname)
#         print(f"    Sample size: {total}")
#         print(f"    Success: {success}")
#         print(f"    Average perplexity of continuations: {avg_ppl}\n")
#         print(f"    Average relevance of continuations: {avg_rel}\n")

#         results_list.append({
#             'Filename': fname,
#             'L': l_value,
#             'C': c_value,
#             'Sample Size': total,
#             'Success': success,
#             'Average Perplexity of Continuations': avg_ppl,
#             'Average Relevance of Continuations': avg_rel
#         })

#     return results_list

# array_filename = ['NegToPos_actadd_10_l=17_c=12_Love_Hate_sentiment_dataset1.jsonl']
# array_fullpath = [f"{save_dir}/{prompts_setting}/" + fn for fn in array_filename]

# get_sent_results = analyze_sentiment(array_fullpath)