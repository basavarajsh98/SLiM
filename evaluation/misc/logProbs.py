# from transformers import pipeline
# import torch
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import torch
# from transformers import AutoTokenizer
# from src.model import SLiMedNet
# from config.config import get_config
# from torch.amp import autocast
# from transformers import AutoModelForCausalLM
# import warnings
# import numpy as np
# warnings.filterwarnings("ignore")
# from sklearn.metrics import roc_auc_score, roc_curve
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import label_binarize

# config = get_config()

# def load_model_and_tokenizer(path):
#     model = AutoModelForCausalLM.from_pretrained(config['model_name'])
#     model = SLiMedNet(state_embed_dim=10, model=model)
#     checkpoint = torch.load(path, map_location=torch.device(config['device']))
#     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#     model.eval()
    
#     tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
#     tokenizer.pad_token = tokenizer.eos_token
#     return model, tokenizer

# def generate_text(model, tokenizer, prompt, state_tensor=None, num_generations=100):
#     model.eval()
#     inputs = tokenizer(prompt, return_tensors='pt').to(config['device'])
#     input_ids = inputs['input_ids']
#     attention_mask = inputs["attention_mask"]
#     with torch.no_grad():
#         if state_tensor is not None:
#             output = model.generate(
#                 input_ids, 
#                 state_tensor=state_tensor, 
#                 attention_mask=attention_mask,
#                 pad_token_id=tokenizer.eos_token_id,
#                 max_length=config['max_length'], 
#                 # max_new_tokens = 100,
#                 num_return_sequences=num_generations,  
#                 no_repeat_ngram_size=2,  
#                 top_k=50,
#                 top_p=0.9,
#                 temperature=1,
#                 do_sample=True
#             )
#         else:
#             output = model.generate(
#                 input_ids, 
#                 attention_mask=attention_mask,
#                 pad_token_id=tokenizer.eos_token_id,
#                 max_length=config['max_length'], 
#                 max_new_tokens = 100,
#                 num_return_sequences=num_generations,  
#                 no_repeat_ngram_size=2,  
#                 top_k=50,
#                 top_p=0.9,
#                 temperature=1,
#                 do_sample=True
#             )
#     return [tokenizer.decode(g, skip_special_tokens=True) for g in output]

# def get_state_tensor(state, device='cuda'):
#     state_mapping = config['topic_mapping']
#     if state in state_mapping:
#         state_vector = state_mapping[state]
#         state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
#         return state_tensor
#     else:
#         print("not a state")
#         return None

# def compute_perplexicity(text, model, tokenizer, state_tensor=None):
#     """
#     Calculate log probabilities of each token in the input text with optional state_tensor.
#     """
#     from torch.nn import CrossEntropyLoss
#     loss_fct = CrossEntropyLoss(reduction="none")
#     encodings = tokenizer(
#         text[:100],
#         add_special_tokens=False,
#         padding=True,
#         truncation=True,
#         max_length=100,
#         return_tensors="pt",
#         return_attention_mask=True,
#     ).to(device)
#     encoded_texts = encodings["input_ids"]
#     attn_mask = encodings["attention_mask"]
#     labels = encoded_texts

#     with torch.no_grad():
#         out_logits = model(encoded_texts, attention_mask=attn_mask, state_tensor=state_tensor)

#     shift_logits = out_logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
#     shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

#     perplexity_batch = torch.exp(
#         (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
#         / shift_attention_mask_batch.sum(1)
#     )

#     ppl = perplexity_batch.item()

#     return ppl


# def fluency(prompt, generated_text):
#     # Get logprobs for prompt and full generated text
#     prompt_logprobs = get_logprobs(prompt)
#     full_logprobs = get_logprobs(generated_text)

#     # Get continuation logprobs by excluding prompt tokens
#     continuation_logprobs = full_logprobs[len(prompt_logprobs):]

#     # Calculate fluency score
#     fluency_score = np.exp(-np.mean(continuation_logprobs))
#     return fluency_score

# def calculate_perplexity(log_probs):
#     """
#     Calculate the perplexity for a given list of log probabilities.
#     """
#     return np.exp(-np.mean(log_probs))

# def generate_and_calculate_perplexity_ratio(model, tokenizer, prompt, states, num_generations=100):
#     """
#     Generate text with and without emotion state input and calculate the perplexity ratio.
#     """
#     ratios = {}
    
#     for state in states:
#         # Get state tensor if applicable
#         state_tensor = get_state_tensor(state)
        
#         # Generate text with emotion steering vector
#         generated_texts_with_state = generate_text(model, tokenizer, prompt, state_tensor=state_tensor, num_generations=num_generations)
        
#         # Generate text without emotion steering vector
#         generated_texts_without_state = generate_text(model, tokenizer, prompt, state_tensor=None, num_generations=num_generations)

#         # Calculate perplexities for each case by feeding back into the model
#         perplexities_with_state = [compute_perplexicity(text, model, tokenizer, state_tensor=state_tensor) for text in generated_texts_with_state]
#         perplexities_without_state = [compute_perplexicity(text, model, tokenizer, state_tensor=None) for text in generated_texts_without_state]
        
#         # Calculate the average perplexity and the ratio
#         avg_perplexity_with_state = np.mean(perplexities_with_state)
#         avg_perplexity_without_state = np.mean(perplexities_without_state)
#         ratio = avg_perplexity_with_state / avg_perplexity_without_state
        
#         ratios[state] = ratio
#         print(f"Completed evaluating {state}")
#     return ratios


# def plot_perplexity_ratios(ratios):
#     """
#     Plot the perplexity ratios for each state.
#     """
#     relevance_increase_df = pd.DataFrame({
#         "State": list(ratios.keys()),
#         "Perplexity Ratio": list(ratios.values())
#     })
#     # Plot the relevance increase
#     plt.figure(figsize=(12, 6))
#     sns.barplot(data=relevance_increase_df, x="State", y="Perplexity Ratio", palette="pastel")
#     # Add labels and title
#     plt.xlabel('Emotion State')
#     plt.ylabel('Perplexity Ratio (With State / Without State)')
#     plt.title('Perplexity Ratio for Emotion States')
#     # plt.xticks(rotation=70)
#     plt.xticks(rotation=45)
#     # Add a baseline line at ratio=1
#     plt.axhline(1, color='red', linestyle='--', label='Baseline (1)')
#     # plt.ylim(0, max(values) + 0.2)  # Adjust the y-axis limit for better visualization
#     plt.legend()
#     plt.tight_layout()

#     # Display the plot
#     plt.savefig('perplexity_ratio.png')

# import torch
# import torch.nn.functional as F

# # Example usage in your KL divergence function
# def calculate_kl_divergence(logits_with_state, logits_without_state):
#     # min_length = min(logits_with_state.size(1), logits_without_state.size(1))
#     # logits_with_state = logits_with_state[:, :min_length, :]
#     # logits_without_state = logits_without_state[:, :min_length, :]
#     # Calculate log softmax (log probabilities)
#     log_probs_with_state = F.log_softmax(logits_with_state, dim=-1)
#     log_probs_without_state = F.log_softmax(logits_without_state, dim=-1)
#     kl_divergence = F.kl_div(log_probs_without_state, log_probs_with_state, reduction='batchmean')
#     return kl_divergence.item()


# def evaluate_kl_divergence(model, tokenizer, prompt, states, num_generations=100):
#     kl_divergences = {}
#     kl_divergences_per_state = {state: [] for state in states}

#     for state in states:
#         state_tensor = get_state_tensor(state)
#         kl_divergence_values = []

#         for _ in range(num_generations):
#             # Generate text with and without state tensor
#             generated_text_with_state = generate_text(model, tokenizer, prompt, state_tensor=state_tensor, num_generations=1)[0]
#             inputs = tokenizer(generated_text_with_state, return_tensors="pt",).to(device)

#             # Get log probabilities for each token in both generations
#             logits_with_state = model(**inputs, state_tensor=state_tensor)
#             logits_without_state = model(**inputs, state_tensor=None)

#             # Calculate log softmax probabilities for each
#             log_probs_state = torch.log_softmax(logits_with_state, dim=-1)
#             log_probs_base = torch.log_softmax(logits_without_state, dim=-1)

#             # Select log prob of the specific token
#             token_log_prob_state = log_probs_state
            
#             token_log_prob_base = log_probs_base
            
            
#             # Compute log-probability difference
#             log_prob_diff = (token_log_prob_state - token_log_prob_base).item()
            
#             # Append to lists
#             log_prob_diffs.append(log_prob_diff)
#             token_texts.append(tokenizer.decode(input_ids[0]))

#         return pd.DataFrame({"Token": token_texts, "Log_Prob_Diff": log_prob_diffs})

#         # Average KL divergence for the state
#         kl_divergences[state] = sum(kl_divergence_values) / len(kl_divergence_values)

#     return kl_divergences, kl_divergences_per_state

# # Example usage
# if __name__ == "__main__":
#     prompt = " I feel"
#     # states =[
#     #     'gourmet-food', 'video-game', 'clothing', 'beauty', 
#     #     'arts', 'book', 'jewelry', 'shoe', 
#     #     'musical-instrument', 'electronics'
#     #     ]
#     states = [
#                    'gourmet-food', 'video-game'
#         ]
#     checkpoint = "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_topic_22.pth"
#     model, tokenizer = load_model_and_tokenizer(checkpoint)
#     base_model = AutoModelForCausalLM.from_pretrained("gpt2").to('cuda')
#     device = torch.device(config['device'])
#     model = model.to(device)

#     print("Calculating perplexity ratios...")
#     ratios = generate_and_calculate_perplexity_ratio(model, tokenizer, prompt, states)
    
#     # Plot the perplexity ratios
#     plot_perplexity_ratios(ratios)
#     print("Plot saved as 'perplexity_ratio.png'")
#     kl_divergences, kl_divergences_per_state = evaluate_kl_divergence(model, tokenizer, prompt, states)
#     print("KL Divergence values per state:", kl_divergences)
#     import matplotlib.pyplot as plt


#     from scipy.stats import shapiro
#     import statsmodels.api as sm

#     # Run Shapiro-Wilk test for normality on each state's KL divergence distribution
#     for state, kl_vals in kl_divergences_per_state.items():
#         stat, p_value = shapiro(kl_vals)
#         print(f"Shapiro-Wilk Test for {state}: W={stat}, p-value={p_value}")
#         if p_value < 0.05:
#             print(f"{state} KL divergence is not normally distributed (reject H0)\n")
#         else:
#             print(f"{state} KL divergence appears normally distributed (fail to reject H0)\n")

#     # Q-Q Plot for one of the distributions
#     for state, kl_vals in kl_divergences_per_state.items():
#         sm.qqplot(np.array(kl_vals), line='45')
#         plt.title(f"Q-Q Plot for {state} KL Divergence")
#         plt.savefig(f"qqplot_{state}_kl_divergence.png")

#     from scipy.stats import levene

#     # Levene’s test for homogeneity of variances across states
#     kl_values_flat = [kl for values in kl_divergences_per_state.values() for kl in values]
#     state_labels = [state for state, values in kl_divergences_per_state.items() for _ in values]
#     levene_stat, levene_p_value = levene(*[kl_divergences_per_state[state] for state in states])

#     print(f"Levene’s test for homogeneity of variances: Statistic={levene_stat}, p-value={levene_p_value}")
#     if levene_p_value < 0.05:
#         print("Variance is not homogeneous (reject H0)")
#     else:
#         print("Variance is homogeneous (fail to reject H0)")


#     # Plot KL divergence values for each state
#     state_names = list(kl_divergences.keys())
#     values = list(kl_divergences.values())

#     plt.figure(figsize=(8, 5))
#     plt.bar(state_names, values, color='skyblue')
#     plt.xlabel('State')
#     plt.ylabel('KL Divergence')
#     plt.title('KL Divergence Across States')
#     plt.axhline(y=0, color='black', linewidth=0.5)
#     plt.savefig("KL_Divergence_Bar_Plot.png")

#     from scipy.stats import f_oneway

#     # Extract KL divergence values for each state
#     kl_divergences_values = [kl_divergences_per_state[state] for state in states]

#     # Perform one-way ANOVA
#     f_stat, p_value = f_oneway(*kl_divergences_values)

#     print(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")

#     # Interpretation:
#     if p_value < 0.05:
#         print("There is a statistically significant difference in KL divergence between states.")
#     else:
#         print("No statistically significant difference in KL divergence between states.")

#     from statsmodels.stats.multicomp import pairwise_tukeyhsd
#     import pandas as pd

#     # Prepare data in a long format for Tukey’s HSD
#     kl_data = []
#     for state, values in kl_divergences_per_state.items():
#         kl_data.extend([(state, val) for val in values])

#     # Convert to DataFrame
#     kl_df = pd.DataFrame(kl_data, columns=['State', 'KL Divergence'])

#     # Perform Tukey's HSD test
#     tukey_result = pairwise_tukeyhsd(kl_df['KL Divergence'], kl_df['State'], alpha=0.05)
#     print(tukey_result)

#     # Plot the pairwise comparisons
#     tukey_result.plot_simultaneous()
#     plt.xlabel('KL Divergence')
#     plt.title('Pairwise Tukey HSD Test for KL Divergence by State')
#     plt.savefig("KLD_topic")


#     # Plot violin plot for KL Divergence distributions by state
#     plt.figure(figsize=(10, 6))
#     sns.violinplot(x='State', y='KL Divergence', data=kl_df, inner='point', palette="muted")
#     plt.title("KL Divergence Distribution Across States")
#     plt.xlabel("State")
#     plt.ylabel("KL Divergence")
#     plt.savefig("KL_Divergence_Violin_Plot.png")

#     # Plot density plot for each state's KL divergence
#     plt.figure(figsize=(10, 6))

#     for state in kl_divergences_per_state:
#         sns.kdeplot(kl_divergences_per_state[state], label=state, fill=True, alpha=0.5)

#     plt.title("Density Plot of KL Divergence Across States")
#     plt.xlabel("KL Divergence")
#     plt.ylabel("Density")
#     plt.legend(title="State")
#     plt.savefig("KL_Divergence_Density_Plot.png")




# # def calculate_log_prob_changes(model, base_model, tokenizer, tokens, state_tensor):
# #     log_prob_diffs = []
# #     token_texts = []

# #     for token in tokens:
# #         # Encode the token
# #         input_ids = tokenizer.encode(token, return_tensors="pt").to(config['device'])

# #         # Get log probabilities with and without state tensor
# #         with torch.no_grad():
# #             logits_state = model(input_ids, state_tensor=state_tensor)[0]
# #             logits_base = model(input_ids, state_tensor=None)[0]
# #         # Calculate log softmax probabilities for each
# #         log_probs_state = torch.log_softmax(logits_state, dim=-1)
# #         log_probs_base = torch.log_softmax(logits_base, dim=-1)
# #         # Select log prob of the specific token
# #         token_log_prob_state = log_probs_state[0,:]
# #         token_log_prob_base = log_probs_base[0,:]

# #         # Compute log-probability difference
# #         log_prob_diff = (token_log_prob_state - token_log_prob_base).detach().cpu().numpy()
# #         # log_prob_diff = np.mean(log_prob_diff)  # Aggregate the differences

# #         # Append to lists
# #         log_prob_diffs.append(log_prob_diff[0])
# #         token_texts.append(tokenizer.decode(input_ids[0]))

# #     return pd.DataFrame({"Token": token_texts, "Log_Prob_Diff": log_prob_diffs})




# # import seaborn as sns
# # import scipy.stats as stats

# # def plot_log_prob_distributions(log_prob_diffs, state_name):
# #     # Plot the distribution
# #     plt.figure(figsize=(10, 6))
# #     breakpoint()
# #     sns.histplot(log_prob_diffs, kde=True, stat="density", color="skyblue", label="Log Prob Changes")
    
# #     # Overlay normal distribution for comparison
# #     mean, std = np.mean(log_prob_diffs), np.std(log_prob_diffs)
# #     x = np.linspace(min(log_prob_diffs), max(log_prob_diffs), 100)
# #     plt.plot(x, stats.norm.pdf(x, mean, std), color="red", label="Normal Distribution (mean, std)")

# #     plt.title(f"Distribution of Mean Log-Probability Changes - {state_name}")
# #     plt.xlabel("Log Probability Change")
# #     plt.ylabel("Density")
# #     plt.legend()
# #     plt.show()


# # # Example usage
# # if __name__ == "__main__":
# #     prompt = " I feel"
# #     # states =[
# #     #     'gourmet-food', 'video-game', 'clothing', 'beauty', 
# #     #     'arts', 'book', 'jewelry', 'shoe', 
# #     #     'musical-instrument', 'electronics'
# #     #     ]
# #     states = [
# #                    'love', 'sadness', 'joy', 'anger', 'fear'
# #         ]
# #     checkpoint = "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_emotions_wo_500_24.pth"
# #     model, tokenizer = load_model_and_tokenizer(checkpoint)
# #     base_model = AutoModelForCausalLM.from_pretrained("gpt2").to('cuda')
# #     device = torch.device(config['device'])
# #     model = model.to(device)

# #     # Run for each state and save results
# #     tokens_to_analyze = ["frustraed", "angry", "good", "great", "bitter", "great", "love", "sad", "guilty", "bad", "happy", "scared", "afraid", "vulnerable"]  # Define a list of tokens
# #     state_log_prob_diffs = {}
# #     for state in states:
# #         state_tensor = get_state_tensor(state)
# #         state_log_prob_diffs[state] = calculate_log_prob_changes(model, base_model, tokenizer, tokens_to_analyze, state_tensor)

# #     # Plot for each state
# #     # for state, log_diff_df in state_log_prob_diffs.items():
# #     #     plot_log_prob_distributions(log_diff_df["Log_Prob_Diff"], state)

# #     def top_changed_tokens(log_prob_df, top_n=10):
# #         # Sort by absolute log-probability difference and select top tokens
# #         log_prob_df["Abs_Log_Prob_Diff"] = log_prob_df["Log_Prob_Diff"].abs()
# #         top_tokens = log_prob_df.nlargest(top_n, "Abs_Log_Prob_Diff")
# #         return top_tokens[["Token", "Log_Prob_Diff", "Abs_Log_Prob_Diff"]]

# #     # Show top changed tokens for each state
# #     for state, log_diff_df in state_log_prob_diffs.items():
# #         print(f"\nTop changed tokens for state: {state}")
# #         print(top_changed_tokens(log_diff_df))
