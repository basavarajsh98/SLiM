from transformers import pipeline
import torch
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
from src.model import SLiMedNet
from config.config import get_config
from torch.amp import autocast
from transformers import AutoModelForCausalLM
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
config = get_config()
# import nltk
# from nltk import pos_tag, word_tokenize
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
from collections import Counter

NUM_STATES=10
STATE_MAPPING='topic_mapping'


# def extract_adjectives(text):
#     words = word_tokenize(text)
#     tagged_words = pos_tag(words)
#     adjectives = [word for word, tag in tagged_words if tag in ["JJ", "JJR", "JJS"]]
#     return adjectives

# def generate_emotion_wordcloud(texts, state):
#     # Extract adjectives from each text
#     adjectives = []
#     for text in texts:
#         adjectives.extend(extract_adjectives(text))
    
#     # Count frequency of each adjective
#     adjective_counts = Counter(adjectives)
    
#     # Generate word cloud
#     wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(adjective_counts)
    
#     # Display the word cloud
#     # plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     plt.savefig(f"word_cloud_{state}")

#     top_adjectives = adjective_counts.most_common(3)
#     print(f"\nTop 10 adjectives for {state.capitalize()}:")
#     for word, freq in top_adjectives:
#         print(f"{word}: {freq}")

def load_model_and_tokenizer(path):
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    model = SLiMedNet(state_embed_dim=NUM_STATES, model=model)
    checkpoint = torch.load(path, map_location=torch.device(config['device']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_text(model, tokenizer, prompt, state_tensor=None, num_generations=100):
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(config['device'])
    input_ids = inputs['input_ids']
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            state_tensor=state_tensor, 
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_length=config['max_length'], 
            max_new_tokens = 100,
            num_return_sequences=num_generations,  
            no_repeat_ngram_size=2,  
            top_k=50,
            top_p=0.9,
            temperature=1,
            do_sample=True
        )
    return [tokenizer.decode(g, skip_special_tokens=True) for g in output]

def get_state_tensor(state, device):
    state_mapping = config[STATE_MAPPING]
    if state in state_mapping:
        state_vector = state_mapping[state]
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
        return state_tensor
    else:
        print("not a state")
        return None

# Compute similarity between generated text and average topic embedding
def compute_similarity(generated_sentence, avg_topic_embedding):
    generated_sentence_embeddings = model_rel.encode(generated_sentence)
    return cosine_similarity(generated_sentence_embeddings.reshape(1, -1), avg_topic_embedding)[0][0]

# # Load GPT-2 model and tokenizer
# fluent_model_name = "NousResearch/Llama-2-7b-hf"
# fluent_tokenizer = AutoTokenizer.from_pretrained(fluent_model_name)
# fluent_model = AutoModelForCausalLM.from_pretrained(fluent_model_name)
# fluent_model.eval()

# def get_logprobs(text):
#     # Tokenize input text and get input IDs
#     input_ids = fluent_tokenizer.encode(text, return_tensors="pt")
    
#     # Get model outputs
#     with torch.no_grad():
#         outputs = fluent_model(input_ids, labels=input_ids)
    
#     # Calculate log probabilities
#     log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
#     logprobs = []
    
#     for i in range(1, input_ids.size(1)):  # Skip the first token
#         token_id = input_ids[0, i]
#         log_prob = log_probs[0, i-1, token_id].item()
#         logprobs.append(log_prob)
    
#     return logprobs

# def fluency(prompt, generated_text):
#     # Get logprobs for prompt and full generated text
#     prompt_logprobs = get_logprobs(prompt)
#     full_logprobs = get_logprobs(generated_text)

#     # Get continuation logprobs by excluding prompt tokens
#     continuation_logprobs = full_logprobs[len(prompt_logprobs):]

#     # Calculate fluency score
#     fluency_score = np.exp(-np.mean(continuation_logprobs))
#     return fluency_score

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
# Define evaluation function
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

model_rel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to('cuda')

# Function to get average topic embedding
def get_average_topic_embedding():
    import os
    if os.path.exists("/home/qequ54zi/thesis/submission/SLiM/average.embedding_5k.pth"):
        return torch.load("/home/qequ54zi/thesis/submission/SLiM/average.embedding_5k.pth")
    else:
        topics = [
            'gourmet-food', 'video-game', 'clothing', 'beauty', 
            'arts', 'book', 'jewelry', 'shoe', 
            'musical-instrument', 'electronics'
        ]
        datasets = {
            topic: load_dataset('contemmcm/amazon_reviews_2013', topic, split="complete") for topic in topics
        }
        from random import shuffle

        avg_topic_embedding = {topic: [] for topic in topics}   
        for i, topic in enumerate(datasets):
            shuffle(datasets[topic]['review/text'])
            sampled_data = datasets[topic]['review/text'][:5000]
            print(f"Topic Size: {topic}: ", len(sampled_data))
            embeddings = np.array([model_rel.encode(sentence) for sentence in sampled_data])
            avg_topic_embedding[topic].append(np.mean(embeddings, axis=0))
        torch.save(avg_topic_embedding, "average.embedding.pth")
        return avg_topic_embedding

# Evaluation function with similarity calculations
def evaluate_generations_with_similarity(model, tokenizer, states, topic_embeddings, num_generations=100):
    similarity_scores_with_state = {state: [] for state in states}
    similarity_scores_without_state = {state: [] for state in states}

    for state in states:
        state_tensor = get_state_tensor(state, device)
        # Generate texts with state tensor
        generated_texts_with_state = generate_text(model=model, tokenizer=tokenizer, prompt="I feel", state_tensor=state_tensor, num_generations=num_generations)
        for gen_text in generated_texts_with_state:
            similarity_score = compute_similarity(gen_text, topic_embeddings[state])
            similarity_scores_with_state[state].append(similarity_score)

        # Generate texts without state tensor
        generated_texts_without_state = generate_text(model=model, tokenizer=tokenizer, prompt="I feel", state_tensor=None, num_generations=num_generations)
        for gen_text in generated_texts_without_state:
            similarity_score = compute_similarity(gen_text, topic_embeddings[state])
            similarity_scores_without_state[state].append(similarity_score)
        print(f"Completed evaluating {state}")

    # Create DataFrames for plotting
    similarity_df_with_state = pd.DataFrame([
        {"State": state, "Similarity Score": score, "Condition": "With State"} 
        for state, scores in similarity_scores_with_state.items() for score in scores
    ])

    similarity_df_without_state = pd.DataFrame([
        {"State": state, "Similarity Score": score, "Condition": "Without State"} 
        for state, scores in similarity_scores_without_state.items() for score in scores
    ])

    # Combine the two DataFrames
    combined_similarity_df = pd.concat([similarity_df_with_state, similarity_df_without_state])

    # Calculate mean and standard deviation for each condition
    mean_scores = combined_similarity_df.groupby(['State', 'Condition']).agg(['mean', 'std']).reset_index()

    # Plot using a bar plot with error bars
    plt.figure(figsize=(14, 8))
    sns.barplot(data=mean_scores, x="State", y=("Similarity Score", "mean"), hue="Condition", 
                palette="Set2", ci=None)

    # Add error bars
    for i in range(len(mean_scores)):
        plt.errorbar(x=i // 2, y=mean_scores[("Similarity Score", "mean")][i], 
                     yerr=mean_scores[("Similarity Score", "std")][i], fmt='none', 
                     c='black', capsize=5)

    # Display values on each bar
    for i in range(len(mean_scores)):
        plt.text(x=(i // 2) + 0.03, y=mean_scores[("Similarity Score", "mean")][i] + 0.01, 
                s=f"{mean_scores[('Similarity Score', 'mean')][i]:.2f}", 
                ha='center', va='bottom', fontsize=10)

    plt.title("Average Relevance per Topic: With vs Without Steering")
    plt.ylabel("Mean Relevance Score")
    plt.xlabel("Topic")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.savefig("resources/evaluation_results/topic/topic_plot_26.11.png")

    # Calculate mean similarity scores for each state
    # mean_similarity_with_state = {state: np.mean(scores) for state, scores in similarity_scores_with_state.items()}
    # mean_similarity_without_state = {state: np.mean(scores) for state, scores in similarity_scores_without_state.items()}

    # # Calculate relevance increase
    # relevance_increase = {
    #     state: 100 * (mean_similarity_with_state[state] - mean_similarity_without_state[state]) / mean_similarity_without_state[state]
    #     for state in states
    # }

    # # Convert relevance increase to DataFrame for plotting
    # relevance_increase_df = pd.DataFrame({
    #     "State": list(relevance_increase.keys()),
    #     "Relevance Increase": list(relevance_increase.values())
    # })

    # # Plot the relevance increase
    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=relevance_increase_df, x="State", y="Relevance Increase", palette="coolwarm")
    # plt.title("Percentage Increase in Relevance Score with State Tensor")
    # plt.ylabel("Relevance Increase (%)")
    # plt.xlabel("State")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig("resources/evaluation_results/topic/relevance_increase_plot.png")

from datasets import load_dataset

if __name__ == "__main__":
    print("Loading checkpoint...\n")
    checkpoint = "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_topic_wo_500_24.pth"
    model, tokenizer = load_model_and_tokenizer(checkpoint)
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Define state list and device
    # Define topics and load datasets

    states = [
                   'gourmet-food', 'video-game', 'clothing', 'beauty', 
        'arts', 'book', 'jewelry', 'shoe', 
        'musical-instrument', 'electronics'
        ]
  
    # Load the average topic embedding
    avg_topic_embedding = get_average_topic_embedding()
    
    # Run evaluation
    evaluate_generations_with_similarity(model, tokenizer, states, avg_topic_embedding)


