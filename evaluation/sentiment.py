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

NUM_STATES=2
STATE_MAPPING='sentiment_mapping'


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

# Initialize the classifier
def get_distilroberta_classifier():
    """Get the emotion-english-distilroberta-base classifier."""
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=False)
    return classifier

classifier = get_distilroberta_classifier()

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
def evaluate_generations_with_auc(model, tokenizer, states, num_generations=100, num_runs=5, steered=False):
    # Dictionary to store binary labels and scores per state for AUC calculation
    unknown_class_idx = len(states)
    states.append("Other")
    all_aucs = {state: [] for state in states}
    all_precisions = {state: [] for state in states}
    all_recalls = {state: [] for state in states}
    all_f1s = {state: [] for state in states}
    for run in range(num_runs):
        print(f"Run {run + 1} of {num_runs}")
        all_true_labels = []
        all_predicted_scores = {state: [] for state in states}
        y_pred = []
        y_true = []
        for state in states[:-1]:
            # Create state tensor
            state_tensor = get_state_tensor(state, device) if steered else None
            
            generated_texts = generate_text(model=model, tokenizer=tokenizer, prompt="I feel", state_tensor=state_tensor, num_generations=num_generations)
            # Get the classification results for each generated text
            classifications = classifier(generated_texts)

            for classification in classifications:
                # Store the true label as binary array for multi-class ROC (one-hot encoding)
                true_label = [1 if state == target_state else 0 for target_state in states[:-1]]
                all_true_labels.append(true_label)

                # Store the predicted scores for each state
                for target_state in states[:-1]:
                    score = 0
                    score = classification['score'] if classification['label'].lower() == target_state else 0
                    all_predicted_scores[target_state].append(score)
                y_true.append(states.index(state))

                # Predicted label: Find the state with the highest classification score
                predicted_label = classification['label'].lower()

                # If the predicted label is not in states, treat it as "unknown"
                if predicted_label in states:
                    predicted_label_idx = states.index(predicted_label)
                else:
                    predicted_label_idx = unknown_class_idx  # Assign to "unknown"
                y_pred.append(predicted_label_idx)

        # Plot ROC curves for each state
        fig, ax = plt.subplots(figsize=(10, 8))
        all_true_labels = np.array(all_true_labels)

        for idx, state in enumerate(states[:-1]):
            # Compute ROC curve and AUC for each state in a one-vs-rest fashion
            fpr, tpr, _ = roc_curve(all_true_labels[:, states.index(state)], all_predicted_scores[state])
            auc = roc_auc_score(all_true_labels[:, states.index(state)], all_predicted_scores[state])
            all_aucs[state].append(auc)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, label=f"{state.capitalize()} (AUC = {auc:.2f})")
            true_state_labels = [1 if label == idx else 0 for label in y_true]
            predicted_state_labels = [1 if label == idx else 0 for label in y_pred]
            
            # Calculate Precision, Recall, and F1
            precision = precision_score(true_state_labels, predicted_state_labels)
            recall = recall_score(true_state_labels, predicted_state_labels)
            f1 = f1_score(true_state_labels, predicted_state_labels)

            all_precisions[state].append(precision)
            all_recalls[state].append(recall)
            all_f1s[state].append(f1)            

    # Plot settings
    # Calculate the variability for each metric (mean and standard deviation)
    for state in states[:-1]:
        auc_mean = np.mean(all_aucs[state])
        auc_std = np.std(all_aucs[state])
        
        precision_mean = np.mean(all_precisions[state])
        precision_std = np.std(all_precisions[state])
        
        recall_mean = np.mean(all_recalls[state])
        recall_std = np.std(all_recalls[state])
        
        f1_mean = np.mean(all_f1s[state])
        f1_std = np.std(all_f1s[state])

        print(f"\nState: {state.capitalize()}")
        print(f"  AUC: {auc_mean:.2f} ± {auc_std:.2f}")
        print(f"  Precision: {precision_mean:.2f} ± {precision_std:.2f}")
        print(f"  Recall: {recall_mean:.2f} ± {recall_std:.2f}")
        print(f"  F1-score: {f1_mean:.2f} ± {f1_std:.2f}")
        
    # Plot settings
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("ROC Curves with AUC for Positive and Negative Sentiment")
    ax.legend(loc="lower right")
    mode = "steered" if steered else "unsteered"
    plt.savefig(f'resources/evaluation_results/sentiment/AUC_senti_{mode}_evaluation_new.png')

# def evaluate_unsteered_generations_with_auc(model, tokenizer, states, num_generations=100, num_runs=5):
#     # Initialize lists to store metrics for each run
#     all_aucs = {state: [] for state in states}
#     all_precisions = {state: [] for state in states}
#     all_recalls = {state: [] for state in states}
#     all_f1s = {state: [] for state in states}
#     for run in range(num_runs):
#         print(f"Run {run + 1} of {num_runs}")
#         # Dictionary to store binary labels and scores per state for AUC calculation
#         all_true_labels = []
#         all_predicted_scores = {state: [] for state in states}
#         y_pred = []
#         y_true = []
#         unknown_class_idx = len(states)
#         states.append("Other")
#         for state in states[:-1]:
#             # Create state tensor
#             state_tensor = get_state_tensor(state, device)
            
#             generated_texts = generate_text(model=model, tokenizer=tokenizer, prompt="I feel", state_tensor=None, num_generations=num_generations)
#             # Get the classification results for each generated text
#             classifications = classifier(generated_texts)

#             for classification in classifications:
#                 # Store the true label as binary array for multi-class ROC (one-hot encoding)
#                 true_label = [1 if state == target_state else 0 for target_state in states[:-1]]
#                 all_true_labels.append(true_label)

#                 # Store the predicted scores for each state
#                 for target_state in states[:-1]:
#                     score = 0
#                     score = classification['score'] if classification['label'].lower() == target_state else 0
#                     all_predicted_scores[target_state].append(score)
#                 y_true.append(states.index(state))

#                 # Predicted label: Find the state with the highest classification score
#                 predicted_label = classification['label'].lower()

#                 # If the predicted label is not in states, treat it as "unknown"
#                 if predicted_label in states:
#                     predicted_label_idx = states.index(predicted_label)
#                 else:
#                     predicted_label_idx = unknown_class_idx  # Assign to "unknown"
#                 y_pred.append(predicted_label_idx)

#         # Plot ROC curves for each state
#         # fig, ax = plt.subplots(figsize=(10, 8))
#         # all_true_labels = np.array(all_true_labels)

#         for idx, state in enumerate(states[:-1]):
#             # Compute ROC curve and AUC for each state in a one-vs-rest fashion
#             fpr, tpr, _ = roc_curve(all_true_labels[:, states.index(state)], all_predicted_scores[state])
#             auc = roc_auc_score(all_true_labels[:, states.index(state)], all_predicted_scores[state])
            
#             # Plot ROC curve
#             ax.plot(fpr, tpr, label=f"{state.capitalize()} (AUC = {auc:.2f})")
#             true_state_labels = [1 if label == idx else 0 for label in y_true]
#             predicted_state_labels = [1 if label == idx else 0 for label in y_pred]
            
#             # Calculate Precision, Recall, and F1
#             precision = precision_score(true_state_labels, predicted_state_labels)
#             recall = recall_score(true_state_labels, predicted_state_labels)
#             f1 = f1_score(true_state_labels, predicted_state_labels)
            
#             all_precisions[state].append(precision)
#             all_recalls[state].append(recall)
#             all_f1s[state].append(f1)

#     # Calculate the variability for each metric (mean and standard deviation)
#     for state in states:
#         auc_mean = np.mean(all_aucs[state])
#         auc_std = np.std(all_aucs[state])
        
#         precision_mean = np.mean(all_precisions[state])
#         precision_std = np.std(all_precisions[state])
        
#         recall_mean = np.mean(all_recalls[state])
#         recall_std = np.std(all_recalls[state])
        
#         f1_mean = np.mean(all_f1s[state])
#         f1_std = np.std(all_f1s[state])

#         print(f"\nState: {state.capitalize()}")
#         print(f"  AUC: {auc_mean:.2f} ± {auc_std:.2f}")
#         print(f"  Precision: {precision_mean:.2f} ± {precision_std:.2f}")
#         print(f"  Recall: {recall_mean:.2f} ± {recall_std:.2f}")
#         print(f"  F1-score: {f1_mean:.2f} ± {f1_std:.2f}")
#     # Plot settings
#     ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     # ax.set_title("ROC Curves with AUC for Positive and Negative Sentiment")
#     ax.legend(loc="lower right")
#     plt.savefig('resources/evaluation_results/sentiment/AUC_senti_unsteered_26.11.png')
    
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == "__main__":
    print("Loading checkpoint...\n")
    checkpoint = "/home/qequ54zi/thesis/submission/SLiM/resources/checkpoints/SLiM_sentiment_wo_500_49.pth"
    model, tokenizer = load_model_and_tokenizer(checkpoint)
    device = torch.device(config['device'])
    model = model.to(device)
    # Define state list and device
    states1 = [ 'positive', 'negative']
    # Run evaluation
    evaluate_generations_with_auc(model, tokenizer, states1, steered=True)
    evaluate_generations_with_auc(model, tokenizer, states1, steered=False)
    # evaluate_unsteered_generations_with_auc(model, tokenizer, states)

