import os

import tqdm
import nltk
import pandas as pd
import torch

from transformers import BertTokenizer, BertForMaskedLM
import re

from compute_metrics import *

nltk.download('punkt')
nltk.download('wordnet')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def read_result_csv(file_path):
    result_df = pd.read_csv(file_path)
    result_df.drop(result_df.columns[0], axis=1, inplace=True)
    return result_df

def clean_text(text):
    """
        Belirli regex ile istenmeyen karakterler çıkartılıyor
    """
    regex = r"[^a-zA-Z0-9.,!?;:'\"()\[\]{}\-\s]"
    cleaned_text = re.sub(regex, "", text)
    lines = re.split(r'(?<=[.!?]) +', cleaned_text)

    seen = set()
    unique_lines = []
    for line in lines:
        cleaned_line = re.sub(r'\s+', " ", line).strip()
        if cleaned_line not in seen:
            seen.add(cleaned_line)
            unique_lines.append(cleaned_line)
    
    unique_lines = unique_lines[:1]
    result = " ".join(unique_lines)
    return result

def cleanup_pred_captions(predicted_cations):
    predicted_cations = predicted_cations.fillna('')
    clean_captions = []

    for caption in predicted_cations:
        clean_caption = f"No response."
        if caption.strip():
            clean_caption = clean_text(caption)
            if not clean_caption.strip():
                clean_caption = f"No response."
        clean_captions.append(clean_caption)

    return clean_captions

def run(csv_path):
    results = {}
    result_df = read_result_csv(csv_path)

    image_paths = result_df["Groud Truth Image"] # Gerçek Resim
    expected_captions = result_df['Expected Caption'] # Beklenen etiket
    predicted_captions = result_df['Generated Caption'] # Üretilen etiket
    expected_object_classes = result_df['Expected object'] # Beklenen nesne
    predicted_object_classes = result_df['Predicted object'] # Üretilen nesne

    predicted_captions = cleanup_pred_captions(predicted_captions)
    references = expected_captions.tolist()
    candidates = predicted_captions

    for i, cand in enumerate(candidates):
        if len(cand) <= 1:
            candidates[i] = "No response"

    mean_bleu_score, bleu_std_dev = compute_bleu(references, candidates)
    results["Mean BLEU Score"] =  mean_bleu_score
    results["SD BLEU Score"] =  bleu_std_dev

    mean_bleu_score, bleu_std_dev = compute_bleu_unigram(references, candidates)
    results["Mean BLEU Unigram Score"] =  mean_bleu_score
    results["SD BLEU Unigram Score"] =  bleu_std_dev

    (
        mean_rouge_1_f_score, 
        mean_rouge_2_f_score, 
        mean_rouge_l_f_score,
        rouge_1_f_std_dev, 
        rouge_2_f_std_dev, 
        rouge_l_f_std_dev
    ) = compute_rouge(references, candidates)

    results["Mean ROUGE-1"] = mean_rouge_1_f_score
    results["SD ROUGE-1"] = rouge_1_f_std_dev
    results["Mean ROUGE-2"] = mean_rouge_2_f_score
    results["SD ROUGE-2"] = rouge_2_f_std_dev
    results["Mean ROUGE-l"] = mean_rouge_l_f_score
    results["SD ROUGE-l"] = rouge_l_f_std_dev

    # METEOR
    mean_meteor_score, meteor_std_dev = compute_meteor_scores(references, candidates)
    results["Mean Meteor Score"] =  mean_meteor_score
    results["SD Meteor Score"] = meteor_std_dev

    # BERTScore
    bert_score_mean, bert_score_std_dev = compute_bert_score(references, candidates)
    results["Mean BERTScore"] = round(bert_score_mean,3)
    results["SD BERTScore"] = round(bert_score_std_dev, 3)

    return results

results_dir = "results"
all_res = {}
for file in tqdm.tqdm(os.listdir(results_dir)):
    print(file)
    fullpath = os.path.join(results_dir,file)
    results = run(csv_path=fullpath)
    all_res[file.replace("csv","")] = results

results_df = pd.DataFrame(all_res).transpose()
results_df.to_csv("all_results.csv")