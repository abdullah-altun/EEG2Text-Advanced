from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
import statistics
import bert_score
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score


def compute_bleu(reference, candidate):
    reference = [[ref.split()] for ref in reference]
    candidate = [cand.split() for cand in candidate]
    smoothing_function = SmoothingFunction().method4
    bleu_scores = [
        corpus_bleu([ref], [cand], smoothing_function=smoothing_function)
        for ref, cand in zip(reference, candidate)
    ]
    mean_bleu_score = np.mean(bleu_scores)
    bleu_std_dev = np.std(bleu_scores)
    return round(mean_bleu_score, 3), round(bleu_std_dev, 3)

def compute_bleu_unigram(reference, candidate):
    reference = [[ref.split()] for ref in reference]
    candidate = [cand.split() for cand in candidate]
    smoothing_function = SmoothingFunction().method4
    weights = (1, 0, 0, 0)
    bleu_scores = [
        corpus_bleu([ref], [cand], 
                    smoothing_function=smoothing_function, 
                    weights=weights)
        for ref, cand in zip(reference, candidate)
    ]
    mean_bleu_score = round(np.mean(bleu_scores), 3)
    bleu_std_dev = round(np.std(bleu_scores), 3)
    return mean_bleu_score, bleu_std_dev

def compute_rouge(reference, candidate):
    rouge = Rouge()
    rouge_scores = [
        rouge.get_scores(cand, ref, avg=True) 
        for ref, cand in zip(reference, candidate)
    ]
    rouge_1_f_scores = [score['rouge-1']['f'] for score in rouge_scores]
    rouge_2_f_scores = [score['rouge-2']['f'] for score in rouge_scores]
    rouge_l_f_scores = [score['rouge-l']['f'] for score in rouge_scores]

    mean_rouge_1_f_score = np.mean(rouge_1_f_scores)
    mean_rouge_2_f_score = np.mean(rouge_2_f_scores)
    mean_rouge_l_f_score = np.mean(rouge_l_f_scores)

    rouge_1_f_std_dev = np.std(rouge_1_f_scores)
    rouge_2_f_std_dev = np.std(rouge_2_f_scores)
    rouge_l_f_std_dev = np.std(rouge_l_f_scores)

    return (
        round(mean_rouge_1_f_score, 3),
        round(mean_rouge_2_f_score, 3),
        round(mean_rouge_l_f_score, 3),
        round(rouge_1_f_std_dev, 3),
        round(rouge_2_f_std_dev, 3),
        round(rouge_l_f_std_dev, 3)
    )


def compute_bert_score(reference, candidate):
    bert_p_scores, bert_r_scores, bert_f1_scores = bert_score.score(
        candidate, reference, lang="en", verbose=False
    )
    return round(bert_f1_scores.mean().item(), 3), round(bert_f1_scores.std().item(), 3)


def compute_meteor_scores(reference, candidate):
    tokenized_candidates = [
        word_tokenize(candidate.replace("<s>", "").replace("</s>", "").strip()) 
        for candidate in candidate
    ]
    tokenized_references = [
        word_tokenize(sentence) 
        for sentence in reference
    ]
    meteor_scores = []
    for ref_sentence, candidate in zip(tokenized_references, tokenized_candidates):
        meteor_scores.append(single_meteor_score(ref_sentence, candidate))
    meteor_scores_mean = sum(meteor_scores) / len(meteor_scores)
    meteor_scores_std = statistics.stdev(meteor_scores)
    return round(meteor_scores_mean, 3), round(meteor_scores_std, 3)
