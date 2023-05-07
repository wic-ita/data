import json
import zipfile
import os
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import spearmanr
import numpy as np
import logging
import argparse


def read_submission(path):
    examples = {}
    archive = zipfile.ZipFile(path, 'r')
    for fname in archive.namelist():
        if not fname.startswith('description'):
            f = archive.open(fname)
            examples[fname] = {}
            for line in f:
                example = json.loads(line)
                if fname.startswith('ranking'):
                    examples[fname][example['id']] = float(example['score'])
                else:
                    examples[fname][example['id']] = int(example['label'])
    return examples

def read_gt(bin_path='binary',rank_path='ranking'):
    gt_bin = {}
    gt_rank = {}

    for fname in ['test.jsonl','test_eng.jsonl']:
        gt_bin[fname] = {}
        with open(os.path.join(bin_path,fname),'r') as f:
            for line in f:
                example = json.loads(line)
                gt_bin[fname][example['id']] = int(example['label'])

    for fname in ['test.jsonl','test_eng.jsonl']:
        gt_rank[fname] = {}
        with open(os.path.join(rank_path, fname),'r') as f:
            for line in f:
                example = json.loads(line)
                gt_rank[fname][example['id']] = float(example['score'])
    return gt_bin, gt_rank

def get_prediction(examples,keys,task):
    predictions = []
    try:
        for k in keys:
            predictions.append(examples[task][k])
    except Exception as e:
        raise Exception(f'Cannot compute metrics for task {task}, error in prediction keys. {e}')
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='WiC-ITA Evaluation',
        description="Evaluation script for WiC-ITA task")

    parser.add_argument('--gt_bin_path', default='binary')
    parser.add_argument('--gt_rank_path', default='ranking')
    parser.add_argument('--submission', required=True)
    args = parser.parse_args()

    binarytopred = {'test.jsonl': 'binary.jsonl', 'test_eng.jsonl': 'binary_eng.jsonl'}
    rankingtopred = {'test.jsonl': 'ranking.jsonl', 'test_eng.jsonl': 'ranking_eng.jsonl'}

    submission_file = args.submission
    submission_name = submission_file.split('.')[0]
    examples = read_submission(submission_file)
    gt_bin, gt_rank = read_gt(args.gt_bin_path,args.gt_rank_path)
    results = {}
    for task in gt_bin:
        keys = []
        labels = []
        for k in gt_bin[task]:
            labels.append(gt_bin[task][k])
            keys.append(k)
        task_name = binarytopred[task]
        if task_name in examples:
            try:
                predictions = get_prediction(examples,keys,task_name)
                precision,recall,f1,_ = precision_recall_fscore_support(labels, predictions, labels=[0,1])
                results[task_name] = {
                    'class 0': {
                        'precision': precision[0],
                        'recall': recall[0],
                        'f1': f1[0]
                    },
                    'class 1': {
                        'precision': precision[1],
                        'recall': recall[1],
                        'f1': f1[1]
                    },
                    'weighted': {
                        'precision': np.mean(precision),
                        'recall': np.mean(recall),
                        'f1': np.mean(f1)
                    }
                }
            except Exception as e:
                logging.warning(e)
                pass
        else:
            logging.warning(f'Missing predictions for task {task}')

    for task in gt_rank:
        keys = []
        scores = []
        for k in gt_rank[task]:
            scores.append(gt_rank[task][k])
            keys.append(k)
        task_name = rankingtopred[task]
        if task_name in examples:
            try:
                predictions = get_prediction(examples,keys,task_name)
                spearman, pvalue = spearmanr(scores, predictions)
                results[task_name] = {
                    'spearman': spearman,
                    'pvalue': pvalue
                }
            except Exception as e:
                logging.warning(e)
                pass
        else:
            logging.warning(f'Missing predictions for task {task}')

    with open(f'results_{submission_name}.json','w+') as f:
        json.dump(results,f,indent=4)