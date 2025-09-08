import argparse
import os
from pathlib import Path
import json

from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)

from multimodal_coding.utils.utils import update_json_file


FILE = Path(__file__).resolve()
MME_ROOT = FILE.parents[0]

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='./LaVIN', type=str)
parser.add_argument('--json-result-path', type=str, default=str(MME_ROOT / 'results.json'))

eval_type_dict = {
    'Perception': ['existence', 'count', 'position', 'color', 'posters', 'celebrity', 'scene', 'landmark', 'artwork', 'OCR'],
    'Cognition': ['commonsense_reasoning', 'numerical_calculation', 'text_translation', 'code_reasoning']
}


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

        return

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ['yes', 'no']:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if 'yes' in prefix_pred_ans:
                pred_label = 'yes'
            elif 'no' in prefix_pred_ans:
                pred_label = 'no'
            else:
                pred_label = 'other'

        return pred_label

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            'yes': 1,
            'no': 0,
            'other': -1,
        }

        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds)

        clean_gts = []
        clean_preds = []
        other_num = 0
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            'TP': tp,
            'FN': fn,
            'TN': tn,
            'FP': fp,
            'precision': precision,
            'recall': recall,
            'other_num': other_num,
            'acc': acc,
        }

        return metric_dict

    def process_result(self, results_dir, json_file):
        model_id = os.path.basename(results_dir)
        self.data = dict()
        with open(json_file, "r") as f:
            data = json.load(f)
        self.data = data.get(model_id)

        model_score_dict = dict()
        for eval_type, task_name_list in eval_type_dict.items():
            print('===========', eval_type, '===========')

            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:

                task_txt = os.path.join(results_dir, task_name + '.txt')
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions

                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    img_correct_num = 0

                    for img_item in img_items:
                        try:
                            img_name, question, gt_ans, pred_ans = img_item.split('\t')
                        except:
                            print(img_item)
                            continue
                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ['yes', 'no'] # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ['yes', 'no', 'other']

                        gts.append(gt_ans)
                        preds.append(pred_ans)

                        if gt_ans == pred_ans:
                            img_correct_num += 1

                        if pred_ans not in ['yes', 'no']:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict['acc_plus'] = acc_plus

                for k, v in metric_dict.items():
                    if k in ['acc', 'acc_plus']:
                        task_score += v*100

                task_score_dict[task_name] = task_score

                scores += task_score

            task_score_dict[eval_type] = scores
            self.save_results(task_score_dict)
            print('total score:', scores, '\n')
            for task_name, score in task_score_dict.items():
                print('\t', task_name, ' score:', score)
            print('\n')
        self.save_results_json(json_file, model_id)
        return

    def save_results(self, task_score_dict):
        orig_bpp = 0.0
        compressed_bpp = 0.0
        count = 0
        for k, v in task_score_dict.items():
            if k in self.data:
                orig_bpp += self.data[k]['orig_bpp'] * self.data[k]['count']
                compressed_bpp += self.data[k]['compressed_bpp'] * self.data[k]['count']
                count += self.data[k]['count']
                self.data[k]['score'] = v
            else:
                self.data[k] = {'orig_bpp': orig_bpp/count, 'compressed_bpp': compressed_bpp/count, 'count': count, 'score': v}

    def save_results_json(self, json_file, model_id):
        self.data['Total'] = {}
        self.data['Total']['orig_bpp'] = (self.data['Perception']['orig_bpp'] * self.data['Perception']['count'] + \
                                                    self.data['Cognition']['orig_bpp'] * self.data['Cognition']['count']) / \
                                                    (self.data['Perception']['count'] + self.data['Cognition']['count'])
        self.data['Total']['compressed_bpp'] = (self.data['Perception']['compressed_bpp'] * self.data['Perception']['count'] + \
                                                        self.data['Cognition']['compressed_bpp'] * self.data['Cognition']['count']) / \
                                                        (self.data['Perception']['count'] + self.data['Cognition']['count'])
        self.data['Total']['count'] = self.data['Perception']['count'] + self.data['Cognition']['count']
        self.data['Total']['score'] = self.data['Perception']['score'] + self.data['Cognition']['score']

        data = {model_id: self.data}

        update_json_file(json_file, data)
        return


if __name__ == '__main__':
    cal = calculate_metrics()

    args = parser.parse_args()
    results_dir = args.results_dir
    json_file = args.json_result_path
    cal.process_result(results_dir, json_file)
