import copy
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import sklearn.metrics
from transformers import AutoTokenizer
import re
import torch
import sklearn
import torch.nn.functional as F
import os
import time
from .interfere import interfere, get_vector, orthogonal
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import LedoitWolf
from pytorch_ood.detector import Mahalanobis

def to_numpy_float32(x):
    """Safely convert torch tensor / numpy array / list to float32 numpy array."""

    # Case 1 — Torch Tensor
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()

    # Case 2 — Numpy array
    if isinstance(x, np.ndarray):
        # ensure float32
        if x.dtype == np.float32:
            return x
        return x.astype(np.float32)

    # Case 3 — Python list or other iterable
    return np.asarray(x, dtype=np.float32)

def cosine_similarity_batch(matrix1, matrix2):    
    dot_products = np.dot(matrix1, matrix2.T)
    norm1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
    return dot_products / (norm1 * norm2.T + 1e-10)

def compute_mahalanobis_distances(class_hidden_states, prototype_vectors):
    cov_matrix = np.cov(class_hidden_states, rowvar=False)
    cov_matrix_inv = np.linalg.inv(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]))
    class_hidden_states_centered = class_hidden_states[:, np.newaxis, :] - prototype_vectors[np.newaxis, :, :]
    distances = np.einsum(
        'ijk,kl,ijl->ij',
        class_hidden_states_centered,
        cov_matrix_inv,
        class_hidden_states_centered
    )
    return np.sqrt(distances)

class Metrics():
    def __init__(self, data_args, model_args, training_args):
        id_list = pd.read_csv(f'data/{data_args.dataset_name}/{data_args.dataset_name}_{data_args.rate}.txt', header=None)[0].tolist()
        self.target_names = pd.read_csv(f'data/{data_args.dataset_name}/origin_data/label.list', header=None)[0].tolist()
        self.id_list = [i + 1 for i in range(len(id_list))]
        self.ood_list = [0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        start_index = 0
        if self.tokenizer.name_or_path.split('/')[-1] in ['Meta-Llama-3.1-8B-Instruct']:
            self.class_token = torch.tensor([self.tokenizer.encode(f" {i:03d}")[1:]  for i in list(range(start_index, len(self.target_names) + 1))])
        elif self.tokenizer.name_or_path.split('/')[-1] in ['Llama-3.2-8B-Instruct']:
            self.class_token = torch.tensor([self.tokenizer.encode(f"{i:03d}")[1:]  for i in list(range(start_index, len(self.target_names) + 1))])
        elif self.tokenizer.name_or_path.split('/')[-1] in ['Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct']:
            self.class_token = torch.tensor([self.tokenizer.encode(f" {i:03d}")[1:]  for i in list(range(start_index, len(self.target_names) + 1))])
        self.ood_token = torch.tensor([self.tokenizer.encode(f"{i}")[-1] for i in [' No', ' Yes']])
        self.output_dir = training_args.output_dir
        self.mode = model_args.mode
        self.segment_length = data_args.segment_length

    def preprocess_predictions(self, class_preds, class_golds, sequence_scores):
        ood_segment_preds = class_preds.reshape(-1,self.segment_length)
        sequence_scores_max = sequence_scores.max(axis=-1)
        sequence_segment_scores = sequence_scores_max.reshape(-1,self.segment_length)

        class_golds = class_golds.reshape(-1,self.segment_length)
        num = (class_golds != 0).sum(axis=1)
        num[num==0] = 1
        golds = class_golds.sum(axis=-1) // num

        sequence_segment_scores[ood_segment_preds==0] = -1
        sequence_segment_scores_index = np.argmax(sequence_segment_scores, axis=-1)
        preds = ood_segment_preds[np.arange(sequence_segment_scores_index.shape[0]), sequence_segment_scores_index]

        return preds, golds

    def logits_to_scores(self, class_logits):
        token_indices_expanded = self.class_token.T.unsqueeze(0).expand(class_logits.shape[0], -1, -1).to(class_logits.device)
        selected_logits = torch.gather(class_logits, 2, token_indices_expanded)
        probs = selected_logits
        sequence_scores = probs.prod(dim=-2)
        return sequence_scores
    
    def process_gather_logits(self, logits, labels):
        ori_logits, hidden_states, token_type_ids, lm_head_weight = logits

        

        ori_logits = ori_logits[:,:-1]
        class_logits = ori_logits[token_type_ids[:,1:]==3].reshape(-1,self.class_token.shape[-1], ori_logits.shape[-1])
        token_indices_expanded = self.class_token.T.unsqueeze(0).expand(class_logits.shape[0], -1, -1).to(class_logits.device)
        selected_logits = torch.gather(class_logits, 2, token_indices_expanded)
        
        
        
        return (selected_logits, hidden_states, token_type_ids, lm_head_weight)

    def cal_metrics(self, preds, golds):
        
        metrics = classification_report(golds, preds, output_dict=True, zero_division=0)
        final_metrics = metrics['macro avg']
        K_F1 = np.array([metrics[str(i)]['f1-score'] for i in self.id_list if str(i) in metrics]).mean()
        N_F1 = np.array([metrics[str(i)]['f1-score'] for i in self.ood_list  if str(i) in metrics]).mean()
        final_metrics.update({'K-f1': K_F1})
        final_metrics.update({'N-f1': N_F1})
        final_metrics.update({'accuracy': metrics['accuracy']})
        return final_metrics

    def cal_ood_metrics(self, preds, golds):
        metrics = classification_report(golds, preds, output_dict=True, zero_division=0)
        final_metrics = metrics['macro avg']
        final_metrics.update({'accuracy': metrics['accuracy']})

        if '1' in metrics:
            N_metrics = metrics['1']
            N_metrics = {f"N_{i}":v for i,v in N_metrics.items()}
            final_metrics.update(N_metrics)

        K_metrics = metrics['0']
        K_metrics = {f"K_{i}":v for i,v in K_metrics.items()}
        final_metrics.update(K_metrics)
        final_metrics = {f"OOD-{i}":v for i,v in final_metrics.items()}

        return final_metrics
    
    def compute_metrics(self, eval_preds, compute_result=True, eval_ablation_list=['binary']):
        if len(eval_preds[0]) == 3:
            (selected_logits, class_hidden_states, token_type_ids), labels = eval_preds
        else:
            (selected_logits, class_hidden_states, token_type_ids, lm_head_weight), labels = eval_preds

        torch.save(class_hidden_states, f"{self.output_dir}/ood_eval_scores/{self.mode}_class_hidden_states.pt")
        torch.save(lm_head_weight[0], f"{self.output_dir}/ood_eval_scores/{self.mode}_lm_head_weight.pt")
        torch.save(selected_logits, f"{self.output_dir}/ood_eval_scores/{self.mode}_selected_logits.pt")

        sequence_scores = selected_logits.prod(axis=-2)

        all_class_preds = np.argmax(sequence_scores, axis=-1)
        
        
        A = labels[token_type_ids == 3].reshape(-1, self.class_token.shape[1])
        matches = np.all(A[:, np.newaxis, :] == self.class_token.cpu().numpy(), axis=2) 
        all_class_golds = np.argmax(matches, axis=1)
        assert (~np.any(matches, axis=1)).sum() == 0
        
        ood_golds = (all_class_golds==0).astype(int)
        all_metrics = {}
        
        for eval_ablation in eval_ablation_list:
            eval_metrics = {}
            if 'scores' in eval_ablation:
                if 'logits' in eval_ablation:
                    scores = selected_logits.prod(axis=-2)[:,1:]
                    
                    normal_scores = (scores - scores.min()) / (scores.max() - scores.min())
                
                    if 'msp' in eval_ablation:
                        ood_probs = 1 - normal_scores.max(axis=-1)
                    elif 'energy' in eval_ablation:
                        ood_probs = 1 - np.log(np.sum(np.exp(normal_scores), axis=1))

                elif 'distance' in eval_ablation:
                    train_class_hidden_states = torch.tensor(torch.load(f'{self.output_dir}/ood_eval_scores/train-ood_class_hidden_states.pt', weights_only=False))
                    train_golds = torch.tensor(torch.load(f'{self.output_dir}/ood_eval_scores/train-ood_golds.pt', weights_only=False))
                    train_class_hidden_states_flat = train_class_hidden_states.reshape(train_class_hidden_states.shape[0], -1)
                    detector = Mahalanobis(None, eps=0.0)
                    detector.fit_features(train_class_hidden_states_flat[train_golds!=0], train_golds[train_golds!=0] - 1)
                    class_hidden_states_flat = class_hidden_states.reshape(class_hidden_states.shape[0], -1)

                    if 'cosine' in eval_ablation:
                        distances = 1 - cosine_similarity_batch(class_hidden_states_flat, detector.mu).min(axis=-1)
                        
                    elif 'maha' in eval_ablation:
                        distances = detector.predict_features(torch.tensor(class_hidden_states_flat)).cpu().numpy()
                    ood_probs = (distances - distances.min()) / (distances.max() - distances.min())

                if 'test' in self.mode:
                    best_thred = np.load(f"{self.output_dir}/ood_eval_scores/best_{eval_ablation}_thred.npy")
                else:
                    if ood_golds.sum() > 0:
                        fpr, tpr, thresholds = sklearn.metrics.roc_curve(ood_golds, ood_probs, pos_label=1)
                        auc = sklearn.metrics.auc(fpr, tpr)
                        best_thred = thresholds[np.argmax(tpr - fpr)]
                    else:
                        best_thred = 0.5
                        auc = 0

                    np.save(f"{self.output_dir}/ood_eval_scores/best_{eval_ablation}_auc.npy", auc)
                    np.save(f"{self.output_dir}/ood_eval_scores/best_{eval_ablation}_thred.npy", best_thred)
                    print(f"The best thred of {eval_ablation} is {best_thred}, the auc is {auc}")
                    eval_metrics['ood_thred'] = best_thred
                    eval_metrics['ood_auc'] = auc

                ood_preds = (ood_probs>best_thred).astype(int)
                class_preds = copy.deepcopy(all_class_preds)
                class_preds[ood_preds==1] = 0

            elif 'interfere' in eval_ablation:
                _, trend, orthogonal_type = eval_ablation.split('-')
                ood_preds = (all_class_preds==0).astype(int)
                lm_head_weight = torch.tensor(lm_head_weight, dtype=torch.bfloat16)
                class_hidden_states = torch.tensor(class_hidden_states, dtype=lm_head_weight.dtype)
                fc_weight = copy.deepcopy(lm_head_weight[0][0])
                fc_weight = fc_weight.to(class_hidden_states.device)

                if 'test' in self.mode:
                    best_delta_W_0 = torch.load(f"{self.output_dir}/ood_eval_scores/best_delta_W_0-{trend}-{orthogonal_type}.pt", weights_only=False)
                else:
                    X_correct = class_hidden_states[ood_golds==ood_preds].permute(1,0,2)
                    X_pos = class_hidden_states[(ood_golds!=ood_preds) & (ood_golds==1)].permute(1,0,2)
                    X_neg = class_hidden_states[(ood_golds!=ood_preds) & (ood_golds==0)].permute(1,0,2)
                    lm_weight = lm_head_weight[0][1:].permute(1,0,2)
                    if X_neg.shape[1] == 0:
                        X_neg = torch.zeros([X_neg.shape[0], 1, X_neg.shape[2]]).to(torch.float32)
                    if X_pos.shape[1] == 0:
                        X_pos = torch.zeros([X_pos.shape[0], 1, X_pos.shape[2]]).to(torch.float32)

                    best_i = 0
                    max_num = 10
                    max_range_num = 10
                    ood_all_metrics = {}
                    P_proj, T_pos, T_neg = get_vector(X_correct, X_pos, X_neg, lm_weight, trend=trend, orthogonal_type=orthogonal_type)
                    for i in tqdm(range(max_range_num)):
                        delta_W_0 = interfere(P_proj, T_pos, T_neg, i / max_num)

                        shift_fc_weight = fc_weight + delta_W_0.to(fc_weight.dtype)
                        pred_logits = (class_hidden_states * shift_fc_weight).sum(dim=-1)
                        new_selected_logits = torch.tensor(selected_logits, dtype=torch.bfloat16)
                        new_selected_logits[:,:,0] = pred_logits
                        new_sequence_scores = new_selected_logits.prod(axis=-2)

                        class_preds = torch.argmax(new_sequence_scores, axis=-1)
                        ood_preds = (class_preds==0).int().numpy()
                        ood_metrics = self.cal_ood_metrics(ood_preds, ood_golds)
                        ood_all_metrics[i] = ood_metrics['OOD-N_recall']
                        if ood_all_metrics[i] > ood_all_metrics[best_i]:
                            best_i = i
                    best_delta_W_0 = interfere(P_proj, T_pos, T_neg, best_i / max_num)
                    torch.save(best_delta_W_0, f"{self.output_dir}/ood_eval_scores/best_delta_W_0-{trend}-{orthogonal_type}.pt")



                shift_fc_weight = fc_weight + best_delta_W_0.to(fc_weight.dtype)
                pred_logits = (class_hidden_states * shift_fc_weight).sum(dim=-1)
                new_selected_logits = torch.tensor(selected_logits, dtype=torch.bfloat16)
                new_selected_logits[:,:,0] = pred_logits
                new_sequence_scores = new_selected_logits.prod(axis=-2)
                class_preds = torch.argmax(new_sequence_scores, axis=-1)
                ood_preds = (class_preds==0).int().numpy()
                
                ood_scores = new_sequence_scores[:,0]
                ood_probs = (ood_scores - ood_scores.min()) / (ood_scores.max() - ood_scores.min())            

            elif eval_ablation in ['class']:
                ood_scores = sequence_scores[:,0]
                
                ood_probs = (ood_scores - ood_scores.min()) / (ood_scores.max() - ood_scores.min())
                if 'test' in self.mode:
                    best_thred = np.load(f"{self.output_dir}/ood_eval_scores/best_{eval_ablation}_thred.npy")
                else:
                    if ood_golds.sum() > 0:
                        fpr, tpr, thresholds = sklearn.metrics.roc_curve(ood_golds, ood_probs, pos_label=1)
                        auc = sklearn.metrics.auc(fpr, tpr)
                        best_thred = thresholds[np.argmax(tpr - fpr)]
                    else:
                        
                        best_thred = 0.5
                        auc = 0
                    np.save(f"{self.output_dir}/ood_eval_scores/best_{eval_ablation}_thred.npy", best_thred)
                    np.save(f"{self.output_dir}/ood_eval_scores/best_{eval_ablation}_auc.npy", auc)
                    print(f"The best thred of {eval_ablation} is {best_thred}, the auc is {auc}")
                    eval_metrics['ood_thred'] = best_thred
                    eval_metrics['ood_auc'] = auc
                ood_preds = (ood_probs>best_thred).astype(int)
                class_preds = copy.deepcopy(all_class_preds)
                class_preds[ood_preds==1] = 0
            else:
                ood_scores = sequence_scores[:,0]
                ood_probs = (ood_scores - ood_scores.min()) / (ood_scores.max() - ood_scores.min())
                class_preds = copy.deepcopy(all_class_preds)
                ood_preds = (class_preds==0).astype(int)

            torch.save(all_class_golds, f"{self.output_dir}/ood_eval_scores/{self.mode}_golds.pt")
            torch.save(class_preds, f"{self.output_dir}/ood_eval_scores/{self.mode}_{eval_ablation}_preds.pt")

            ood_metrics = self.cal_ood_metrics(ood_preds, ood_golds)

            # fpr, tpr, thresholds = sklearn.metrics.roc_curve(ood_golds, ood_probs, pos_label=1)

            ood_probs_np = to_numpy_float32(ood_probs)
            ood_golds_np = to_numpy_float32(ood_golds)

            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                ood_golds_np,
                ood_probs_np,
                pos_label=1
            )            

            auc = sklearn.metrics.auc(fpr, tpr)
            ood_metrics["AUC"] = auc
            
            preds, golds = self.preprocess_predictions(class_preds, all_class_golds, sequence_scores)
            final_metrics = self.cal_metrics(preds, golds)
            final_metrics.update(ood_metrics)
            final_metrics.update(eval_metrics)
            for key, value in final_metrics.items():
                all_metrics[f"{key}--{eval_ablation}"] = value
        return all_metrics
    
    def preprocess_logits_for_metrics(self, logits, labels):
        return logits
    
    def compute_anology_metrics(self, analogy_preds, class_golds, sequence_scores):
    
        preds, golds = self.preprocess_predictions(analogy_preds, class_golds, sequence_scores)
        final_metrics = self.cal_metrics(preds, golds)

        return final_metrics
