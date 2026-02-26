import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import os

def obtain_bert_embedding(df_train, df_eval, df_test, model_name):
    df = pd.concat([df_train, df_eval])

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    def compute_bert_embeddings(texts, batch_size=128):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)

            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  
                embeddings.append(cls_embeddings.cpu())
        return torch.cat(embeddings, dim=0)

    
    bert_embeddings_train = compute_bert_embeddings(df['text'].tolist()).detach().cpu()
    bert_embeddings_test = compute_bert_embeddings(df_test['text'].tolist()).detach().cpu()


    return bert_embeddings_train, bert_embeddings_test


if __name__ == '__main__':
    choose_dataset_list = ['banking', 'clinc', 'stackoverflow', 'atis', 'ele', 'news', 'snips', 'thucnews', 'reuters', 'mcid', 'linkedln']
    
    for dataset_name in choose_dataset_list:
        df_train = pd.read_csv(f'data/{dataset_name}/origin_data/train.tsv', sep='\t')
        df_eval = pd.read_csv(f'data/{dataset_name}/origin_data/eval.tsv', sep='\t')
        df_test = pd.read_csv(f'data/{dataset_name}/origin_data/test.tsv', sep='\t')
        model_name = 'llms/bert-base-chinese' if dataset_name =='thucnews' else 'llms/bert-base-uncased'
        bert_embeddings_train, bert_embeddings_test = obtain_bert_embedding(df_train, df_eval, df_test, model_name)
        os.makedirs(f"data/bert_embeddings/{dataset_name}", exist_ok=True)
        torch.save(bert_embeddings_train, f"data/bert_embeddings/{dataset_name}/traineval_bert.pt")
        torch.save(bert_embeddings_test, f"data/bert_embeddings/{dataset_name}/test_bert.pt")
