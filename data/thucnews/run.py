import pandas as pd
import os
from tqdm import tqdm
content_list = []
labels_list = []
root_dir = 'raw_data'
for category in tqdm(os.listdir(root_dir)):
    for i in os.listdir(f"{root_dir}/{category}"):
        path = f"{root_dir}/{category}/{i}"
        content = open(path, 'r').read()[:10]
        content_list.append(content)
        labels_list.append(category)
ans = pd.DataFrame(labels_list, content_list).reset_index()
ans.columns = ['text', 'label']
ans.to_parquet('origin_data/total.parquet')