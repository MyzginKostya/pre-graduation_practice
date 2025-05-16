#где-то на этом этапе человек перестает быть человеком......
import os
import pandas as pd
import torch
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# загружаем размеченный датасет
df = pd.read_json(r"C:\Users\Kostya\Desktop\СУСУР\Дипломная работа\balanced_dataset11.json")
df = df.dropna(subset=["message"])

# создаём PyTorch Dataset для извлечения эмбеддингов BERT
class AnomalyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        message = self.dataframe.loc[idx, "message"]
        label = self.dataframe.loc[idx, "outlier"]
        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }

# загружаем дообученную модель
domain_bert_path = "./domain_bert_mlm"  # путь к сохранённой модели

tokenizer = BertTokenizer.from_pretrained(domain_bert_path)
domain_bert_model = BertModel.from_pretrained(domain_bert_path)
domain_bert_model.eval()

# извлекаем вектора
max_length = 128
dataset = AnomalyDataset(df, tokenizer, max_length)
train_loader = DataLoader(dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
domain_bert_model.to(device)

all_features = []
all_labels = []

with torch.no_grad():
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["label"].cpu().numpy()

        outputs = domain_bert_model(input_ids=input_ids, attention_mask=attention_mask)
        
        
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output.cpu().numpy()
        else:
            features = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        
        all_features.append(features)
        all_labels.append(labels_batch)

features_array = np.concatenate(all_features, axis=0)
labels_array   = np.concatenate(all_labels, axis=0)

print("Получены эмбеддинги из дообученного BERT:", features_array.shape)

# масштабирование признаков перед обучением алгоритма обнаружения аномалий
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)

# отделение нормальных данных (а я уже ненормальный)
normal_data = features_scaled[labels_array == 0]


clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  
clf.fit(normal_data)

# gредсказания по всему набору данных
predictions_raw = clf.predict(features_scaled)  
predicted_labels = np.where(predictions_raw == 1, 0, 1)

# jценка качества модели
print(classification_report(labels_array, predicted_labels))
print(confusion_matrix(labels_array, predicted_labels))

# ROC-AUC 
if hasattr(clf, "decision_function"):
    scores = clf.decision_function(features_scaled)  
    roc_score   = roc_auc_score(labels_array, scores)
    ap_score    = average_precision_score(labels_array, scores)
else:
    scores_samples_normalized   = clf.score_samples(features_scaled)
    scores_outliers             = -scores_samples_normalized  
    roc_score   = roc_auc_score(labels_array, scores_outliers)
    ap_score    = average_precision_score(labels_array, scores_outliers)

print("ROC AUC:", roc_score)
print("Average Precision:", ap_score)


save_dir ="final_domain_pipeline"
os.makedirs(save_dir, exist_ok=True)

domain_bert_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

with open(os.path.join(save_dir,"scaler.pkl"),"wb") as f:
    pickle.dump(scaler,f)

with open(os.path.join(save_dir,"clf.pkl"),"wb") as f:
    pickle.dump(clf,f)

#рисуем картинки всякие там
pca = PCA(n_components=2)
features_2d   = pca.fit_transform(features_scaled)

plt.figure(figsize=(8,6))
colors      =[ "red" if lbl==1 else "green" for lbl in predicted_labels]
plt.scatter(features_2d[:,0], features_2d[:,1], c=colors , alpha=0.5 )
plt.title("Дообученный BERT + OneClassClassifier")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()