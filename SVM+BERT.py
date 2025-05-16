import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_json(r"C:\Users\Kostya\Desktop\СУСУР\Дипломная работа\balanced_dataset11.json")
df = df.dropna(subset=["message"])

#Определение датасета
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
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.float)
        }

# Совместная модель BERT + Классификатор
class JointModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

#Инициализация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
model = JointModel(bert_model).to(device)

# Параметры обучения
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCELoss()
epochs = 5
batch_size = 16

#Подготовка данных
dataset = AnomalyDataset(df, tokenizer, max_length=128)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Обучение с совместной оптимизацией
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids, attention_mask).squeeze()
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

#Сохранение модели
save_dir = "saved_joint_model"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "joint_model.pt"))
tokenizer.save_pretrained(save_dir)

# Оценка модели
model.eval()
all_preds = []
all_labels = []
all_scores = []

with torch.no_grad():
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().numpy()
        
        outputs = model(input_ids, attention_mask).squeeze().cpu().numpy()
        preds = (outputs > 0.5).astype(int)
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_scores.extend(outputs)

# Метрики
print("Classification report:")
print(classification_report(all_labels, all_preds))
print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))
print("ROC AUC:", roc_auc_score(all_labels, all_scores))
print("Average Precision:", average_precision_score(all_labels, all_scores))

#Визуализация
def get_features(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            features.append(outputs.pooler_output.cpu().numpy())
    return np.concatenate(features, axis=0)

features = get_features(model, train_loader)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
colors = ["red" if lbl == 1 else "green" for lbl in all_preds]
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.5)
plt.title("Визуализация аномалий (красный) и нормальных (зелёный)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()