import os
import pandas as pd
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_json(r"C:\Users\Kostya\Desktop\СУСУР\Дипломная работа\balanced_dataset11.json")
df = df.dropna(subset=["message"])

# 2. Определение датасета
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

# Инициализация токенизатора и датасета / DataLoader
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 128
dataset = AnomalyDataset(df, tokenizer, max_length)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Извлечение эмбеддингов из BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
bert_model.eval()

all_features = []
all_labels = []

with torch.no_grad():
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().numpy()

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # Пуллерный выход [CLS]-токена размером [batch_size, 768]
        features = outputs.pooler_output  

        all_features.append(features.cpu().numpy())
        all_labels.append(labels)

features_array = np.concatenate(all_features, axis=0)
labels_array = np.concatenate(all_labels, axis=0)

# 4. Масштабирование признаков (StandardScaler)
scaler = StandardScaler()
features_array_scaled = scaler.fit_transform(features_array)

# 5. Обучение OneClassSVM (только на нормальных объектах: label=0)
normal_features_scaled = features_array_scaled[labels_array == 0]
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
ocsvm.fit(normal_features_scaled)

# 6. Предсказания: +1 — в норме (inlier), -1 — аномалия (outlier)
predictions = ocsvm.predict(features_array_scaled)
predicted_labels = np.where(predictions == 1, 0, 1)  # приводим к (0 - норма, 1 - аномалия)

print("Classification report:")
print(classification_report(labels_array, predicted_labels))
print("Confusion matrix:")
print(confusion_matrix(labels_array, predicted_labels))

# 6а. Вычисление ROC AUC и Average Precision по decision_function
scores = ocsvm.decision_function(features_array_scaled)
roc_auc = roc_auc_score(labels_array, scores)
ap_score = average_precision_score(labels_array, scores)
print("ROC AUC:", roc_auc)
print("Average Precision:", ap_score)

# 7. Сохранение всей «цепочки»
save_dir = "saved_pipeline"
os.makedirs(save_dir, exist_ok=True)

# (7a) Сохраняем модель BERT и токенизатор
bert_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# (7b) Сохраняем scaler и модель ocsvm
with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(save_dir, "ocsvm_model.pkl"), "wb") as f:
    pickle.dump(ocsvm, f)

print(f"Вся цепочка сохранена в: {save_dir}")

# 8. Визуализация с помощью PCA в 2D пространстве
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_array_scaled)

plt.figure(figsize=(8, 6))
colors = ["red" if lbl == 1 else "green" for lbl in predicted_labels]
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.5)
plt.title("Визуализация аномалий (красный) и нормальных (зелёный)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()