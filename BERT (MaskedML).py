from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import pandas as pd

# Берем сначала укропу (датасет)
df_corpus = pd.read_json(r"C:\Users\Kostya\Desktop\СУСУР\Дипломная работа\test1_data.json")


# Потом кота ( надо именно из huggingface/datasets иначе trainer крашится)
dataset = Dataset.from_pandas(df_corpus[['text']])

# Загружаем базовый токенайзер
base_model = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(base_model)

# создаем токены (25 картошек берем еще)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Применяем токенизацию к дате
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ведро воды
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# короче без этого не работает, это валидация
# я уже хочу выйти из окна........
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_data = split_dataset["train"]
test_data  = split_dataset["test"]

# моделька (машинка - волга, такую хочу)
model = BertForMaskedLM.from_pretrained(base_model)

# параметры обучения
training_args = TrainingArguments(
    output_dir="./domain_bert_mlm",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_steps=1000,
    logging_steps=500,
    save_total_limit=1
)

# тренируйся
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# обучаем
trainer.train()

# сохраняем дообученную модель 
trainer.save_model("./domain_bert_mlm") 
tokenizer.save_pretrained("./domain_bert_mlm")

print("Завершено дообучение BERT на MLM, модель сохранена")