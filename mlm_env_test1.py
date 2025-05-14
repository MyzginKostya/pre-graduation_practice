#from transformers import pipeline
#
## Загрузка модели для предсказания [MASK]
#mask_filler = pipeline(
#    "fill-mask",
#    model="bert-base-uncased",  # Можно заменить на "roberta-base", "distilbert-base-uncased" и др.
#    device="cpu"  # Для GPU: "cuda:0" 
#)
#
## Пример предсказания
#results = mask_filler("The capital of France is [MASK].")
#
## Вывод результатов
#for result in results:
#    print(f"Токен: {result['token_str']}, Оценка: {result['score']:.4f}, Фраза: {result['sequence']}")





import pandas as pd
from pathlib import Path
import random
import json#

def load_and_mark_data(normal_path: Path, anomalies_path: Path) -> pd.DataFrame:
    """Загружает данные и автоматически размечает аномалии"""
    # Загрузка данных с явной разметкой
    normal_df = pd.read_json(normal_path, encoding='utf-8')
    anomalies_df = pd.read_json(anomalies_path, encoding='utf-8')
    
    # Добавляем метки
    normal_df['outlier'] = 0  # Все из первого файла - нормальные
    anomalies_df['outlier'] = 1  # Все из второго файла - аномалии
    
    # Проверка обязательных колонок
    required_columns = ['message_id', 'message']
    for col in required_columns:
        if col not in normal_df.columns or col not in anomalies_df.columns:
            raise ValueError(f"Отсутствует обязательная колонка: {col}")
    
    return normal_df, anomalies_df#
def merge_with_duplicate_handling(normal_df: pd.DataFrame, anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """Объединяет датасеты с обработкой дубликатов message_id"""
    # Преобразуем ID в строки
    normal_df['message_id'] = normal_df['message_id'].astype(str)
    anomalies_df['message_id'] = anomalies_df['message_id'].astype(str)
    
    # Обработка дубликатов
    duplicate_mask = anomalies_df['message_id'].isin(normal_df['message_id'])
    anomalies_df.loc[duplicate_mask, 'message_id'] += '_anomaly'
    
    return pd.concat([normal_df, anomalies_df], ignore_index=True)#
def balance_dataset(df: pd.DataFrame, target_ratio: float = 0.7) -> pd.DataFrame:
    normal = df[df['outlier'] == 0]
    anomalies = df[df['outlier'] == 1]#
    # Изначальное количество
    len_normal = len(normal)
    len_anomalies = len(anomalies)#
    # Общее количество
    total_count = len_normal + len_anomalies#
    # Рассчитываем желаемое количество нормальных и аномальных сообщений
    desired_normals_count = int(total_count * target_ratio)
    desired_anomalies_count = total_count - desired_normals_count#
    # Дублируем аномалии, если их недостаточно
    if desired_anomalies_count > len_anomalies:
        needed_anomalies = desired_anomalies_count - len_anomalies
        anomalies = pd.concat([anomalies, anomalies.sample(needed_anomalies, replace=True)], ignore_index=True)
        print(f"Дублировано {needed_anomalies} аномальных сообщений")#
    # Дублируем нормальные сообщения только до нужного количества
    if desired_normals_count > len(normal):
        needed_normals = desired_normals_count - len(normal)
        normal = pd.concat([normal, normal.sample(needed_normals, replace=True)], ignore_index=True)
        print(f"Дублировано {needed_normals} нормальных сообщений")#
    # Итоговое объединение и перемешивание
    balanced = pd.concat([
        normal.sample(n=desired_normals_count, random_state=42),
        anomalies.sample(n=desired_anomalies_count, random_state=42)  # Используем нужное количество аномалий
    ]).sample(frac=1.0, random_state=42)#
    return balanced#
def save_dataset(df: pd.DataFrame, output_path: Path):
    """Сохраняет датасет в нужном формате"""
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif output_path.suffix == '.json':
        df.to_json(output_path, orient='records', force_ascii=False, indent=4)
    else:
        raise ValueError("Поддерживаются только форматы .csv и .json")#
def main():
    try:
        # Пути к файлам
        base_dir = Path(r"C:\Users\Kostya\Desktop\СУСУР\Дипломная работа")
        normal_path = base_dir / "dataset_brokervtb.json"
        anomalies_path = base_dir / "dataset_rbc_investments.json"
        output_path = base_dir / "balanced_dataset_test6.json"#
        
        # 1. Загрузка и разметка данных
        normal_df, anomalies_df = load_and_mark_data(normal_path, anomalies_path)
        print(f"Загружено нормальных сообщений: {len(normal_df)}")
        print(f"Загружено аномалий: {len(anomalies_df)}")
        
        # 2. Объединение с обработкой дубликатов
        merged_df = merge_with_duplicate_handling(normal_df, anomalies_df)
        
        # 3. Балансировка
        balanced_df = balance_dataset(merged_df, target_ratio=0.7)
        
        # 4. Сохранение
        save_dataset(balanced_df, output_path)
        
        # Статистика
        normal_count = len(balanced_df[balanced_df['outlier'] == 0])
        anomaly_count = len(balanced_df) - normal_count
        print("\nРезультат балансировки:")
        print(f"Всего сообщений: {len(balanced_df)}")
        print(f"Нормальные: {normal_count} ({normal_count/len(balanced_df):.1%})")
        print(f"Аномалии: {anomaly_count} ({anomaly_count/len(balanced_df):.1%})")
        print(f"Сохранено в: {output_path}")
        
    except Exception as e:
        print(f"\nОшибка: {str(e)}")#
if __name__ == "__main__":
    main()

