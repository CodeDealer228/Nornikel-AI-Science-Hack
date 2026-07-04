import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
from nltk.stem import SnowballStemmer

# === Инициализация стеммера ===
stemmer = SnowballStemmer('russian')

def stem_text(text: str) -> str:
    """Стемминг текста: токенизация + стемминг каждого слова"""
    # Убираем пунктуацию и приводим к lower
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Токенизация и стемминг
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def stem_mentions(mentions: List[str]) -> Set[str]:
    """Стемминг списка mentions"""
    return {stem_text(m) for m in mentions}

def load_predictions(pred_file: str) -> Dict[str, list]:
    pred_by_doc = defaultdict(list)
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                pred_by_doc[data['doc_id']].append(data)
    return pred_by_doc

def load_gold(gold_file: str) -> Dict[str, list]:
    gold_by_doc = defaultdict(list)
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                gold_by_doc[data['source_file']].append(data)
    return gold_by_doc

def extract_entities_with_stemmed_mentions(chunks: List[dict], is_pred: bool = False) -> List[dict]:
    """Извлекает сущности со стеммингованными mentions"""
    key_field = 'parsed' if is_pred else 'gold'
    entities = []
    for chunk in chunks:
        for ent in chunk.get(key_field, {}).get('entities', []):
            entities.append({
                'local_id': ent['local_id'],
                'type': ent['type'],
                'canonical_name': ent['canonical_name'],
                'stemmed_mentions': stem_mentions(ent.get('mentions', [])),
                'stemmed_canonical': stem_text(ent['canonical_name'])
            })
    return entities

def extract_relations_stemmed(chunks: List[dict], is_pred: bool = False) -> Tuple[Set[Tuple], Dict[str, dict]]:
    """Извлекает отношения со стеммингованными canonical_name"""
    key_field = 'parsed' if is_pred else 'gold'
    
    # Сначала соберём маппинг local_id → entity
    local_to_entity = {}
    for chunk in chunks:
        for ent in chunk.get(key_field, {}).get('entities', []):
            local_to_entity[ent['local_id']] = {
                'stemmed_canonical': stem_text(ent['canonical_name']),
                'type': ent['type']
            }
    
    relations = set()
    for chunk in chunks:
        for r in chunk.get(key_field, {}).get('relations', []):
            subj = local_to_entity.get(r['subject'])
            obj = local_to_entity.get(r['object'])
            if subj and obj:
                # Стеммингованное отношение: (subj_canonical, predicate, obj_canonical)
                relations.add((subj['stemmed_canonical'], r['predicate'], obj['stemmed_canonical']))
    
    return relations, local_to_entity

def match_entities_by_stemmed_mentions(gold_ents: List[dict], pred_ents: List[dict]) -> Tuple[List[str], List[str]]:
    """Матчинг сущностей по пересечению стеммингованных mentions"""
    y_true, y_pred = [], []
    
    # Создаём маппинг: stemmed_mention → entity
    pred_mention_to_ent = {}
    for ent in pred_ents:
        for m in ent['stemmed_mentions']:
            pred_mention_to_ent[m] = ent
    
    matched_pred_ids = set()
    
    # Для каждой gold сущности ищем совпадение в pred
    for gold_ent in gold_ents:
        matched = False
        for m in gold_ent['stemmed_mentions']:
            if m in pred_mention_to_ent:
                pred_ent = pred_mention_to_ent[m]
                # Совпадение по stemmed mention
                y_true.append(gold_ent['type'])
                y_pred.append(pred_ent['type'])
                matched_pred_ids.add(pred_ent['local_id'])
                matched = True
                break
        
        if not matched:
            # Не нашли совпадения → FN
            y_true.append(gold_ent['type'])
            y_pred.append('O')
    
    # Добавляем unmatched pred как FP
    for pred_ent in pred_ents:
        if pred_ent['local_id'] not in matched_pred_ids:
            y_true.append('O')
            y_pred.append(pred_ent['type'])
    
    return y_true, y_pred

def compute_metrics(y_true: List[str], y_pred: List[str]) -> dict:
    # Убираем 'O' из списка классов
    labels = sorted(set(y_true + y_pred) - {'O'})
    
    true_counter = Counter(y_true)
    pred_counter = Counter(y_pred)
    tp_counter = Counter([t for t, p in zip(y_true, y_pred) if t == p and t != 'O'])
    
    # Per-class metrics
    per_class = {}
    for label in labels:
        tp = tp_counter[label]
        fp = pred_counter[label] - tp
        fn = true_counter[label] - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        per_class[label] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': true_counter[label]}
    
    # Micro F1
    total_tp = sum(tp_counter.values())
    total_fp = sum(pred_counter[l] for l in labels) - total_tp
    total_fn = sum(true_counter[l] for l in labels) - total_tp
    micro_prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if micro_prec + micro_rec > 0 else 0.0
    
    # Macro F1
    macro_f1 = sum(per_class[l]['f1'] for l in labels) / len(labels) if labels else 0.0
    
    # Weighted F1
    total_support = sum(true_counter[l] for l in labels)
    weighted_f1 = sum(per_class[l]['f1'] * true_counter[l] for l in labels) / total_support if total_support > 0 else 0.0
    
    return {
        'labels': labels,
        'per_class': per_class,
        'micro_f1': micro_f1,
        'micro_precision': micro_prec,
        'micro_recall': micro_rec,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

# === Настройка путей ===
PRED_FILE = r"extraction_results_statyi_batch.jsonl"
GOLD_FILE = r"golden_set/golden_set.jsonl"

# === Загрузка ===
print("Загрузка данных...")
pred_docs = load_predictions(PRED_FILE)
gold_docs = load_gold(GOLD_FILE)

# === Нахождение пересечения документов ===
pred_keys = set(pred_docs.keys())
gold_keys = set(gold_docs.keys())

# Нормализуем gold keys для сравнения
gold_keys_normalized = {Path(k).stem for k in gold_keys}

# Находим пересечение
common_docs = pred_keys & gold_keys_normalized

print(f"Всего документов в pred: {len(pred_keys)}")
print(f"Всего документов в gold: {len(gold_keys_normalized)}")
print(f"Общих документов: {len(common_docs)}")

if not common_docs:
    print("\n❌ Нет общих документов! Проверьте имена файлов.")
    exit(1)

# === Обработка только общих документов ===
y_true_ner, y_pred_ner = [], []
y_true_re, y_pred_re = [], []

print("\nОбработка документов...")
for doc_name in common_docs:
    # Находим оригинальный ключ в gold
    gold_key = None
    for k in gold_keys:
        if Path(k).stem == doc_name:
            gold_key = k
            break
    
    gold_chunks = gold_docs[gold_key]
    pred_chunks = pred_docs[doc_name]
    
    # Извлекаем entities со стеммингом
    gold_ents = extract_entities_with_stemmed_mentions(gold_chunks, is_pred=False)
    pred_ents = extract_entities_with_stemmed_mentions(pred_chunks, is_pred=True)
    
    # Матчинг NER по стеммингованным mentions
    y_true_doc, y_pred_doc = match_entities_by_stemmed_mentions(gold_ents, pred_ents)
    y_true_ner.extend(y_true_doc)
    y_pred_ner.extend(y_pred_doc)
    
    # Матчинг RE
    gold_rels, _ = extract_relations_stemmed(gold_chunks, is_pred=False)
    pred_rels, _ = extract_relations_stemmed(pred_chunks, is_pred=True)
    
    all_rels = gold_rels | pred_rels
    for rel in all_rels:
        subj, pred_type, obj = rel
        is_true = rel in gold_rels
        is_pred = rel in pred_rels
        
        if is_true and is_pred:
            # TP
            y_true_re.append(pred_type)
            y_pred_re.append(pred_type)
        elif is_true and not is_pred:
            # FN
            y_true_re.append(pred_type)
            y_pred_re.append('O')
        elif not is_true and is_pred:
            # FP
            y_true_re.append('O')
            y_pred_re.append(pred_type)

# === Вычисление метрик ===
print("\nВычисление метрик...")
ner_res = compute_metrics(y_true_ner, y_pred_ner)
re_res = compute_metrics(y_true_re, y_pred_re)

# === Вывод результатов ===
def print_results(task_name: str, res: dict):
    print(f"\n{'='*60}")
    print(f"=== {task_name} ===")
    print(f"{'='*60}")
    print(f"Micro F1:        {res['micro_f1']:.4f}")
    print(f"Micro Precision: {res['micro_precision']:.4f}")
    print(f"Micro Recall:    {res['micro_recall']:.4f}")
    print(f"Macro F1:        {res['macro_f1']:.4f}")
    print(f"Weighted F1:     {res['weighted_f1']:.4f}")
    print(f"\nПо классам:")
    print(f"{'Класс':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 71)
    for label in res['labels']:
        p = res['per_class'][label]['precision']
        r = res['per_class'][label]['recall']
        f1 = res['per_class'][label]['f1']
        supp = res['per_class'][label]['support']
        print(f"{label:<25} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {supp:<10}")

print_results("NER", ner_res)
print_results("RE", re_res)

# === Дополнительная статистика ===
print(f"\n{'='*60}")
print("Статистика:")
print(f"NER: TP={sum(1 for t, p in zip(y_true_ner, y_pred_ner) if t == p and t != 'O')}, "
      f"FP={sum(1 for t, p in zip(y_true_ner, y_pred_ner) if t == 'O' and p != 'O')}, "
      f"FN={sum(1 for t, p in zip(y_true_ner, y_pred_ner) if t != 'O' and p == 'O')}")
print(f"RE:  TP={sum(1 for t, p in zip(y_true_re, y_pred_re) if t == p and t != 'O')}, "
      f"FP={sum(1 for t, p in zip(y_true_re, y_pred_re) if t == 'O' and p != 'O')}, "
      f"FN={sum(1 for t, p in zip(y_true_re, y_pred_re) if t != 'O' and p == 'O')}")