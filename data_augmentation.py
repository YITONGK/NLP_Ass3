import json
import random
from pathlib import Path
from collections import Counter
import nltk
from nltk.corpus import wordnet

# 如果还没下载：
nltk.download('punkt')
nltk.download('wordnet')

# ── 1) 加载原始训练集 ───────────────────────────────────────────────
DATA_DIR     = Path("data")
TRAIN_J      = DATA_DIR/"train-claims.json"
DEV_J        = DATA_DIR/"dev-claims.json"
OUT_AUG_J    = DATA_DIR/"dev-claims-augmented.json"

with open(TRAIN_J, "r", encoding="utf-8") as f:
    train = json.load(f)
with open(DEV_J, "r", encoding="utf-8") as f:
    dev = json.load(f)
# ── 2) 定义 EDA 函数 ────────────────────────────────────────────────

def get_synonyms(word):
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                syns.add(name)
    return list(syns)

def synonym_replacement(words, n):
    new_words = words.copy()
    candidates = [i for i,w in enumerate(words) if get_synonyms(w)]
    random.shuffle(candidates)
    for i in candidates[:n]:
        new_words[i] = random.choice(get_synonyms(words[i]))
    return new_words

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        candidates = [w for w in words if get_synonyms(w)]
        if not candidates:
            break
        word = random.choice(candidates)
        new_words.insert(random.randrange(len(new_words)), random.choice(get_synonyms(word)))
    return new_words

def random_swap(words, n):
    new_words = words.copy()
    length = len(new_words)
    for _ in range(n):
        i, j = random.sample(range(length), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words

def random_deletion(words, p):
    if len(words) == 1:
        return words
    return [w for w in words if random.random() > p] or [random.choice(words)]

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
    words = nltk.word_tokenize(sentence)
    num_words = len(words)
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))
    out = []
    out.append(" ".join(synonym_replacement(words, n_sr)))
    out.append(" ".join(random_insertion(words, n_ri)))
    out.append(" ".join(random_swap(words, n_rs)))
    out.append(" ".join(random_deletion(words, p_rd)))
    return out[:num_aug]

# ── 3) 统计每个类别数量，确定目标扩增数 ───────────────────────────────
counts = Counter(obj["claim_label"] for obj in train.values())
max_count = counts.most_common(1)[0][1]   # SUPPORTS 的数量

print("Before augmentation:", counts)

# ── 4) 对少数类做增强 ────────────────────────────────────────────────
augmented = {}
# 先把原始都拷贝过去
augmented.update(train)
augmented.update(dev)

# 按类别收集所有 id
ids_by_label = {}
for cid, obj in train.items():
    lbl = obj["claim_label"]
    ids_by_label.setdefault(lbl, []).append(cid)

for lbl, id_list in ids_by_label.items():
    n_needed = max_count - len(id_list)
    if n_needed <= 0:
        continue
    for i in range(n_needed):
        orig_id = random.choice(id_list)
        orig = train[orig_id]
        aug_text = eda(orig["claim_text"])[i % 4]
        new_id = f"{orig_id}_aug{i}"
        augmented[new_id] = {
            "claim_text": aug_text,
            "claim_label": lbl,
            "evidences": orig["evidences"],
        }

new_counts = Counter(obj["claim_label"] for obj in augmented.values())
print("After augmentation :", new_counts)

# ── 5) 写出增强后的训练集 ──────────────────────────────────────────────
with open(OUT_AUG_J, "w", encoding="utf-8") as f:
    json.dump(augmented, f, ensure_ascii=False, indent=2)

print(f"Saved augmented training set ({len(augmented)} examples) to {OUT_AUG_J}")