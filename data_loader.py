import re, unicodedata
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from dataclasses import dataclass, field

@dataclass
class Passage:
    id: str
    title: str
    text: str
    answers_question_ids: list = field(default_factory=list)  

@dataclass
class QAItem:
    qid: str
    question: str
    answer: str
    answerable: bool = True
    correct_article_titles: list = field(default_factory=list)

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)        
    text = re.sub(r"[ \t]+", " ", text)   
    return text.strip()


def remove_duplicates(passages: list):
    checker = MinHashLSH(threshold=0.9, num_perm=64)

    final_passage = []
    cleared = 0

    for p in passages:
        id_card = MinHash(num_perm=64)

        words = p.title.split()

        for word in words:
            id_card.update(word.encode('utf8'))

        if checker.query(id_card):
            cleared = cleared + 1 
            continue
        else:
            checker.insert(p.id,id_card)
            final_passage.append()


def get_datasets():
    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)

    passages = {}
    question = []

    for ex in hotpot_dataset:
        gold = list(set(ex['supporting_facts']['title']))
        
        for t, ss in zip(ex["context"]["title"], ex["context"]["sentences"]):
            empty_list = []
            for s in ss:
                empty_list.append(s.strip())
            para = " ".join(empty_list)

            if len(para) < 40:
                continue

            pid = f"{t}::0"

            p = passages.setdefault(pid, Passage(id=pid, title=t, text=para, answers_question_ids=[]))

            if t in gold:
                p.answers_question_ids.append(ex["id"])

        question.append(QAItem(ex["id"], ex["question"], ex["answer"], True, correct_article_titles=gold))

        if len(passages) >= 5000:
            break

    return list(passages.values()), question