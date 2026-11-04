import os
import re
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


def ask(system, user):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


# 1. 切块：把 PDF 按页读出，再切成约 800 字的小段，并记住每段在第几页
def load_chunks(pdf_path, size=800, overlap=150):
    chunks = []
    for page_no, page in enumerate(PdfReader(pdf_path).pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "")
        start = 0
        while start < len(text):
            chunks.append({"page": page_no, "text": text[start:start + size]})
            start += size - overlap
    return chunks


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


# 2. 检索：用 BM25 算法给每段打分，挑出最相关的几段
pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
chunks = load_chunks(pdf_path)
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
print(f"文档被切成了 {len(chunks)} 段\n")


def retrieve(question, k=4):
    # 文档是英文，先让 AI 把问题改写成英文关键词，检索会准很多
    keywords = ask("把用户的问题改写成用于搜索英文文档的英文关键词，只输出关键词，用空格分隔。", question)
    scores = bm25.get_scores(tokenize(keywords))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return keywords, [chunks[i] for i in top]


# 3. 生成：只把相关段落发给 AI，并要求标注出处
def answer(question):
    keywords, hits = retrieve(question)
    context = "\n\n".join(f"[片段{n}，第{h['page']}页] {h['text']}" for n, h in enumerate(hits, 1))
    reply = ask(
        "你只根据给出的文档片段回答问题，并在句末用 [片段N] 标注出处。片段里没有的内容，就说文档里没有提到。",
        f"文档片段：\n{context}\n\n问题：{question}",
    )
    return keywords, hits, reply


if __name__ == "__main__":
    while True:
        q = input("你的问题（直接回车退出）：").strip()
        if not q:
            break
        keywords, hits, reply = answer(q)
        print(f"\n（检索关键词：{keywords}）")
        print("（找到的片段来自第", ", ".join(str(h["page"]) for h in hits), "页）")
        print("\nAI：", reply, "\n")
