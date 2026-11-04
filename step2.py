import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 1. 读取 PDF，把每一页的文字拼起来
pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
reader = PdfReader(pdf_path)
text = "\n".join(page.extract_text() or "" for page in reader.pages)
text = text[:20000]  # 先只取前 2 万字，避免太长
print(f"读取了 {len(reader.pages)} 页，共 {len(text)} 字\n")

# 2. 循环提问：把 PDF 内容和问题一起发给 AI
while True:
    question = input("你的问题（直接回车退出）：").strip()
    if not question:
        break
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你只根据用户提供的文档回答问题。文档里没有的内容，就说不知道。"},
            {"role": "user", "content": f"文档内容：\n{text}\n\n问题：{question}"},
        ],
    )
    print("\nAI：", resp.choices[0].message.content, "\n")
