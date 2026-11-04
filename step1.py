import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个耐心的学习助手。"},
        {"role": "user", "content": "用三句话解释什么是 RAG。"},
    ],
)
print(resp.choices[0].message.content)
