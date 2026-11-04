"""溯光的核心逻辑：读文档、切块、建索引、混合检索、Excel 统计、调用 DeepSeek、判断意图、生成回答。
这里不依赖 streamlit，网页（app.py）和评测脚本（eval.py）都直接 import 这个文件。"""
import datetime
import io
import json
import os
import hashlib
import re
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import openai
import pandas as pd
from docx import Document
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from openpyxl import load_workbook
from pypdf import PdfReader
from rank_bm25 import BM25Plus
import rjieba

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---------- 可调参数（数值和改造前完全一样） ----------
MODEL = "deepseek-chat"
API_BASE_URL = "https://api.deepseek.com"
API_TIMEOUT = 60          # 调用 DeepSeek 最多等多少秒
API_MAX_RETRIES = 1       # 出错后自动重试几次
TOP_K = 6                 # 每次最多交给 DeepSeek 的片段数
CHUNK_SIZE = 800          # 每段最多多少字
CHUNK_OVERLAP = 150       # 相邻两段重叠多少字（每次往后挪 CHUNK_SIZE - CHUNK_OVERLAP = 650 字）
GUARANTEED_TOP = 2        # 关键词检索、向量检索各自的前几名一定入选
RRF_POOL = 20             # RRF 融合时每种检索取前多少名
RRF_K = 60                # RRF 公式 1 / (RRF_K + 名次) 里的常数
MAX_GROUP_VALUES = 20     # Excel 统计时，一列最多有多少种不同取值才拿来分组
TABLE_RANK_ROWS = 15      # Excel 统计时，表不超过这么多行就附上每个数字列从大到小的完整排名
# 总结模式（map-reduce）
SUMMARY_BATCH_CHARS = 6000   # 每批最多多少字交给模型做“分段要点”
SUMMARY_MAX_BATCHES = 12     # 所有文档加起来最多分几批；文档太长时均匀抽取段落
SUMMARY_WORKERS = 12         # 同时并发调用几次 DeepSeek（和最大批数一样，一轮就能跑完）
SUMMARY_STATS_CHARS = 6000   # 总结 Excel 时，最多附上多少字的 pandas 统计
SUMMARY_MAX_CITES = 14       # 最终回答最多引用几段原文
SUMMARY_CITES_PER_MARK = 2   # 每处标注最多保留几个编号，让引用分散到每一条要点
SUMMARY_CACHE_SIZE = 512     # 内存里最多缓存多少批的要点
# 嵌入模型：名字 → (问题前缀, 资料前缀, 大约多大)。
# e5 训练时问题前面带 "query: "、资料前面带 "passage: "，用的时候也要加上，效果才对
EMBED_MODELS = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: ", "2.2GB"),                  # 最长 512 token，800 字的段落基本能读完
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ("", "", "240MB"),   # 最长 128 token，只看得到每段开头；体积小，在线体验版用它
}
SMALL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "intfloat/multilingual-e5-large"   # 本地默认用 e5；在线体验版由 app.py 换成小模型


def _pick_cache_dir():
    """嵌入模型放哪：本地用项目里的 models/；云端没有这个文件夹（或者不能写）时，改用系统临时目录。
    也可以用环境变量 FASTEMBED_CACHE_PATH 指定"""
    if os.getenv("FASTEMBED_CACHE_PATH"):
        return os.getenv("FASTEMBED_CACHE_PATH")
    local = os.path.join(BASE_DIR, "models")
    try:
        os.makedirs(local, exist_ok=True)
        if os.access(local, os.W_OK):
            return local
    except OSError:
        pass
    return os.path.join(tempfile.gettempdir(), "suguang-models")


EMBED_CACHE_DIR = _pick_cache_dir()
EMBED_BATCH_SIZE = 32     # 每次算多少段的向量：调小能明显降低内存峰值（云端内存有限），结果不变

# 扫描件 OCR：PDF 里某一页直接读不出字，就把这页“拍成照片”（pypdfium2），再让 OCR 模型看图认字（rapidocr）
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") != "0"   # 在线体验版由 app.py 关掉（云端机器慢）
OCR_SCALE = 2             # 渲染倍数：PDF 一页是 72 dpi，放大 2 倍 = 144 dpi，字够清楚，每页约 2 秒
OCR_MIN_CHARS = 10        # 一页直接读出的字少于这么多，就当成扫描页去识别
OCR_IMAGE_PAGE_CHARS = 200   # 字比这少、而且一张大图盖住了半页以上，也当成扫描页（“扫描全能王”这类 App 会在角上印一行真文字水印）
OCR_IMAGE_RATIO = 0.5
OCR_TAG = "（扫描识别）"   # 加在出处后面，提醒这段文字是认出来的，可能有个别错字

LIBRARY_DIR = os.path.join(BASE_DIR, "library")   # 资料库存放位置（录入的原文件、段落、向量）
# 处理规则的版本号：读文档、切段、OCR 的规则变了就加 1。资料库里用旧规则处理的资料，打开时会自动从原文件重新处理一次。
# 1 = 第 5 步以前（没有 OCR）；2 = 有 OCR，也能识别带文字水印的扫描页；
PIPELINE_VERSION = 3   # 3 = OCR 换行拼接改成按大多数行的右边判断（第四轮审查 R1），旧扫描件会重新识别一次

ERROR_PREFIX = "⚠️"       # ask() 出错时返回的提示都以它开头
NO_TEXT_REPLY = "现在没有可以检索的文档内容（文件可能是扫描件），请上传带文字的文档后再问。"

INTRO = """✨ 我是**溯光**，你的文档问答助手。几百页的资料，几秒读懂；每一句回答，都能溯回原文。

- 📄 **丢进文档就能问**：PDF、Word、Excel 都行，讲义、合同、报表可以一次上传多份
- 🎯 **每句话都有出处**：回答会标注来自哪个文件的哪个位置，点开就能核对原文，不怕 AI 瞎编
- 🌏 **中英文都能问**：英文论文也能直接用中文提问、中文回答
- 📊 **表格能算数**：Excel 里的合计、平均、最多最少、排名，程序先算好再回答

试试在左边上传一份文档，然后问我"这份文档的核心观点是什么？"
"""

# ---------- 提示词 ----------
PROMPT_INTENT = (
    "判断用户这句话的类型：如果是在问你能做什么、有什么功能、怎么用、你是谁，输出 INTRO；如果是打招呼、闲聊、感谢，输出 CHAT；"
    "如果需要对表格数据做统计计算（合计、求和、平均、最多、最少、最高、最低、排名、一共多少），输出 CALC；"
    "如果是要求概括整份文档（核心观点、主要讲了什么、总结全文、概括、要点、大意；问某一份文档、文件、档案、资料、材料、报告、书整体讲什么也算），"
    "输出 SUMMARY（只总结某一章、某一条款、某个话题的不算，输出 DOC）；其他任何问题输出 DOC。只输出这一个词。"
)
# 问题里点名了资料时，接在 PROMPT_INTENT 后面：顺便认出点名的是哪几份（不多调用一次 DeepSeek）
PROMPT_PICK_DOCS = (
    "\n\n补充：下面是资料清单（编号是页面侧栏里的顺序，清单可能只列出了一部分）。如果问题里明确点名了其中一份或几份——按编号（“第一个”“第2份”“最后一个”）、"
    "按名称（“NIST 那份”）、或按描述（“讲物联网的那个文件”“那篇英文论文”）——就在类型后面空一格写编号，多个只用逗号分隔，例如：SUMMARY 1 或 DOC 2,4。"
    "问题里没有点名资料时只输出类型；不要因为猜测某份资料里可能有答案就写编号（例如“押金是多少？”没有点名，只输出 DOC）。\n资料清单：\n{docs}"
)
# 先在本地判断问题是不是真的在点名资料，是的话才把资料清单发给 DeepSeek（宁可少认：没认出只是在全部资料里找，认错了答案会被漏掉）
_DOC_NOUN = r"(?:文件|文档|档案|资料|材料|报告|论文|表格|表|合同|手册|说明书|方案|公报|书)"
ORDINAL_RE = re.compile(r"(?:第\s*[一二三四五六七八九十\d]+|最后[一]?|倒数第[一二三四五六七八九十\d]+)\s*[个份本篇张]"
                        rf"(?=\s*(?:{_DOC_NOUN}|里|中|的|讲|主要|说|写|和|与|跟|、|，|,|？|\?|$))")
DESCRIBE_RE = re.compile(rf"[这那][一]?[个份篇本张][^，。？?]{{0,10}}?{_DOC_NOUN}")
DOC_WORD_RE = re.compile(rf"{_DOC_NOUN}|[这那][一]?[份篇本张]")   # 不含“这个、那个”：太常见（“那个时候”）
_NAME_STOP = {"pdf", "docx", "xlsx", "doc", "xls", "test", "paper", "data", "file", "report", "copy", "scan", "final", "new", "the", "and",
              # 资料名里常见、提问时也常说的中文词，不能拿来认资料（“NIST 那份文档主要讲了什么”里的“主要”）
              "主要", "中国", "国家", "地区", "全国", "发展", "创新", "产业", "行动", "统计", "工作", "情况", "内容", "问题", "研究", "分析",
              "报告", "方案", "文件", "资料", "文档", "材料", "说明", "总结", "计划", "规划", "管理", "技术", "关于", "通知"}


def name_words(label):
    """资料名里能拿来认的词：去掉扩展名和“（11-03 19:56 录入）”这类后缀，英文按完整单词（3 个字母以上），中文按分词（2 个字以上）"""
    core = re.sub(r"（[^）]*录入）$", "", label)
    core = re.sub(r"\.\w+$", "", core).lower()
    eng = {w for w in re.findall(r"[a-z]{3,}", core) if w not in _NAME_STOP}
    zh = {w for w in rjieba.cut(core) if re.fullmatch(r"[\u4e00-\u9fff]{2,}", w) and w not in _NAME_STOP}
    return eng, zh


def doc_mentions(question, docs):
    """问题有没有点名资料。返回 "all"（按第几个、按描述点名，要发完整清单）、命中名字的编号列表（只发这几份），或 None（没点名，不发清单）"""
    if ORDINAL_RE.search(question):
        return "all"
    q, has_doc_word = question.lower(), bool(DOC_WORD_RE.search(question))
    hits = []
    for n, (_, label) in enumerate(docs, 1):
        eng, zh = name_words(label)
        # 英文名用完整单词就够特别（NIST、RMF）；中文词太常见（创新、发展、中国），还要同时说到“文件、报告、那个”这类资料的说法
        if any(re.search(rf"(?<![a-z]){w}(?![a-z])", q) for w in eng) or (has_doc_word and any(w in question for w in zh)):
            hits.append(n)
    return hits or ("all" if DESCRIBE_RE.search(question) else None)


def _parse_pick(reply, allowed):
    """DeepSeek 的回复 → (类型, 编号列表)。支持 “DOC 2,4” “DOC 1-3” “DOC 2、4”；只认清单里列出过的编号"""
    m = re.match(r"\s*([A-Z]+)(.*)", reply.strip().upper(), re.S)
    kind, rest = (m.group(1), m.group(2)) if m else (reply.strip().upper(), "")
    nums = set()
    for a, b in re.findall(r"(\d+)\s*(?:[-–~至到]\s*(\d+))?", rest):
        nums.update(range(int(a), int(b or a) + 1))
    return kind, sorted(n for n in nums if n in allowed)


# “理解”这一步：一次调用把问题理解清楚（类型、在问什么、要找的每样信息、在哪些资料里找、怎么得出答案），
# 代替原来的“判断类型”+“改写关键词”两次调用。只在后台用，不显示给用户
PROMPT_UNDERSTAND = (
    "你是文档问答系统里负责“理解问题”的一步。读用户的问题，只输出一个 JSON 对象，格式：\n"
    '{"type": "类型", "ask": "用一句话复述用户在问什么", "find": [{"what": "要在资料里找的一样信息", "keywords": "检索关键词"}], '
    '"docs": [资料编号], "how": "怎么得出答案（一句话）"}\n'
    "type 只能是以下之一：闲聊（打招呼、感谢）；介绍（问你能做什么、你是谁、怎么用）；查事实（问资料里的某个事实、数字、定义、规定）；"
    "比较排名（谁最多、最高、最少、下降最多、排第几，要看整张表才能比）；计算（合计、相减、差额、占比、倍数、平均，要用几个数字算出来）；"
    "总结（概括一整份资料讲了什么、核心观点、要点；只问某一章、某个话题的算查事实）；对比（两样东西有什么不同、分别怎样）。\n"
    "find：答案需要的每一样信息单独列一项。计算题把算式里要用的每个数都列出来；对比题每一方列一项；查事实一般一项；闲聊、介绍、总结写空列表 []。"
    "keywords 要猜原文会怎么写：中文、英文都写，用空格分隔；英文资料写英文术语（比如“能承受多大风险”写 risk tolerance），表格写可能的行名、列名。\n"
    "ask：把“这份、第一个、那个文件”这类说法换成具体的资料名。how：计算题写算式思路，比较排名题写“看整张表的排名”，其他写一句话。\n"
    "docs：没有附资料清单时写 []。"
)
PROMPT_UNDERSTAND_DOCS = (
    "\n附：资料清单（编号是页面侧栏里的顺序）。只有问题里明确点名了其中某几份（按编号如“第一个”、按名称如“NIST 那份”、按描述如“讲物联网的那个文件”），"
    "docs 才写这几份的编号；如果回答还要用到清单里别的资料的内容（比如拿方案里的数字和某张表里的数据比），把那几份的编号也写上。"
    "没有点名就写 []，不要因为猜某份资料里可能有答案就写编号。\n资料清单：\n{docs}"
)
UNDERSTAND_TYPES = {"闲聊": "CHAT", "介绍": "INTRO", "查事实": "DOC", "对比": "DOC", "比较排名": "CALC", "计算": "CALC", "总结": "SUMMARY"}
DOC_LIST_FULL = 30   # 资料不超过这么多份时，按名字点名也附完整清单：问题可能还要用到别的资料（“是东盟出口额的几倍”）


def understand(question, docs, listed):
    """理解问题，返回 dict（type / ask / find / docs / how），出错时返回 ⚠️ 开头的提示；
    DeepSeek 没按格式回答时返回 None，由调用方退回原来的“判断类型 + 改写关键词”"""
    system = PROMPT_UNDERSTAND + (PROMPT_UNDERSTAND_DOCS.format(docs="\n".join(f"{n}. {docs[n - 1][1]}" for n in listed)) if listed else "")
    reply = ask(system, question, temperature=0, json_mode=True)
    if is_error(reply):
        return reply
    try:
        u = json.loads(reply)
        if u.get("type") not in UNDERSTAND_TYPES:
            return None
        find = [f for f in u.get("find") or [] if isinstance(f, dict) and (f.get("what") or f.get("keywords"))]
        return {"type": u["type"], "ask": str(u.get("ask") or question), "how": str(u.get("how") or ""),
                "find": [{"what": str(f.get("what") or ""), "keywords": str(f.get("keywords") or f.get("what"))} for f in find],
                "docs": sorted({int(n) for n in u.get("docs") or [] if str(n).isdigit() and int(n) in listed})}
    except (ValueError, TypeError, AttributeError):
        return None


def retrieve_targets(question, targets, chunks, bm25, vectors):
    """按“要找的信息”分别检索再合起来。只有一样时和原来一样（问题原文算向量 + 关键词）；
    有好几样时（计算题要两个数、对比题要两方），每样各检索一次，轮流取，最多 2 × TOP_K 段"""
    if len(targets) <= 1:
        kw = targets[0]["keywords"] if targets else question
        return retrieve(question, kw, chunks, bm25, vectors)
    lists = [retrieve(t["what"] or question, t["keywords"], chunks, bm25, vectors) for t in targets]
    picked = []
    for rank in range(TOP_K):
        for lst in lists:
            if rank < len(lst) and lst[rank] not in picked:
                picked.append(lst[rank])
    return picked[:2 * TOP_K]


PROMPT_CHAT = "你是「溯光」，一个友好的文档问答助手。用中文回应用户，最多两句话、不超过 50 字，不要分点列举。如果用户问你能做什么，就说：你可以上传 PDF、Word 或 Excel，我会根据文档内容回答问题并标注出处。"
PROMPT_KEYWORDS = (
    "把用户的问题改写成检索关键词，同时给出中文和英文关键词。只输出关键词本身，用空格分隔；"
    "不要输出“中文关键词”“英文关键词”这类标签，也不要任何解释或标点。"
)
PROMPT_ANSWER = (
    "你只根据给出的文档片段用中文回答问题，并在句末用 [片段N] 标注出处。片段里没有的内容，就说文档里没有提到。"
    "第一句话就直接回答问题（是或否、数字、名称），再补充说明。"
    "问题需要用片段里的数字做简单计算（相加、相减、相除、占比、倍数、平均）时，直接算出结果，写出算式，并标注每个数字的出处；"
    "片段里找不到算式需要的数字时，才说文档里没有提到。"
    "标有“程序计算”的片段是用程序对整张表算好的准确统计，涉及合计、平均、最多最少、排名时优先使用它，不要自己估算。"
)
PROMPT_MAP = (
    "下面是文档的一部分，每段前面有编号，例如 [c12]。请用中文提炼这部分最重要的 2–4 条要点，每条不超过 60 字，"
    "每条末尾用方括号标注 1–2 个最主要的依据段落编号，例如 [c12]。只依据原文，不要编造；如果这部分只是参考文献、致谢、目录之类，就只写一条“（本部分为参考文献等附属内容）”。"
)
PROMPT_REDUCE = (
    "下面是一份或几份文档按原文顺序整理出的分段要点，每条末尾的 [cN] 是原文段落编号。请据此用中文回答用户的问题："
    "先用一两句话概括全文主旨，再分条列出核心内容：用户指定了条数就严格按用户的条数；用户要求“一段话”时就写成一段，不分条；否则 3–6 条。每条末尾保留 1–2 个最相关的编号，例如 [c12]，只能使用要点里出现过的编号。"
    "忽略参考文献等附属内容，不要编造要点里没有的信息。"
)


# ---------- 调用 DeepSeek ----------
def set_api_key(key):
    """由网页传入 Key（例如云端的 st.secrets）。传入后不再用 .env 覆盖它"""
    global _key_from_app
    _key_from_app = bool(key)
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key


_key_from_app = False


def has_api_key():
    # 本地：每次都重新读一遍 .env（override=True），这样用户改完 .env 刷新页面就能生效，不用重启。
    # 云端（set_api_key 传入过 Key）：不读 .env
    if not _key_from_app:
        load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
    return bool(os.getenv("DEEPSEEK_API_KEY"))


@lru_cache(maxsize=4)
def get_client(key, max_retries=API_MAX_RETRIES):
    # 按 Key 缓存客户端：Key 换了会自动新建。max_retries=1：最多重试 1 次，最坏等待约 2×API_TIMEOUT 秒
    return OpenAI(api_key=key, base_url=API_BASE_URL, timeout=API_TIMEOUT, max_retries=max_retries)


def ask(system, user, temperature=None, retries=API_MAX_RETRIES, json_mode=False):
    """调用一次 DeepSeek。网络不通、超时、限流等情况不抛异常，而是返回一句以 ⚠️ 开头的中文提示。
    temperature 不传时用模型默认值（和原来一样）；总结模式传 0，结果更稳定、也方便缓存。
    retries：出错后自动重试几次（总结模式的分批提炼用 0，避免最坏情况下等太久）
    json_mode：要求只输出一个 JSON 对象（“理解”这一步用）"""
    extra = {} if temperature is None else {"temperature": temperature}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    try:
        resp = get_client(os.getenv("DEEPSEEK_API_KEY"), retries).chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        return resp.choices[0].message.content
    except openai.APITimeoutError:
        return f"{ERROR_PREFIX} DeepSeek 响应超时了，请稍后再试一次。"
    except openai.APIConnectionError:
        return f"{ERROR_PREFIX} 连不上 DeepSeek，请检查网络后再试。"
    except openai.RateLimitError:
        return f"{ERROR_PREFIX} 请求太频繁或账户余额不足，请稍等一会儿，或到 DeepSeek 后台看看余额。"
    except openai.AuthenticationError:
        return f"{ERROR_PREFIX} DeepSeek API Key 无效，请检查 DEEPSEEK_API_KEY（本地在 .env 里，云端在 Streamlit Secrets 里）。"
    except openai.APIError as e:
        return f"{ERROR_PREFIX} DeepSeek 暂时出错了（{type(e).__name__}），请稍后再试。"


def is_error(text):
    return isinstance(text, str) and text.startswith(ERROR_PREFIX)


# ---------- 读文档、切块 ----------
_WORD = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(s):
    """把文字切成检索用的词：中文用 rjieba（jieba 的 Rust 版）的“搜索模式”分词，英文按单词、数字按整串。
    搜索模式会同时给出长词和里面的短词，例如“书面通知”→ 书面 / 通知 / 书面通知，
    这样问题和资料就算切法略有不同也能对上；标点和空白丢掉。"""
    return [w for piece in rjieba.cut_for_search(s.lower()) for w in _WORD.findall(piece)]


def read_pages(name, data):
    """把不同格式的文件统一读成 [(位置说明, 文字)] 的列表"""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        pages = [(f"第 {i} 页", p.extract_text() or "") for i, p in enumerate(PdfReader(io.BytesIO(data)).pages, 1)]
        if OCR_ENABLED and any(len(text.strip()) < OCR_IMAGE_PAGE_CHARS for _, text in pages):
            pages = ocr_pages(data, pages, name)
        return pages
    if ext == "docx":
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for t in doc.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return [("正文", "\n".join(parts))]
    if ext == "xlsx":
        # Excel 按“行”切：每一行单独成一段，并且带上列名，例如 “订单号: SO..., 销售员: 王强, 数量: 17”
        wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        pages = []
        for ws in wb.worksheets:
            header = None
            for n, row in enumerate(ws.iter_rows(values_only=True), 1):
                vals = ["" if v is None else str(v).strip() for v in row]
                if not any(vals):
                    continue  # 跳过空行
                if header is None:
                    header = [v or f"第{j}列" for j, v in enumerate(vals, 1)]  # 第一行非空的当表头
                    continue
                text = "，".join(f"{h}: {v}" for h, v in zip(header, vals) if v)
                pages.append((f"工作表「{ws.title}」第 {n} 行", f"[{ws.title}] {text}"))
        return pages
    return []


@lru_cache(maxsize=1)
def load_ocr():
    # 第一次碰到扫描页时才加载（import 也放这里），平时不占内存
    from rapidocr import RapidOCR
    # 找文字位置时，把整页图片缩到最长边 1280 像素再找：识别结果和默认一字不差，内存峰值 2.2GB → 1.8GB
    return RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": 1280})


_progress = threading.local()   # 每个网页会话在自己的线程里录入，进度回调、认字失败的页各记各的，不会串


@contextmanager
def ocr_progress(callback):
    """在这个 with 里录入的扫描件，每认完一页调用一次 callback(文件名, 第几页, 共几页)，网页用它显示进度条"""
    _progress.callback = callback
    try:
        yield
    finally:
        _progress.callback = None


def _needs_ocr(pdf, i, text):
    """这一页要不要 OCR：字太少；或者字不多、而且一张大图盖住了半页以上（带水印的扫描页）"""
    n = len(text.strip())
    if n < OCR_MIN_CHARS:
        return True
    if n >= OCR_IMAGE_PAGE_CHARS:
        return False
    try:
        return image_ratio(pdf[i]) >= OCR_IMAGE_RATIO
    except Exception:   # 这一页 pdfium 读不了（两个库读出的页数不一致、页面损坏）：当它不是扫描页，不让整份失败
        return False


def ocr_pages(data, pages, name=""):
    """把读不出字的页渲染成图片，用 OCR 认字。只处理这些页，能直接读出字的页原样保留"""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    out = list(pages)
    callback = getattr(_progress, "callback", None)
    try:
        todo = [i for i, (_, text) in enumerate(pages) if _needs_ocr(pdf, i, text)]
        for k, i in enumerate(todo, 1):
            where = pages[i][0]
            try:
                result = load_ocr()(pdf[i].render(scale=OCR_SCALE).to_numpy())
                ocr_text = join_ocr_lines(result.txts or (), getattr(result, "boxes", None))
            except Exception as e:   # 一页认不出来（图太大、内存不够、这页坏了）只少这一页，接着认下一页
                print(f"[OCR] {where} 识别失败，跳过：{type(e).__name__}: {e}", flush=True)
                ocr_text = ""
                if getattr(_progress, "failed", None) is not None:
                    _progress.failed.append(i + 1)
            # 原来有少量真文字的页（封面、PPT 导出的“大背景图 + 几行字”）：OCR 认出的字明显更多才替换，否则保留准确的文字层
            n = len(pages[i][1].strip())
            if ocr_text.strip() and (n < OCR_MIN_CHARS or len(ocr_text.strip()) > 2 * n + 20):
                out[i] = (where + OCR_TAG, ocr_text)
            if callback:
                callback(name, k, len(todo))
    finally:
        pdf.close()
    return out


_CJK_END = re.compile(r"[\u4e00-\u9fff，、；：“（]$")
_CJK_START = re.compile(r"[\u4e00-\u9fff“（]")
_HEADING = re.compile(r"[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|第[一二三四五六七八九十百\d]+[章节条部分篇]|目录")


def join_ocr_lines(lines, boxes=None):
    """OCR 按“行”返回，中文一句话常被行尾截断（“经济增长4.5%—5%，在实际工作 / 中努力”），要接上，分词才不会切错。
    只接“写满了”的行：没写满的是标题或段落结尾，保留换行；下一行是“一、”“（一）”“第一章”“目录”这类标题时也不接。
    boxes 是 OCR 给的每行四个角的坐标，没有时按都写满了处理"""
    if not lines:
        return ""
    full = at_left = [True] * len(lines)
    if boxes is not None and len(boxes) == len(lines):
        full, at_left = _line_layout([min(p[0] for p in b) for b in boxes], [max(p[0] for p in b) for b in boxes])
    out = lines[0]
    for i in range(1, len(lines)):
        # 上一行写满了、这一行又从左边开始，才是一句话被行尾截断
        join = full[i - 1] and at_left[i] and _CJK_END.search(lines[i - 1]) and _CJK_START.match(lines[i]) and not _HEADING.match(lines[i])
        out += ("" if join else "\n") + lines[i]
    return out


def _cluster_edge(xs, tol, from_right):
    """好几行都停在附近的位置（至少 2 行、或两成的行）：从右往左（或从左往右）找第一个。找不到时用最外面的那个"""
    need = max(2, 0.2 * len(xs))
    order = sorted(xs, reverse=from_right)
    return next((x for x in order if sum(abs(y - x) <= tol for y in xs) >= need), order[0])


def _line_layout(lefts, rights):
    """每一行是不是“写满了”、是不是从这一栏的左边开始。左右边界都取“好几行都停在附近”的位置，不取最外面那一行：
    右上角的日期、顶格的页眉只有自己一行在那儿——原来拿最宽的那一行当标准，正文一行都接不上，日期反而和下一行粘在一起。
    “写满了”还要从左边开始（首行缩进几个字可以）：靠右对齐的署名、居中的标题右边也可能到边界，但不是被截断的正文。
    左边一组的行本身到不了页面中间时，才按两栏排版、左右栏各算各的；一栏不到 3 行不当正文接"""
    n, lo = len(lefts), min(lefts)
    width = (max(rights) - lo) or 1
    tol = 0.03 * width
    column = [0] * n
    right_side = [i for i in range(n) if lefts[i] - lo > 0.4 * width]
    left_side = [i for i in range(n) if i not in right_side]
    if len(right_side) >= 3 and left_side and np.median([rights[i] for i in left_side]) < lo + 0.6 * width:
        column = [int(i in right_side) for i in range(n)]
    full, at_left = [False] * n, [False] * n
    for col in set(column):
        idx = [i for i in range(n) if column[i] == col]
        if len(idx) < 3:
            continue
        edge = _cluster_edge([rights[i] for i in idx], tol, from_right=True)
        start = _cluster_edge([lefts[i] for i in idx], tol, from_right=False)
        for i in idx:
            at_left[i] = abs(lefts[i] - start) <= 0.15 * width
            full[i] = at_left[i] and abs(rights[i] - edge) <= tol
    return full, at_left


def image_ratio(page):
    """这一页被图片盖住的比例（0–1）。扫描件每页基本就是一张大图"""
    import pypdfium2.raw as pdfium_c
    w, h = page.get_size()
    area = 0
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
        left, bottom, right, top = obj.get_bounds()
        area += max(0, min(right, w) - max(left, 0)) * max(0, min(top, h) - max(bottom, 0))
    return min(1, area / (w * h)) if w and h else 0


def split_text(text):
    """按固定长度切块，相邻两段有重叠，避免一句话刚好被切断就找不到"""
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[start:start + CHUNK_SIZE] for start in range(0, len(text), step)]


# ---------- 向量 ----------
def set_embed_model(name):
    """换用另一个嵌入模型（app.py 在线体验版用它换成小模型）。要在第一次算向量之前调用"""
    global EMBED_MODEL
    EMBED_MODEL = name


def embed_model_size():
    return EMBED_MODELS.get(EMBED_MODEL, ("", "", "几百 MB"))[2]


def embedder_cached():
    """嵌入模型是否已经下载过（没下载过时，第一次加载要联网下载，大小见 embed_model_size()）"""
    import glob
    hits = glob.glob(os.path.join(EMBED_CACHE_DIR, "**", "*.onnx"), recursive=True)
    return any(EMBED_MODEL.split("/")[-1].lower() in h.lower() for h in hits)


@lru_cache(maxsize=2)
def _load_embedder(model):
    return TextEmbedding(model, cache_dir=EMBED_CACHE_DIR)


def load_embedder():
    # 本地的多语言嵌入模型：把一段文字变成一串数字（向量），意思相近的文字向量也相近
    return _load_embedder(EMBED_MODEL)


def _embed(texts, prefix):
    v = np.array(list(load_embedder().embed([prefix + t for t in texts], batch_size=EMBED_BATCH_SIZE)))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed(texts):
    """给资料段落算向量"""
    return _embed(texts, EMBED_MODELS.get(EMBED_MODEL, ("", ""))[1])


def embed_query(question):
    """给问题算向量（有的模型要求问题和资料加不同的前缀）"""
    return _embed([question], EMBED_MODELS.get(EMBED_MODEL, ("", ""))[0])[0]


def file_chunks(name, data):
    """把一个文件读出来并切成段落，返回 [{file, page, text}, ...]。一个字都没读到（多半是扫描件）时返回空列表"""
    pages = read_pages(name, data)
    if not any(text.strip() for _, text in pages):
        return []
    chunks = []
    for where, text in pages:
        text = re.sub(r"[ \t]+", " ", text)
        for piece in split_text(text):
            chunks.append({"file": name, "page": where, "text": piece})
    return chunks


def build_index(files, failed=None):
    """切块 + 建关键词索引 + 算向量。files 是 ((文件名, 字节), ...)。返回 (片段列表, BM25, 向量, 没读到文字的文件名列表)。
    读取出错的文件跳过，不影响其他文件；传入 failed 列表时，把 (文件名, 原因) 记在里面"""
    chunks, empty_files = [], []
    for name, data in files:
        try:
            pieces = file_chunks(name, data)
        except Exception as e:
            if failed is not None:
                failed.append((name, "没有录入：" + explain_error(e)))
            continue
        if not pieces:
            empty_files.append(name)  # 一个字都没读到，多半是扫描件（图片），跳过
            continue
        chunks.extend(pieces)
    if not chunks:
        return [], None, None, empty_files
    # BM25Plus 是 BM25 的改进版，分数不会变成负数（文档段数很少时原版 BM25 会出现负分）
    bm25 = BM25Plus([tokenize(c["text"]) for c in chunks])
    vectors = embed([c["text"] for c in chunks])  # 每段文字提前算好向量，存起来
    return chunks, bm25, vectors, empty_files


# ---------- 资料库：把切好的段落和向量存到硬盘，重启后不用重新上传 ----------
class FileStore:
    """资料库的「存 / 读」层（方案 A：一个文件夹 + json/npy 文件）。
    每份资料用内容哈希（sha256，内容一样哈希就一样）当编号，存三样东西：
      <编号>.<扩展名>  原文件（Excel 统计、以后换模型重算向量、OCR 都要用）
      <编号>.json      切好的段落
      <编号>.npy       这些段落的向量
    manifest.json 是目录，记录有哪些资料。它总是最后写，所以录入到一半程序崩了，这份资料只是“没录入”，不会留下半截数据。
    以后要换成向量数据库（LanceDB / Chroma），写一个方法相同（docs / has / add / load / original / update_vectors / remove）的类替换它即可。"""

    def __init__(self, root=LIBRARY_DIR):
        self.root = root

    def _path(self, name):
        return os.path.join(self.root, name)

    def _write(self, name, write):
        """先写到临时文件，写完再改名替换：改名是一瞬间完成的，不会出现写了一半的文件"""
        os.makedirs(self.root, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                write(f)
            os.replace(tmp, self._path(name))
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _write_json(self, name, obj):
        self._write(name, lambda f: f.write(json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")))

    def docs(self):
        """资料目录：[{id, name, ext, n_chunks, embed_model, added}, ...]，按录入先后排列。
        目录文件坏了（断电、磁盘满、手动改错）时，改名备份成 manifest.json.bad-时间，再抛出 LibraryBroken 说明情况；
        之后资料库从空的开始，不会每次都报错"""
        try:
            with open(self._path("manifest.json"), encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except ValueError:   # JSON 格式坏了（JSONDecodeError 是 ValueError 的一种）
            backup = f"manifest.json.bad-{datetime.datetime.now():%Y%m%d-%H%M%S}"
            try:
                os.replace(self._path("manifest.json"), self._path(backup))
            except FileNotFoundError:   # 另一个页面刚刚已经备份过了
                return []
            raise LibraryBroken(f"资料库目录（manifest.json）损坏了，已备份为 library/{backup}。资料库先从空的开始，请重新上传资料。")

    def has(self, doc_id):
        return any(d["id"] == doc_id for d in self.docs())

    def add(self, doc_id, name, data, chunks, vectors, embed_model, extra=None):
        """extra：要一起记进目录的其他字段（比如 OCR 没认出来的页）"""
        ext = name.lower().rsplit(".", 1)[-1]
        self._write(f"{doc_id}.{ext}", lambda f: f.write(data))
        self._write_json(f"{doc_id}.json", chunks)
        if chunks:
            self._write(f"{doc_id}.npy", lambda f: np.save(f, vectors))
        doc = {"id": doc_id, "name": name, "ext": ext, "n_chunks": len(chunks), "embed_model": embed_model,
               "added": datetime.datetime.now().isoformat(timespec="seconds"), "ocr": OCR_ENABLED, "pipeline": PIPELINE_VERSION, **(extra or {})}
        self._write_json("manifest.json", self.docs() + [doc])   # 最后才登记到目录

    @staticmethod
    def _data_files(doc):
        """这份资料的段落、向量文件名。重新处理过的资料带版本标记（<编号>.v时间.json），见 replace()"""
        stem = f"{doc['id']}.{doc['data']}" if doc.get("data") else doc["id"]
        return f"{stem}.json", f"{stem}.npy"

    def load(self, doc_id):
        """返回 (段落列表, 向量)；没有文字的资料返回 ([], None)"""
        json_name, npy_name = self._data_files(next(d for d in self.docs() if d["id"] == doc_id))
        with open(self._path(json_name), encoding="utf-8") as f:
            chunks = json.load(f)
        return chunks, (np.load(self._path(npy_name)) if chunks else None)

    def original(self, doc_id):
        doc = next(d for d in self.docs() if d["id"] == doc_id)
        with open(self._path(f"{doc_id}.{doc['ext']}"), "rb") as f:
            return f.read()

    def remove(self, doc_id):
        """删除一份资料：先从目录里划掉，再删它的文件。删到一半出错，最多在硬盘上留下几个没人用的文件，资料库本身不会乱。
        返回是否真的删了（编号不存在时返回 False）"""
        docs = self.docs()
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if doc is None:
            return False
        self._write_json("manifest.json", [d for d in docs if d["id"] != doc_id])
        for name in (f"{doc_id}.{doc['ext']}", *self._data_files(doc)):
            try:
                os.remove(self._path(name))
            except FileNotFoundError:   # 没有文字的资料没有 .npy
                pass
        return True

    def replace(self, doc_id, name, data, chunks, vectors, embed_model, extra=None):
        """用重新处理的结果替换一份已有的资料（比如旧扫描件用 OCR 重录），录入时间和在列表里的位置不变。
        新的段落、向量写成带版本标记的新文件，最后一次性改目录指向它们，再删旧文件：
        哪一步失败，目录指着的都还是一组配套的旧文件（原来直接覆盖，段落写了新的、向量还是旧的，后面所有资料的引用都会错位）。
        原文件内容没变（编号就是内容哈希），不用再写"""
        old = next(d for d in self.docs() if d["id"] == doc_id)
        new = {k: v for k, v in old.items() if k not in ("ocr_failed_pages", "ocr_retries")}   # 这两项以这次的结果为准
        new.update(name=name, n_chunks=len(chunks), embed_model=embed_model, ocr=OCR_ENABLED, pipeline=PIPELINE_VERSION,
                   data=f"v{datetime.datetime.now():%Y%m%d%H%M%S%f}", **(extra or {}))
        json_name, npy_name = self._data_files(new)
        self._write_json(json_name, chunks)
        if chunks:
            self._write(npy_name, lambda f: np.save(f, vectors))
        self._write_json("manifest.json", [new if d["id"] == doc_id else d for d in self.docs()])
        for f in self._data_files(old):   # 删旧文件失败也不要紧，只是留下没人用的文件
            try:
                os.remove(self._path(f))
            except OSError:
                pass

    def update_vectors(self, doc_id, vectors, embed_model):
        """换了嵌入模型后，用新向量替换旧向量（段数不变，只换向量）"""
        self._write(self._data_files(next(d for d in self.docs() if d["id"] == doc_id))[1], lambda f: np.save(f, vectors))
        self._write_json("manifest.json", [dict(d, embed_model=embed_model) if d["id"] == doc_id else d for d in self.docs()])


class LibraryBroken(Exception):
    """资料库目录坏了，消息里是给用户看的中文说明"""


_library_lock = threading.Lock()   # 同时打开好几个网页时，防止两边同时写目录


def explain_error(e):
    """把读文件时的常见报错翻成给用户看的中文，并说清楚该怎么办"""
    name = type(e).__name__
    if name == "FileNotDecryptedError":
        return "这份 PDF 设了打开密码，请先去掉密码再上传"
    if name in ("PdfStreamError", "PdfReadError", "EmptyFileError"):
        return "PDF 文件损坏或不完整，请重新下载，或用原来的软件另存一份再上传"
    if name == "BadZipFile" or (isinstance(e, KeyError) and "archive" in str(e)):
        return "文件损坏，或者不是真正的 Word / Excel 文件（比如只改了扩展名），请用 Office 或 WPS 另存为 .docx / .xlsx 再上传"
    if isinstance(e, MemoryError):
        return "文件太大，内存不够，可以拆成几份再上传"
    return f"读取失败（{name}: {e}）"


def library_add(files, store=None):
    """把文件录入资料库。files 是 ((文件名, 字节), ...)。按内容哈希判断：录入过的直接跳过，只处理新文件。
    返回 (这次新录入的文件名列表, 读取失败的 [(文件名, 原因)])。一个文件坏了只跳过它，不影响其他文件。"""
    store = store or FileStore()
    added, failed = [], []
    for name, data in files:
        state, why = _ingest(store, hashlib.sha256(data).hexdigest(), name, data)
        if state == "ok":
            added.append(name)
        if why:   # 失败的原因，或者录进去了但有几页没认出来
            failed.append((name, why))
    return added, failed


def library_upgrade(store=None, skip=()):
    """把用旧规则（PIPELINE_VERSION 更小、或者当时没开 OCR 的 PDF）处理的资料，从存好的原文件重新处理一遍。
    编号、名字、录入时间都不变；段落和原来一样时沿用原来的向量，不重算。skip 是这次不处理的编号（之前失败过的）。
    返回 (处理过的文件名列表, 失败的 [(编号, 文件名, 原因)])"""
    store = store or FileStore()
    done, failed = [], []
    for doc in [d for d in store.docs() if library_outdated(d) and d["id"] not in skip]:
        try:
            data = store.original(doc["id"])
        except OSError as e:
            failed.append((doc["id"], doc["name"], f"没有更新：资料库里的原文件读不到，可以把它删掉再重新上传（{type(e).__name__}: {e}）"))
            continue
        state, why = _ingest(store, doc["id"], doc["name"], data)
        if state == "ok":
            done.append(doc["name"])
        if why:
            failed.append((doc["id"], doc["name"], why))
    return done, failed


def library_outdated(doc):
    """这份资料是不是用旧规则处理的，需要重新处理"""
    return (doc.get("pipeline", 1) < PIPELINE_VERSION or (OCR_ENABLED and doc["ext"] == "pdf" and not doc.get("ocr"))
            or (OCR_ENABLED and bool(doc.get("ocr_failed_pages")) and doc.get("ocr_retries", 0) < 1))   # 有页没认出来：再试一次


def _needs_ingest(store, doc_id):
    """要不要录入：没录过的要；录过、但是用旧规则处理的也要"""
    old = next((d for d in store.docs() if d["id"] == doc_id), None)
    return old is None or library_outdated(old)


def _ingest(store, doc_id, name, data):
    """录入或重新处理一份资料。返回 (状态, 说明)：状态是 "skip"（不需要处理）、"ok"、"fail"；
    "ok" 时说明不为空，表示录进去了但有提醒（比如有几页没认出来）"""
    if not _needs_ingest(store, doc_id):
        return "skip", ""
    old = next((d for d in store.docs() if d["id"] == doc_id), None)
    # 切段（扫描件还要 OCR）和算向量很慢，放在锁外面做：录入大文件时，别的页面照样能载入资料库
    _progress.failed = []
    try:
        chunks = file_chunks(name, data)
    except Exception as e:
        return "fail", "没有录入：" + explain_error(e)
    finally:
        failed_pages, _progress.failed = _progress.failed, None
    try:
        vectors = _reuse_vectors(store, doc_id, chunks)
        if vectors is None and chunks:
            vectors = embed([c["text"] for c in chunks])
    except Exception as e:   # 不是文件的问题，多半是第一次用时嵌入模型没下载下来
        return "fail", f"没有录入：嵌入模型加载失败，文件本身没问题。第一次使用要联网下载模型，请检查网络后刷新页面（{type(e).__name__}: {e}）"
    # OCR 没认出来的页记进目录，下次打开时再试一次（只再试一次，免得每次打开都卡在同一页）
    retries = (old.get("ocr_retries", 0) + 1) if old and old.get("ocr_failed_pages") else 0
    extra = {"ocr_failed_pages": failed_pages, "ocr_retries": retries} if failed_pages else {}
    try:
        with _library_lock:   # 锁里只做“写进资料库”这一小步
            if not _needs_ingest(store, doc_id):   # 别的页面刚好也在处理同一个文件，已经好了
                return "skip", ""
            if store.has(doc_id):   # 旧规则处理的资料：原地替换，编号、录入时间不变
                store.replace(doc_id, name, data, chunks, vectors, EMBED_MODEL, extra)
            else:
                store.add(doc_id, name, data, chunks, vectors, EMBED_MODEL, extra)
    except OSError as e:   # 磁盘满、没有写权限……旧资料还在，只是这份没录进去
        return "fail", f"没有录入：保存到资料库失败，可能是磁盘满了或没有写入权限（{type(e).__name__}: {e}）"
    if failed_pages:
        pages = "、".join(map(str, failed_pages))
        later = "下次打开溯光时会再试一次" if retries < 1 else "已经重试过，还是没认出来，可以换一份更清楚的扫描件"
        return "ok", f"第 {pages} 页没认出来，其他页已录入；{later}"
    return "ok", ""


def _reuse_vectors(store, doc_id, chunks):
    """重新处理后段落和原来一模一样（文字版 PDF、Word、Excel 基本都是这样），就沿用原来的向量，省掉最慢的一步"""
    old = next((d for d in store.docs() if d["id"] == doc_id), None)
    if not old or not old["n_chunks"] or old["embed_model"] != EMBED_MODEL:
        return None
    try:
        old_chunks, old_vectors = store.load(doc_id)
    except (OSError, ValueError):
        return None
    same = [(c["page"], c["text"]) for c in old_chunks] == [(c["page"], c["text"]) for c in chunks]
    return old_vectors if same else None


def library_remove(doc_id, store=None):
    """从资料库删除一份资料（按编号，也就是内容哈希）。返回是否真的删了"""
    store = store or FileStore()
    with _library_lock:
        return store.remove(doc_id)


def library_load(store=None, problems=None):
    """载入整个资料库，返回 (片段列表, BM25, 向量, 没读到文字的文件名列表, Excel 统计)。
    前四项和 build_index() 的返回值一样，可以直接交给 answer()。
    某份资料的文件缺了或坏了，只跳过这一份；传入 problems 列表时，把 (文件名, 原因) 记在里面"""
    store = store or FileStore()
    chunks, parts, empty_files, excel_files = [], [], [], []
    with _library_lock:
        for doc in store.docs():
            if doc["n_chunks"] == 0:
                empty_files.append(doc["name"])
                continue
            try:
                pieces, vectors = store.load(doc["id"])
                # 段数、向量行数、目录里记的段数必须一致：对不上的话，这份和后面所有资料的引用都会错位
                if not len(pieces) == len(vectors) == doc["n_chunks"]:
                    raise ValueError(f"段落 {len(pieces)} 段、向量 {len(vectors)} 行、目录记的 {doc['n_chunks']} 段，对不上")
            except (OSError, ValueError) as e:   # 文件被删了、写坏了
                if problems is not None:
                    problems.append((doc["name"], f"{type(e).__name__}: {e}"))
                continue
            if doc["embed_model"] != EMBED_MODEL:
                # 存的向量是旧模型算的，和新模型的向量不能混在一起比较，必须重算
                vectors = embed([c["text"] for c in pieces])
                store.update_vectors(doc["id"], vectors, EMBED_MODEL)
            for c in pieces:
                c["doc"] = doc["id"]   # 记下每段属于哪份资料（按编号，不按文件名，同名文件也分得开），“只在某本书里找”要用
            chunks.extend(pieces)
            parts.append(vectors)
            if doc["ext"] == "xlsx":
                excel_files.append((doc["id"], doc["name"], store.original(doc["id"])))
    # Excel 统计按文件分别算（excel_stats 本来就是逐个文件算的，结果一样），这样每条统计也能标上属于哪份资料
    stats = [dict(row, doc=doc_id) for doc_id, name, data in excel_files for row in excel_stats([(name, data)])]
    if not chunks:
        return [], None, None, empty_files, stats
    # 关键词索引（BM25）不存盘，每次载入时现场重建，几万段也只要一两秒
    bm25 = BM25Plus([tokenize(c["text"]) for c in chunks])
    return chunks, bm25, np.vstack(parts), empty_files, stats



def library_subset(chunks, vectors, stats, doc_ids):
    """“只在这几份资料里找”：把选中资料的段落、向量、Excel 统计挑出来，再给它们单独建一个关键词索引。
    返回 (片段列表, BM25, 向量, Excel 统计)，可以直接交给 answer()；retrieve() / answer() 都不用改。
    doc_ids 为空时表示不限范围，原样返回全部（BM25 返回 None，由调用方用全量的）。"""
    keep = set(doc_ids or ())
    if not keep:
        return chunks, None, vectors, stats
    idx = [i for i, c in enumerate(chunks) if c.get("doc") in keep]
    sub_chunks = [chunks[i] for i in idx]
    sub_stats = [s for s in stats if s.get("doc") in keep]
    if not sub_chunks:
        return [], None, None, sub_stats
    # 关键词索引要在小书架上重建：BM25 会看“一个词在多少段里出现过”，范围变了这个数也跟着变
    return sub_chunks, BM25Plus([tokenize(c["text"]) for c in sub_chunks]), vectors[idx], sub_stats

# ---------- Excel 统计 ----------
def num(x):
    """数字显示得好看一点：整数不带小数，其他最多保留两位，末尾的 0 去掉（39.8 不写成 39.80，和原表一致）"""
    x = float(x)
    return f"{int(x)}" if x.is_integer() else f"{x:.2f}".rstrip("0")


def describe_table(title, df):
    """用 pandas 把一张表算成一段统计文字：每个数字列的合计/平均/最小/最大，以及按类别分组的合计。
    有一列能当名字（每行都不一样，比如 国家和地区、指标、订单号）时，最大最小写明是哪一行；表不大时再附从大到小的排名"""
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    label = next((c for c in df.columns if c not in nums and df[c].notna().all() and df[c].nunique() == len(df)), None)
    # 只按“重复较多”的列分组（比如 产品、区域、销售员），日期、订单号这种几乎每行都不同的列分组没意义
    cats = [c for c in df.columns if c not in nums and 1 < df[c].nunique() <= min(MAX_GROUP_VALUES, len(df) / 2)]
    lines = [f"{title}：共 {len(df)} 行数据"]
    for c in nums:
        s = df[c].dropna()
        if s.empty:
            continue
        line = f"「{c}」合计 {num(s.sum())}，平均 {num(s.mean())}，最小 {num(s.min())}，最大 {num(s.max())}"
        if label:   # 原来只有数字，看不出最大的是哪一行（“进口额最多的是谁”就答不出来）
            line += f"（最小是「{df.loc[s.idxmin(), label]}」，最大是「{df.loc[s.idxmax(), label]}」）"
            if len(s) <= TABLE_RANK_ROWS:
                line += "；从大到小：" + "、".join(f"{df.loc[i, label]} {num(v)}" for i, v in s.sort_values(ascending=False).items())
        lines.append(line)
    for g in cats:
        counts = df.groupby(g).size()
        sums = df.groupby(g)[nums].sum() if nums else None
        for key, cnt in counts.items():
            extra = "".join(f"，「{c}」合计 {num(sums.loc[key, c])}" for c in nums) if nums else ""
            lines.append(f"按「{g}」分组 = {key}：{cnt} 行{extra}")
    return "\n".join(lines)


def excel_stats(files):
    """给 Excel 算好统计数字（只用 pandas 计算，不执行 AI 写的代码），回答“合计/最多/最少/平均/排名”这类问题用"""
    out = []
    for name, data in files:
        if not name.lower().endswith(".xlsx"):
            continue
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        except Exception:   # 坏的 Excel 在切段时已经报过错了，统计这里直接跳过
            continue
        same_columns = {}  # 列名完全一样的工作表（比如 1月、2月、3月明细），再合在一起算一遍
        for title, df in sheets.items():
            df = df.dropna(how="all")
            if df.empty:
                continue
            out.append({"file": name, "page": f"工作表「{title}」统计（程序计算）", "text": describe_table(f"工作表「{title}」", df)})
            same_columns.setdefault(tuple(df.columns), []).append((title, df))
        for group in same_columns.values():
            if len(group) > 1:
                titles = "、".join(f"「{t}」" for t, _ in group)
                merged = pd.concat([df for _, df in group])
                out.append({"file": name, "page": f"工作表{titles}合并统计（程序计算）", "text": describe_table(f"工作表{titles}合并", merged)})
    return out


# ---------- 混合检索 ----------
def retrieve(question, keywords, chunks, bm25, vectors):
    """混合检索，返回要交给 DeepSeek 的片段编号"""
    if len(chunks) <= TOP_K:
        return list(range(len(chunks)))  # 文档很短：全部给模型，不用挑
    # 关键词检索：按字面匹配打分
    kw_scores = bm25.get_scores(tokenize(keywords))
    kw_rank = sorted(range(len(chunks)), key=lambda i: kw_scores[i], reverse=True)
    # 向量检索：按意思相似度打分
    vec_scores = vectors @ embed_query(question)
    vec_rank = sorted(range(len(chunks)), key=lambda i: vec_scores[i], reverse=True)
    # 先保底：两种检索各自的前 GUARANTEED_TOP 名一定入选（防止某一边排第一的段落被融合挤掉）
    picked = []
    for i in kw_rank[:GUARANTEED_TOP] + vec_rank[:GUARANTEED_TOP]:
        if i not in picked:
            picked.append(i)
    # 再用 RRF 融合排序补满 TOP_K 段：两边排名都靠前的段落得分最高
    fused = {}
    for rank_list in (kw_rank[:RRF_POOL], vec_rank[:RRF_POOL]):
        for r, i in enumerate(rank_list):
            fused[i] = fused.get(i, 0) + 1 / (RRF_K + r)
    for i in sorted(fused, key=fused.get, reverse=True):
        if len(picked) >= TOP_K:
            break
        if i not in picked:
            picked.append(i)
    return picked


# ---------- 总结模式：先分批提炼要点（map），再合并成一个回答（reduce） ----------
_summary_cache = OrderedDict()   # 批次内容的哈希 → 这一批的要点。同一份文档再问总结时不用重新提炼
_summary_lock = threading.Lock()


def _spread(items, k):
    """从 items 里均匀挑 k 个，一定包含第一个和最后一个（开头、中间、结尾都看得到）"""
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[0]]
    return [items[round(j * (len(items) - 1) / (k - 1))] for j in range(k)]


def summary_batches(chunks):
    """按文档分组、按原文顺序把段落装进一批批（每批不超过 SUMMARY_BATCH_CHARS 字）。
    总批数超过 SUMMARY_MAX_BATCHES 时：先保证每份文档至少 1 批，剩下的名额按批数比例分给各文档，
    每份文档在自己的批次里均匀挑（包含第一批和最后一批）。"""
    by_file = OrderedDict()
    for i, c in enumerate(chunks):
        by_file.setdefault(c["file"], []).append(i)
    per_file = []
    for ids in by_file.values():
        groups, cur, size = [], [], 0
        for i in ids:
            n = len(chunks[i]["text"])
            if cur and size + n > SUMMARY_BATCH_CHARS:
                groups.append(cur)
                cur, size = [], 0
            cur.append(i)
            size += n
        if cur:
            groups.append(cur)
        per_file.append(groups)
    total = sum(len(g) for g in per_file)
    if total <= SUMMARY_MAX_BATCHES:
        return [b for g in per_file for b in g]
    per_file = per_file[:SUMMARY_MAX_BATCHES]          # 文件数比批数还多时，只看前 SUMMARY_MAX_BATCHES 份
    quota = [1] * len(per_file)
    spare = SUMMARY_MAX_BATCHES - len(per_file)
    extra = [len(g) - 1 for g in per_file]
    while spare > 0 and any(e > 0 for e in extra):      # 按“还差多少批”的比例，一批一批地分名额
        j = max(range(len(per_file)), key=lambda t: extra[t] / quota[t])
        quota[j] += 1
        extra[j] -= 1
        spare -= 1
    return [b for g, q in zip(per_file, quota) for b in _spread(g, q)]


def _map_one(batch, pool):
    text = "\n\n".join(f"[c{i}]（{pool[i]['file']} {pool[i]['page']}）{pool[i]['text']}" for i in batch)
    key = hashlib.sha1((MODEL + PROMPT_MAP + text).encode("utf-8")).hexdigest()
    with _summary_lock:
        if key in _summary_cache:
            _summary_cache.move_to_end(key)
            return _summary_cache[key], True
    out = ask(PROMPT_MAP, text, temperature=0, retries=0)
    if not is_error(out):
        with _summary_lock:
            _summary_cache[key] = out
            while len(_summary_cache) > SUMMARY_CACHE_SIZE:
                _summary_cache.popitem(last=False)
    return out, False


# 一处引用标注：可以是 [c12]、[c3, c5]，也可以是连着写的 [c1][c2][c3]（连着的算同一处）
CMARK = re.compile(r"(?:\[\s*c\d+(?:\s*[,，、;；]\s*c?\d+)*\s*\][ \t]*)+", re.I)
# 模型偶尔写成 [c12-c15]、【c12】、(c12) 等其他格式：替换完后兜底删掉
CMARK_LEFT = re.compile(r"[\[【(（]\s*c\d+[^\]】)）\n]{0,30}[\]】)）]", re.I)


def summarize(question, chunks, trace, stats=()):
    """总结整份文档，返回 (回答, 引用片段)。回答里的 [cN] 会换成 [片段N]，并把这些段落作为引用返回。
    stats 是 Excel 的 pandas 统计片段：会直接附在合并那一步，让表格里的数字准确"""
    stats = list(stats)
    kept, size = [], 0
    for h in stats:                       # 统计太长时只带前面一部分
        if kept and size + len(h["text"]) > SUMMARY_STATS_CHARS:
            break
        kept.append(h)
        size += len(h["text"])
    pool = list(chunks) + kept             # 编号 0..len(chunks)-1 是原文段落，后面是统计
    stat_ids = list(range(len(chunks), len(pool)))
    batches = summary_batches(chunks)
    trace["batches"] = len(batches)
    if len(batches) == 1:  # 文档很短：一次就能放下，不必分批，直接按普通方式把全部段落交给模型
        hits = kept + [chunks[i] for i in batches[0]]
        context = "\n\n".join(f"[片段{n}]（{h['file']} {h['page']}）{h['text']}" for n, h in enumerate(hits, 1))
        reply = ask(PROMPT_ANSWER, f"文档片段：\n{context}\n\n问题：{question}", temperature=0)
        return (reply, []) if is_error(reply) else (reply, hits)
    with ThreadPoolExecutor(max_workers=min(SUMMARY_WORKERS, len(batches))) as pool_exec:
        results = list(pool_exec.map(lambda b: _map_one(b, pool), batches))
    trace["cache_hits"] = sum(1 for _, cached in results if cached)
    notes = [(b, out) for b, (out, _) in zip(batches, results) if not is_error(out)]
    trace["map_errors"] = len(batches) - len(notes)
    if not notes:
        return results[0][0], []
    head = lambda b: pool[b[0]]
    parts = "\n\n".join(f"【{head(b)['file']} {head(b)['page']} 起】\n{out}" for b, out in notes)
    if kept:
        parts = "【程序计算的表格统计（准确数字，涉及合计、平均、最多最少时优先用它）】\n" + "\n\n".join(
            f"[c{i}]（{pool[i]['file']} {pool[i]['page']}）{pool[i]['text']}" for i in stat_ids) + "\n\n" + parts
    reply = ask(PROMPT_REDUCE, f"分段要点：\n{parts}\n\n问题：{question}", temperature=0)
    if is_error(reply):
        return reply, []
    # 把 [c12] 这类编号换成页面认识的 [片段N]；只保留真实存在、且出现在某一批（或统计）里的编号
    valid = {i for b, _ in notes for i in b} | set(stat_ids)
    picked = lambda m: [int(d) for d in re.findall(r"\d+", m.group(0)) if int(d) in valid][:SUMMARY_CITES_PER_MARK]
    order = []
    for m in CMARK.finditer(reply):
        for i in picked(m):
            if i not in order and len(order) < SUMMARY_MAX_CITES:
                order.append(i)
    num_of = {i: n for n, i in enumerate(order, 1)}

    def to_cite(m):
        ns = [num_of[i] for i in picked(m) if i in num_of]
        return "".join(f"[片段{n}]" for n in dict.fromkeys(ns))

    reply = CMARK_LEFT.sub("", CMARK.sub(to_cite, reply))
    reply = re.sub(r"[ \t]+(?=\n|$)", "", reply)
    if trace["map_errors"]:
        reply += f"\n\n（注：有 {trace['map_errors']} 部分内容这次没能读取，总结可能不完整，可以稍后再问一次。）"
    return reply, [pool[i] for i in order]


# ---------- 意图判断 + 回答（网页和评测共用这一个入口） ----------
def answer(question, chunks, bm25, vectors, stats=(), trace=None, no_text_reply=NO_TEXT_REPLY, docs=None, narrow=None):
    """回答一个问题，返回 (回答文字, 引用的片段列表)。
    trace 如果传入一个 dict，会把中间结果写进去（kind 意图、keywords 检索关键词、error 是否调用模型出错、
    docs 问题点名的资料编号），方便评测和调试。
    no_text_reply：没有任何可检索文字时的回复。
    docs / narrow：资料库模式用。docs 是全部资料 [(编号, 显示名), ...]，顺序和侧栏一样；问题点名了其中几份时，
    用 narrow(编号列表) 换成只含这几份的 (chunks, bm25, vectors, stats)，这一问只在里面找。"""
    trace = {} if trace is None else trace
    trace.update(kind=None, intent=None, understanding=None, keywords=None, error=False, docs=[])
    hits = []
    # 有好几份资料、问题里又真的点名了时，才附上资料清单（资料不多时附完整清单，多时只附名字命中的那几份）
    mention = doc_mentions(question, docs) if docs and narrow and len(docs) > 1 else None
    full = mention == "all" or (mention and len(docs) <= DOC_LIST_FULL)
    listed = list(range(1, len(docs) + 1)) if full else (mention or [])
    # 先理解问题：类型、在问什么、要找的每样信息、在哪些资料里找
    u = understand(question, docs, listed)
    if is_error(u):
        trace.update(kind="ERROR", error=True)
        return u, []
    if u:
        kind, nums = UNDERSTAND_TYPES[u["type"]], u["docs"]
        trace.update(intent=u["type"], understanding=u)
    else:   # DeepSeek 没按格式回答：退回原来的“判断类型”
        system = PROMPT_INTENT + (PROMPT_PICK_DOCS.format(docs="\n".join(f"{n}. {docs[n - 1][1]}" for n in listed)) if listed else "")
        reply = ask(system, question, temperature=0)   # 分类用 temperature 0，同一个问题每次分得一样
        if is_error(reply):
            trace.update(kind="ERROR", error=True)
            return reply, []
        kind, nums = _parse_pick(reply, set(listed))
    if nums and kind not in ("INTRO", "CHAT"):
        trace["docs"] = [docs[n - 1][0] for n in nums]
        chunks, bm25, vectors, stats = narrow(trace["docs"])   # 这一问只看点名的资料；侧栏的提问范围不受影响
    trace["kind"] = kind
    if "INTRO" in kind:
        return INTRO, hits
    if "CHAT" in kind:
        reply = ask(PROMPT_CHAT, question)
        trace["error"] = is_error(reply)
        return reply, hits
    if not chunks:
        if trace["docs"]:   # 点名的那份没读到字：说清楚是哪份，不要说成“没有文档”
            names = "、".join(f"《{label}》" for i, label in docs if i in trace["docs"])
            return f"{names}里没有读到文字（可能是扫描件，OCR 也没认出来）。可以换一份更清楚的文件，或者在侧栏把它删掉重新上传。", hits
        return no_text_reply, hits
    if "SUMMARY" in kind:
        reply, hits = summarize(question, chunks, trace, stats)
        trace["error"] = is_error(reply)
        return reply, hits
    if u:   # 按理解出的“要找的信息”分别检索
        targets = u["find"] or [{"what": question, "keywords": question}]
        trace["keywords"] = " | ".join(t["keywords"] for t in targets)
        hits = [chunks[i] for i in retrieve_targets(question, targets, chunks, bm25, vectors)]
    else:
        keywords = ask(PROMPT_KEYWORDS, question)
        if is_error(keywords):
            trace["error"] = True
            return keywords, []
        trace["keywords"] = keywords
        hits = [chunks[i] for i in retrieve(question, keywords, chunks, bm25, vectors)]
    # 把 pandas 算好的 Excel 统计放在最前面，让模型直接用准确数字：统计类问题放全部；
    # 其他问题检索到了某张**小表**（不超过 TABLE_RANK_ROWS 行）的行时，也放上这张表的统计和排名：
    # 只看到 6 行时，“进口额最多的是谁”会在看得到的几行里挑。大表不放：查“R5 单价多少”时，
    # 一堆“按产品分组的单价合计”反而会把模型带偏
    if stats:
        sheets = {(h["file"], m.group(1)) for h in hits if (m := re.match(r"工作表「(.+?)」第 \d+ 行", h["page"]))}
        small = lambda st: (m := re.match(r"工作表「(.+?)」：共 (\d+) 行", st["text"])) and (st["file"], m.group(1)) in sheets \
            and int(m.group(2)) <= TABLE_RANK_ROWS
        hits = [st for st in stats if "CALC" in kind or small(st)] + hits
    context = "\n\n".join(f"[片段{n}]（{h['file']} {h['page']}）{h['text']}" for n, h in enumerate(hits, 1))
    # 问题点名了资料（“第二个文件”）时告诉模型指的是哪份，不然它会说“文档里没有提到第二个文件”
    named = "、".join(f"《{label}》" for i, label in (docs or ()) if i in trace["docs"])
    note = f"（问题里说的资料指的是{named}，下面的片段都来自它们。）\n" if named else ""
    if u:   # 把理解结果也告诉回答这一步：计算题按思路算，对比题分别说
        note += f"（对问题的理解：{u['ask']}。{('答案的得出方法：' + u['how']) if u['how'] else ''}）\n"
    reply = ask(PROMPT_ANSWER, f"{note}文档片段：\n{context}\n\n问题：{question}")
    if is_error(reply):
        trace["error"] = True
        return reply, []
    return reply, hits
