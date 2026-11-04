"""溯光的网页界面（Streamlit）。检索和回答逻辑都在 rag.py 里，这个文件只管页面。"""
import base64
import contextlib
import datetime
import hashlib
import html
import os
import re
import threading
from pathlib import Path

import streamlit as st

import rag

BASE_DIR = Path(__file__).parent
TEST_PDF = BASE_DIR / "test.pdf"          # 没上传文件时默认使用的示例论文
CITE_PREVIEW_CHARS = 400                  # 引用卡片里最多显示多少字原文
CACHE_ENTRIES = 5                         # 最多缓存几组文件的索引，防止内存越占越多



# ---------- 配置：云端从 st.secrets 读，本地从环境变量 / .env 读 ----------
def cfg(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # 本地没有 secrets.toml 时会报错，忽略即可
        pass
    return os.getenv(name, default)


def cfg_bool(name, default=False):
    v = cfg(name)
    return default if v is None else str(v).strip().lower() in ("1", "true", "yes", "on")


def cfg_int(name, default):
    try:
        return int(cfg(name, default))
    except (TypeError, ValueError):
        return default


# DEMO_MODE（公开的在线体验版）才启用限额；本地不设置就没有任何限制
DEMO_MODE = cfg_bool("DEMO_MODE")
SESSION_LIMIT = cfg_int("SESSION_LIMIT", 15)    # 每个浏览器会话最多提问次数
DAILY_LIMIT = cfg_int("DAILY_LIMIT", 200)       # 整个应用每天（北京时间）最多提问次数
ALLOW_Q_PARAM = cfg_bool("ALLOW_Q_PARAM", default=not DEMO_MODE)   # ?q= 链接自动提问：本地默认开，在线版默认关
MAX_FILES = cfg_int("MAX_FILES", 5)            # 在线体验版：一次最多上传几个文件
MAX_TOTAL_MB = cfg_int("MAX_TOTAL_MB", 40)      # 在线体验版：上传文件总大小上限
# 资料库：上传的文件存到硬盘，重启后不用重新上传。在线体验版默认关闭（所有访客共用一个应用，开了会互相看到对方的文件）
PERSIST_LIBRARY = cfg_bool("PERSIST_LIBRARY", default=not DEMO_MODE)
# 嵌入模型：本地默认用效果更好的 e5-large（2.2GB）；在线体验版云端内存有限，默认用小模型。也可以用 EMBED_MODEL 指定
rag.set_embed_model(cfg("EMBED_MODEL") or (rag.SMALL_EMBED_MODEL if DEMO_MODE else rag.EMBED_MODEL))
# 扫描件 OCR：本地默认开；在线体验版默认关（云端机器慢，一份几十页的扫描件要识别好几分钟；
# OCR 时内存会多占约 1.5GB，云端免费版内存不够，不要打开）
rag.OCR_ENABLED = cfg_bool("OCR_ENABLED", default=not DEMO_MODE)


def secret_key():
    # 只把 Streamlit Secrets 里的 Key 交给 rag；本地的 .env 由 rag 自己每次重新读取（改了 .env 刷新就生效）
    try:
        return st.secrets.get("DEEPSEEK_API_KEY")
    except Exception:
        return None


rag.set_api_key(secret_key())

# 品牌素材：logo 和两个头像（由 assets/make_assets.py 生成）
LOGO = str(BASE_DIR / "assets" / "logo.png")
AVATARS = {"user": str(BASE_DIR / "assets" / "avatar_user.png"), "assistant": str(BASE_DIR / "assets" / "avatar_ai.png")}
FILE_KINDS = {"pdf": "PDF", "docx": "Word", "xlsx": "Excel"}
# 首页示例问题：按当前载入的文档类型给出（都实际跑过，能答好）
EXAMPLES_DEMO = ["这篇论文的核心观点是什么？", "RAG 用什么做检索器，用什么做生成器？",       # 默认的 test.pdf（RAG 论文）
                 "RAG-Sequence 和 RAG-Token 有什么区别？", "RAG 的知识库可以不重新训练就更新吗？"]
EXAMPLES_EXCEL = ["帮我总结一下表格里的关键数据", "这份表格主要记录了什么？", "哪一项的合计最高？"]
EXAMPLES_DOC = ["这份文档的核心内容是什么？", "帮我总结成 3 条要点", "文档里提到了哪些关键数字？", "有哪些需要特别注意的地方？"]
EX_ICONS = [":material/auto_awesome:", ":material/format_list_bulleted:", ":material/tag:", ":material/priority_high:"]  # 示例卡片的小图标


def pick_examples(names):
    names = [name.lower() for name in names]
    if not names or names == ["test.pdf"]:
        return EXAMPLES_DEMO
    if all(n.endswith(".xlsx") for n in names):
        return EXAMPLES_EXCEL
    return EXAMPLES_DOC

# 回答里的出处标记，例如 [片段3]、[片段1、2]、[片段1,4]
CITE_MARK = re.compile(r"\[片段\s*([\d\s、,，]+)\]")


# 同一组文件只建一次索引（Streamlit 每次交互都会重跑整个脚本，缓存能省掉重复计算）
@st.cache_resource(max_entries=CACHE_ENTRIES)
def build_index(files):
    failed = []
    return (*rag.build_index(files, failed), failed)


@st.cache_resource(max_entries=CACHE_ENTRIES)
def excel_stats(files):
    return rag.excel_stats(files)


# 资料库只在内容变化时（录入了新文件）重新载入；version 是全部资料的编号，变了缓存就自动失效
@st.cache_resource(max_entries=1)
def load_library(version):
    problems = []   # 文件缺了或坏了、被跳过的资料：[(文件名, 原因)]
    return (*rag.library_load(problems=problems), problems)


# “只在这几份资料里找”：同一个范围只拼一次小书架
@st.cache_resource(max_entries=8)
def scoped_index(version, doc_ids):
    chunks, _, vectors, _, stats, _ = load_library(version)
    return rag.library_subset(chunks, vectors, stats, doc_ids)


@st.cache_data
def img_src(path):
    """把图片转成网页能直接显示的 base64 字符串（用在自己写的 HTML 里）"""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


# ---------- 出处：回答里的角标 + 下方的原文卡片 ----------
def cited_numbers(reply):
    """找出回答里实际标注过的片段编号（从 1 开始）"""
    nums = []
    for group in CITE_MARK.findall(reply or ""):
        for n in re.findall(r"\d+", group):
            if int(n) not in nums:
                nums.append(int(n))
    return nums


def render_reply(reply):
    """显示回答：先把模型输出里的 HTML 全部转义（防止文档内容注入网页代码），再把 [片段N] 换成小角标"""
    safe = html.escape(reply or "", quote=False)

    def to_sup(m):
        return "".join(f'<sup class="ref">{n}</sup>' for n in re.findall(r"\d+", m.group(1)))

    st.markdown(CITE_MARK.sub(to_sup, safe), unsafe_allow_html=True)


def cite_card(n, h, dim=False):
    text = html.escape(h["text"][:CITE_PREVIEW_CHARS]) + ("…" if len(h["text"]) > CITE_PREVIEW_CHARS else "")
    return (
        f'<div class="cite{" dim" if dim else ""}"><span class="cite-n">{n}</span><div class="cite-body">'
        f'<div class="cite-src"><b>{html.escape(h["file"])}</b><span>{html.escape(h["page"])}</span></div>'
        f'<p class="cite-text"><mark>{text}</mark></p></div></div>'
    )


def show_citations(hits, reply):
    """回答里标注过的片段排在前面并高亮；检索到但没被引用的放在后面、颜色变淡"""
    used = [n for n in cited_numbers(reply) if 1 <= n <= len(hits)]
    if not used:  # 模型没标出处时，全部按引用显示，避免什么都看不到
        used = list(range(1, len(hits) + 1))
    rest = [n for n in range(1, len(hits) + 1) if n not in used]
    label = f"查看原文出处（{len(used)} 处）"
    with st.expander(label):
        st.markdown("".join(cite_card(n, hits[n - 1]) for n in used), unsafe_allow_html=True)
        if rest:
            st.markdown(f'<div class="rest-label">另外检索到 {len(rest)} 段，回答里没有用到</div>'
                        + "".join(cite_card(n, hits[n - 1], dim=True) for n in rest), unsafe_allow_html=True)


def show_user_text(text):
    """显示用户的问题：把 Markdown 符号转义，避免 ![](外部地址) 之类被渲染成图片或链接"""
    st.markdown(re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<])", r"\\\1", text))


def ask_example(q):
    # 点示例问题按钮时，把问题先存起来，下一轮当作用户输入来处理
    st.session_state.pending = q


# ---------- 资料库：删除要点两次（删除图标 → 弹窗里的红色「删除」） ----------
def ask_delete(doc_id):
    st.session_state.confirm_delete = doc_id


def do_delete(doc_id, name):
    rag.library_remove(doc_id)
    st.session_state.deleted_name = name
    # 上传框会记住传过的文件，不清空的话，下一轮又会把刚删的文件录回去。
    # 换一个 key 就是一个全新的空上传框（其他文件早就存进资料库了，不会丢）
    st.session_state.uploader_gen = st.session_state.get("uploader_gen", 0) + 1


@contextlib.contextmanager
def ocr_progress_bar():
    """录入扫描件时显示「正在识别第 37/300 页」的进度条；没有扫描页时什么都不显示，录完自动消失"""
    slot = st.empty()

    def report(name, done, total):
        slot.progress(done / total, text=f"正在识别「{name}」的扫描页：第 {done} / {total} 页")

    try:
        with rag.ocr_progress(report):
            yield
    finally:
        slot.empty()


def clear_scope():
    st.session_state.scope = []


def doc_labels(library):
    """每份资料的显示名：一般就是文件名；有同名文件时，后面加上录入时间区分"""
    same_name = {}
    for d in library:
        same_name[d["name"]] = same_name.get(d["name"], 0) + 1
    return {d["id"]: d["name"] + (f"（{d['added'][5:16].replace('T', ' ')} 录入）" if same_name[d["name"]] > 1 else "")
            for d in library}


def library_rows(library):
    """侧边栏的资料列表：类型标签 + 文件名 + 段数 + 删除按钮（点了会弹出确认框）"""
    labels = doc_labels(library)
    for d in library:
        kind = FILE_KINDS.get(d["ext"], "文件")
        label = labels[d["id"]]
        size = f'{d["n_chunks"]} 段' if d["n_chunks"] else "无文字"
        left, right = st.columns([0.86, 0.14], vertical_alignment="center", gap="small")
        left.markdown(
            f'<div class="file-item"><span class="kind {kind if kind in FILE_KINDS.values() else "PDF"}">{kind}</span>'
            f'<span class="name" title="{html.escape(label)}">{html.escape(label)}</span><span class="n">{size}</span></div>',
            unsafe_allow_html=True)
        right.button(":material/delete:", key=f"del_{d['id']}", help="从资料库删除", type="tertiary",
                     on_click=ask_delete, args=(d["id"],))


@st.dialog("删除这份资料？")
def confirm_delete_dialog(d):
    """居中弹窗：显示要删的是哪份资料，红色「删除」才真正删除；「取消」或右上角 ✕ 都只是关掉弹窗"""
    kind = FILE_KINDS.get(d["ext"], "文件")
    size = f'{d["n_chunks"]} 段' if d["n_chunks"] else "无文字（可能是扫描件）"
    st.markdown(
        f'<div class="del-card"><span class="kind {kind if kind in FILE_KINDS.values() else "PDF"}">{kind}</span>'
        f'<div><div class="del-name">{html.escape(d["name"])}</div>'
        f'<div class="del-meta">{size} · {d["added"][:16].replace("T", " ")} 录入</div></div></div>'
        '<p class="del-tip">删除后，提问时就找不到它了；原文件、切好的段落和向量都会从本机资料库移除，要重新上传才能找回。</p>',
        unsafe_allow_html=True)
    _, no, yes = st.columns([0.4, 0.3, 0.3])
    if no.button("取消", key="cancel_del", width="stretch"):
        st.rerun()
    if yes.button("删除", key="confirm_del", icon=":material/delete:", width="stretch"):
        do_delete(d["id"], d["name"])
        st.rerun()


st.set_page_config(page_title="溯光 · 每一句回答，都能溯回原文", page_icon=LOGO, layout="centered", initial_sidebar_state="expanded")

# ---------- 白天 / 黑夜：两套颜色 ----------
# 切换时只换下面这组颜色变量，页面不刷新，聊天记录也不会丢。选过的模式记在网址的 ?mode= 里，刷新后还是它
MODES = {"light": "白天", "dark": "黑夜"}
THEMES = {
    "light": {
        "bg": "#FAFAF7", "side": "#F3F2EC", "surface": "#FFFFFF", "surface2": "#F1F0EA", "solid": "#FFFFFF",
        "line": "#E6E4DC", "line2": "#D9D6CC", "ink": "#1B2232", "muted": "#5E6675", "faint": "#8A909B", "quote": "#2E3545",
        "amber": "#F0AE2E", "amber-hi": "#A86B05", "amber-soft": "rgba(245,185,66,.16)", "amber-line": "rgba(217,148,15,.45)",
        "mark-bg": "rgba(255,205,70,.38)", "mark-line": "#E0A21E", "mark-ink": "#1B2232",
        "glow-a": "rgba(255,196,110,.20)", "glow-b": "rgba(79,140,255,.05)", "logo-glow": "0 6px 18px rgba(27,34,50,.18)",
        "input-bg": "#FFFFFF", "input-shadow": "0 10px 30px rgba(27,34,50,.08)", "card-shadow": "0 1px 3px rgba(27,34,50,.05)",
        "pdf-bg": "rgba(248,113,113,.14)", "pdf-ink": "#C2453C", "xls-bg": "rgba(52,211,153,.15)", "xls-ink": "#1E8A5B",
        "word-bg": "rgba(96,165,250,.16)", "word-ink": "#2B5FC4",
        "info-bg": "rgba(79,140,255,.10)", "info-ink": "#1F4FA8", "warn-bg": "rgba(245,185,66,.18)", "warn-ink": "#7A5000",
        "err-bg": "rgba(239,68,68,.10)", "err-ink": "#B42318",
    },
    "dark": {
        "bg": "#0A1020", "side": "#0C1326", "surface": "rgba(255,255,255,.035)", "surface2": "rgba(255,255,255,.06)", "solid": "#141C33",
        "line": "rgba(255,255,255,.08)", "line2": "rgba(255,255,255,.14)", "ink": "#E9EDF6", "muted": "#9AA4BC", "faint": "#6F7992", "quote": "#D5DBE7",
        "amber": "#F5B942", "amber-hi": "#FFD27A", "amber-soft": "rgba(245,185,66,.13)", "amber-line": "rgba(245,185,66,.38)",
        "mark-bg": "rgba(245,185,66,.15)", "mark-line": "rgba(245,185,66,.6)", "mark-ink": "#F4E6C4",
        "glow-a": "rgba(255,196,110,.11)", "glow-b": "rgba(79,140,255,.06)", "logo-glow": "0 0 30px rgba(245,185,66,.28)",
        "input-bg": "#121A30", "input-shadow": "0 20px 50px rgba(0,0,0,.45)", "card-shadow": "none",
        "pdf-bg": "rgba(248,113,113,.14)", "pdf-ink": "#FCA5A5", "xls-bg": "rgba(52,211,153,.13)", "xls-ink": "#6EE7B7",
        "word-bg": "rgba(96,165,250,.14)", "word-ink": "#93C5FD",
        "info-bg": "rgba(79,140,255,.14)", "info-ink": "#BFD4FF", "warn-bg": "rgba(245,185,66,.14)", "warn-ink": "#FFD27A",
        "err-bg": "rgba(239,68,68,.16)", "err-ink": "#FCA5A5",
    },
}
if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = st.query_params.get("mode") if st.query_params.get("mode") in MODES else "light"


def switch_mode():
    pick = st.session_state.get("mode_pick")
    if pick in MODES:
        st.session_state.ui_mode = pick
        st.query_params["mode"] = pick
    else:   # 再点一次已经选中的按钮会变成「都没选」，改回原来的模式
        st.session_state.mode_pick = st.session_state.ui_mode


theme_vars = ";".join(f"--{k}:{v}" for k, v in THEMES[st.session_state.ui_mode].items())
st.markdown(f"<style>:root{{{theme_vars};color-scheme:{st.session_state.ui_mode}}}</style>", unsafe_allow_html=True)

# 自定义样式：颜色全部用上面的变量，所以白天、黑夜共用这一份
st.markdown("""
<style>
:root {--serif: 'Songti SC', 'STSong', 'Noto Serif CJK SC', 'Source Han Serif SC', 'SimSun', serif;}
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stToolbarActions"], [data-testid="stDeployButton"], [data-testid="stDecoration"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 3.2rem; max-width: 780px;}

/* ---------- 底色、文字（Streamlit 自带组件也换成变量里的颜色） ---------- */
.stApp {color: var(--ink);
  background: radial-gradient(700px 480px at 18% -8%, var(--glow-a), transparent 70%),
              radial-gradient(600px 500px at 100% 100%, var(--glow-b), transparent 70%), var(--bg);}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottomBlockContainer"] {background: transparent;}
/* 底部输入框那一条：往上渐变成底色，滚动的回答从输入框后面经过时不会和输入框叠在一起 */
[data-testid="stBottom"] > div {background: linear-gradient(180deg, transparent, var(--bg) 28%);}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {background: var(--side);}
section[data-testid="stSidebar"] {border-right: 1px solid var(--line);}
.stApp label, .stApp h1, .stApp h2, .stApp h3, [data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p, [data-testid="stSpinner"] p {color: var(--ink);}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: var(--faint) !important;}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stExpandSidebarButton"] svg, [data-testid="stTooltipIcon"] svg {color: var(--muted);}
.stApp hr {border-color: var(--line);}
::selection {background: var(--amber-soft);}

/* ---------- 欢迎区 ---------- */
.brand {display: flex; align-items: center; gap: 18px;}
.brand img {width: 60px; height: 60px; border-radius: 15px; box-shadow: var(--logo-glow);}
.brand h1 {font-family: var(--serif); font-size: 2.85rem; font-weight: 600; letter-spacing: .16em;
           color: var(--ink); margin: 0; padding: 0; line-height: 1;}
.stApp p.tagline {font-family: var(--serif); font-size: 1.35rem; color: var(--ink); margin: 22px 0 0;}
.pills {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;}
.pills span {font-size: .76rem; color: var(--muted); padding: 3px 11px; border-radius: 999px; border: 1px solid var(--line2); background: var(--surface);}
.beam {height: 1px; margin: 28px 0 26px; background: linear-gradient(90deg, var(--amber), rgba(245,185,66,.25) 40%, transparent);}

/* 示例回答：直接用真实的回答样式 + 出处卡片，让人一眼看懂“溯回原文”是什么意思 */
.demo {border: 1px solid var(--line); border-radius: 16px; padding: 18px 22px 6px; background: var(--surface); box-shadow: var(--card-shadow);}
.demo .q {color: var(--faint); font-size: .84rem; margin-bottom: 6px;}
.demo .a {font-size: 1rem; line-height: 1.85; color: var(--ink); margin-bottom: 10px;}
.stApp p.demo-cap {color: var(--faint); font-size: .78rem; margin: 10px 2px 30px;}

.stApp p.ex-head {font-family: var(--serif); font-size: 1.08rem; color: var(--ink); margin: 0 0 10px;}
.st-key-examples [data-testid="stHorizontalBlock"] {gap: 12px;}
.st-key-examples [data-testid="stColumn"] [data-testid="stVerticalBlock"] {gap: 12px;}
.st-key-examples .stButton > button {
  justify-content: flex-start; text-align: left; width: 100%; min-height: 62px; padding: 12px 16px; gap: 12px;
  border: 1px solid var(--line); border-radius: 14px; background: var(--surface); box-shadow: var(--card-shadow);
  color: var(--ink); font-size: .92rem;
}
.st-key-examples .stButton > button > div {flex: 1; justify-content: flex-start;}
.st-key-examples .stButton > button span[data-has-shortcut] {display: flex; align-items: center; gap: 12px;}
.st-key-examples .stButton > button [data-testid="stIconMaterial"] {
  flex: none; width: 32px; height: 32px; border-radius: 9px; background: var(--amber-soft); color: var(--amber-hi);
  display: flex; align-items: center; justify-content: center; font-size: 17px;}
.st-key-examples .stButton > button p {color: var(--ink); text-align: left;}
.st-key-examples .stButton > button::after {content: "→"; margin-left: auto; padding-left: 10px; color: var(--faint);}
.st-key-examples .stButton > button:hover {border-color: var(--amber-line); background: var(--amber-soft);}
.st-key-examples .stButton > button:hover::after {color: var(--amber-hi);}
.st-key-examples .stButton > button:focus-visible {outline: 2px solid var(--amber); outline-offset: 2px;}

/* ---------- 回答里的出处角标 ---------- */
sup.ref {display: inline-flex; align-items: center; justify-content: center; min-width: 1.45em; height: 1.45em; padding: 0 .3em; margin: 0 2px;
         border-radius: 999px; background: var(--amber-soft); border: 1px solid var(--amber-line); color: var(--amber-hi);
         font-size: .66em; font-weight: 600; vertical-align: .55em; line-height: 1;}

/* ---------- 原文出处卡片：左边一道琥珀色的「光」照着原文 ---------- */
.cite {display: flex; gap: 14px; padding: 14px 0; border-top: 1px solid var(--line);}
.cite:first-child {border-top: none;}
.cite-n {flex: none; width: 1.55rem; height: 1.55rem; border-radius: 999px; background: var(--amber); color: #1A1406;
         font-size: .76rem; font-weight: 700; display: flex; align-items: center; justify-content: center; margin-top: 1px;
         box-shadow: 0 0 12px rgba(245,185,66,.4);}
.cite-body {min-width: 0; flex: 1; padding-left: 14px; border-left: 2px solid; border-image: linear-gradient(180deg, var(--amber), transparent) 1;}
.cite-src {display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; font-size: .8rem; margin-bottom: 6px;}
.cite-src b {color: var(--ink); font-weight: 600; overflow-wrap: anywhere;}
.cite-src span {color: var(--faint);}
.stApp p.cite-text {font-family: var(--serif); color: var(--quote); font-size: .9rem; line-height: 1.9; margin: 0;
            max-height: 9.5em; overflow: hidden;}
.cite-text mark {background: var(--mark-bg); color: var(--mark-ink); border-bottom: 1.5px solid var(--mark-line); padding: .1em 2px; border-radius: 3px;
                 -webkit-box-decoration-break: clone; box-decoration-break: clone;}
.cite.dim .cite-n {background: var(--surface2); color: var(--muted); box-shadow: none;}
.cite.dim .cite-body {border-image: linear-gradient(180deg, var(--line2), transparent) 1;}
.stApp .cite.dim p.cite-text {color: var(--faint);}
.cite.dim .cite-text mark {background: transparent; border: none; color: inherit;}
.rest-label {color: var(--faint); font-size: .78rem; margin: 14px 0 2px; padding-top: 12px; border-top: 1px dashed var(--line2);}
.rest-label + .cite {border-top: none;}

/* ---------- 对话：用户的问题靠右成气泡，回答在左边 ---------- */
[data-testid="stChatMessage"] {background: transparent; padding: 10px 0; gap: 16px;}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {justify-content: flex-end; padding: 6px 0 14px;}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > :first-child:not([aria-label]) {display: none;}
[aria-label="Chat message from user"] {flex: 0 1 auto; max-width: 78%; margin-left: auto !important; margin-right: 0 !important; background: var(--surface2); border: 1px solid var(--line);
  border-radius: 18px 18px 4px 18px; padding: 10px 18px;}
[data-testid="stChatMessage"] p {line-height: 1.9;}
[data-testid="stChatMessageAvatarImage"] {width: 34px !important; height: 34px !important; border-radius: 50% !important; object-fit: cover;}
[data-testid="stExpander"] details {border: 1px solid var(--line); border-radius: 14px; background: var(--surface); box-shadow: var(--card-shadow);}
[data-testid="stExpander"] summary {font-size: .85rem;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary svg {color: var(--amber-hi);}
[data-testid="stExpander"] summary {background: transparent !important; border-radius: 14px;}
[data-testid="stExpander"] summary:hover {background: var(--amber-soft) !important;}

/* ---------- 输入框 ---------- */
[data-testid="stChatInput"] {background: var(--input-bg); border: 1px solid var(--line2); border-radius: 18px; box-shadow: var(--input-shadow);}
[data-testid="stChatInput"]:focus-within {border-color: var(--amber-line);}
[data-testid="stChatInput"] > div {background: transparent;}
[data-testid="stChatInput"] textarea {color: var(--ink); background: transparent;}
[data-testid="stChatInput"] textarea::placeholder {color: var(--faint);}
[data-testid="stChatInputSubmitButton"] {background: var(--amber); border-radius: 12px; color: #1A1406;}
[data-testid="stChatInputSubmitButton"] svg {color: #1A1406; fill: #1A1406;}
[data-testid="stChatInputSubmitButton"]:disabled {background: var(--amber-soft); opacity: 1;}
[data-testid="stChatInputSubmitButton"]:disabled svg {color: var(--amber-hi); fill: var(--amber-hi);}

/* ---------- 侧边栏 ---------- */
.side-brand {display: flex; align-items: center; gap: 10px;}
.side-brand img {width: 34px; height: 34px; border-radius: 9px;}
.side-brand span {font-family: var(--serif); font-size: 1.4rem; font-weight: 600; letter-spacing: .14em; color: var(--ink);}
.side-brand em {font-style: normal; font-size: .68rem; font-weight: 600; color: var(--amber-hi); border: 1px solid var(--amber-line);
                border-radius: 6px; padding: 0 6px; letter-spacing: .04em; line-height: 1.5;}
.st-key-brand_row {margin-bottom: 10px;}
.st-key-brand_row [data-testid="stHorizontalBlock"] {align-items: center; flex-wrap: nowrap;}
.st-key-mode_pick {display: flex; justify-content: flex-end;}
.st-key-mode_pick [data-testid="stButtonGroup"] > div {gap: 0; padding: 3px; border-radius: 999px; background: var(--surface2); border: 1px solid var(--line);}
.st-key-mode_pick button {min-height: 0; height: 28px; width: 32px; padding: 0; border: none !important; border-radius: 999px !important; background: transparent; color: var(--faint);}
.st-key-mode_pick button[kind="segmented_controlActive"] {background: var(--amber); color: #1A1406; box-shadow: 0 0 10px rgba(245,185,66,.4);}
.st-key-mode_pick button [data-testid="stIconMaterial"] {font-size: 17px;}
.files-head {font-size: .8rem; color: var(--muted); margin: 22px 0 6px; display: flex; justify-content: space-between;}
.files-head b {color: var(--ink); font-weight: 600;}
.file-item {display: flex; align-items: center; gap: 9px; padding: 8px 2px; border-bottom: 1px solid var(--line); font-size: .85rem;}
.file-item .kind, .del-card .kind {flex: none; font-size: .66rem; font-weight: 600; padding: 1px 6px; border-radius: 4px;}
.kind.PDF {background: var(--pdf-bg); color: var(--pdf-ink);}
.kind.Word {background: var(--word-bg); color: var(--word-ink);}
.kind.Excel {background: var(--xls-bg); color: var(--xls-ink);}
.file-item .name {flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink);}
.file-item .n {color: var(--faint); font-size: .72rem; white-space: nowrap;}
/* 资料库列表：每行 = 文件信息 + 删除按钮。Streamlit 给文字块设了 -16px 的下边距，会让两格垂直居中算错，这里去掉 */
.stVerticalBlock.st-key-library, .st-key-library [data-testid="stVerticalBlock"] {gap: 0;}
.st-key-library [data-testid="stHorizontalBlock"] {align-items: center; gap: 4px; border-bottom: 1px solid var(--line);}
.st-key-library [data-testid="stMarkdownContainer"] {margin-bottom: 0;}
.st-key-library .file-item {border-bottom: none;}
.st-key-library .stButton > button[kind="tertiary"] {min-height: 0; height: 28px; padding: 0 4px; color: var(--faint);}
.st-key-library .stButton > button[kind="tertiary"]:hover {color: var(--pdf-ink);}
/* 提问范围多选框 */
.st-key-scope input {caret-color: transparent; min-width: 2px !important; color: var(--ink);}
.st-key-scope [aria-label="Clear all"] {display: none;}
[data-testid="stMultiSelect"] > div > div {background: var(--surface); border-color: var(--line2); color: var(--ink);}
[data-testid="stMultiSelectTagsContainer"] > span > span {background: var(--amber-soft); border: 1px solid var(--amber-line);}
[data-testid="stMultiSelectTagsContainer"] > span > span, [data-testid="stMultiSelectTagsContainer"] > span > span * {color: var(--amber-hi);}
[data-testid="stMultiSelect"] svg {color: var(--faint);}
[data-testid="stMultiSelect"] input::placeholder {color: var(--muted); opacity: 1;}
[data-testid="stMultiSelectDropdown"] {background: var(--solid); border: 1px solid var(--line); box-shadow: var(--input-shadow);}
[data-testid="stMultiSelectDropdown"] [role="option"] {color: var(--ink);}
[data-testid="stMultiSelectDropdown"] [role="option"]:hover, [data-testid="stMultiSelectDropdown"] [aria-selected="true"] {background: var(--amber-soft);}
[data-baseweb="select"] > div {background: var(--surface); border-color: var(--line2); color: var(--ink);}
[data-baseweb="select"] svg {color: var(--faint);}
[data-baseweb="select"] [data-baseweb="tag"] {background: var(--amber-soft); border: 1px solid var(--amber-line);}
[data-baseweb="select"] [data-baseweb="tag"] span, [data-baseweb="select"] [data-baseweb="tag"] svg {color: var(--amber-hi);}
[data-baseweb="select"] [placeholder], [data-baseweb="select"] div[aria-hidden] {color: var(--muted);}
[data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"] {background: var(--solid); border: 1px solid var(--line);}
[data-baseweb="popover"] li {color: var(--ink);}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {background: var(--amber-soft);}
[data-baseweb="tooltip"] > div, [data-baseweb="tooltip"] [role="tooltip"] {background: var(--solid); color: var(--ink);}
.st-key-clear_scope {margin-top: -12px;}
.st-key-clear_scope .stButton > button[kind="tertiary"] {min-height: 0; height: 28px; padding: 0 2px; font-size: .85rem; color: var(--amber-hi);}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {border-radius: 10px; background: transparent; color: var(--muted); border-color: var(--line2);}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {color: var(--amber-hi); border-color: var(--amber-line);}
section[data-testid="stSidebar"] .stButton > button:disabled {opacity: .55;}
.side-foot {display: flex; gap: 8px; color: var(--faint); font-size: .72rem; margin-top: 28px; line-height: 1.7;}
.side-foot i {flex: none; width: 6px; height: 6px; border-radius: 50%; background: #34D399; margin-top: 7px; box-shadow: 0 0 8px #34D399;}

/* 上传框：把英文提示换成中文（原文字缩到 0，再用 ::after 补上中文） */
[data-testid="stFileUploaderDropzone"] {border-radius: 12px; border: 1.5px dashed var(--amber-line); background: rgba(245,185,66,.04);}
[data-testid="stFileUploaderDropzone"] svg {color: var(--amber-hi);}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {font-size: 0 !important; background: var(--surface); border-color: var(--line2);}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] * {display: none;}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]::after {content: "选择文件"; font-size: .88rem; color: var(--ink);}
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small {font-size: 0 !important;}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {content: "或拖到这里，单个最大 20MB"; font-size: .76rem; color: var(--muted);}
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFileName"] {color: var(--ink);}
[data-testid="stFileUploaderFile"] small {color: var(--faint);}

/* 提示框（信息 / 警告 / 出错）和弹出提示 */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {background: var(--info-bg);}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) * {color: var(--info-ink);}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {background: var(--warn-bg);}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) * {color: var(--warn-ink);}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {background: var(--err-bg);}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) * {color: var(--err-ink);}
[data-testid="stToast"] {background: var(--solid); color: var(--ink); border: 1px solid var(--line);}

/* 删除确认弹窗：Streamlit 默认贴在页面上方，用上下 auto 边距让它垂直居中（屏幕太矮放不下时会自动退回贴顶，不会被裁掉） */
[data-testid="stDialog"] > div {margin-top: auto; margin-bottom: auto;}
[data-testid="stDialog"] [role="dialog"] {background: var(--solid); color: var(--ink); border: 1px solid var(--line);}
[data-testid="stDialog"] [role="dialog"] button[aria-label="Close"] svg {color: var(--muted);}
.del-card {display: flex; gap: 12px; align-items: flex-start; padding: 14px 16px; border: 1px solid var(--line);
           border-radius: 12px; background: var(--surface2);}
.del-card .kind {margin-top: 3px;}
.del-name {color: var(--ink); font-weight: 600; overflow-wrap: anywhere; line-height: 1.5;}
.del-meta {color: var(--muted); font-size: .82rem; margin-top: 2px;}
.stApp p.del-tip {color: var(--muted); font-size: .9rem; line-height: 1.7; margin: 14px 0 6px;}
.st-key-cancel_del button {background: transparent; color: var(--muted); border-color: var(--line2);}
.st-key-confirm_del button {background: #C0392B; border-color: #C0392B; color: #fff;}
.st-key-confirm_del button p {color: #fff;}
.st-key-confirm_del button:hover, .st-key-confirm_del button:focus-visible {background: #A93226; border-color: #A93226; color: #fff;}

@media (max-width: 640px) {
  .block-container {padding-top: 2rem;}
  .brand h1 {font-size: 2.2rem;}
  .demo {padding: 14px 14px 4px;}
}
</style>
""", unsafe_allow_html=True)

# 启动检查：没有 API Key 就提示怎么填，不再往下运行
if not rag.has_api_key():
    st.error("没有找到 DeepSeek API Key。本地运行：在项目文件夹的 .env 文件里写上 `DEEPSEEK_API_KEY=你的key`（格式可参考 .env.example），保存后刷新页面。"
             "部署到 Streamlit Cloud：在应用的 Settings → Secrets 里填写 `DEEPSEEK_API_KEY = \"你的key\"`。")
    st.stop()


# 嵌入模型：第一次运行要下载（大小见 rag.EMBED_MODELS），下载时给个提示；之后同一个进程里只加载一次
@st.cache_resource
def warm_embedder():
    return rag.load_embedder()


if rag.embedder_cached():
    warm_embedder()
else:
    with st.spinner(f"第一次启动：正在下载多语言嵌入模型（约 {rag.embed_model_size()}），需要几分钟…"):
        warm_embedder()


# 每天的提问总数（所有访客共用，进程重启会清零）
@st.cache_resource
def daily_counter():
    return {"date": None, "count": 0, "lock": threading.Lock()}


def today():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date()


def check_quota():
    """在线体验版的限额检查。返回 None 表示可以提问，否则返回给用户看的提示"""
    if not DEMO_MODE:
        return None
    if st.session_state.get("asked", 0) >= SESSION_LIMIT:
        return f"这是在线体验版，每次最多问 {SESSION_LIMIT} 个问题，已经用完啦。想继续体验，可以在 GitHub 下载源码，填上自己的 DeepSeek Key 在本地运行。"
    c = daily_counter()
    with c["lock"]:
        if c["date"] != today():
            c["date"], c["count"] = today(), 0
        if c["count"] >= DAILY_LIMIT:
            return "今天在线体验的总提问次数已经用完了，明天再来吧。也可以在 GitHub 下载源码，在本地运行。"
        c["count"] += 1
    st.session_state.asked = st.session_state.get("asked", 0) + 1
    return None


if "history" not in st.session_state:
    st.session_state.history = []

# ---------- 侧边栏 ----------
with st.sidebar:
    # 第一行：logo + 名字 + 版本，右边是白天 / 黑夜切换
    with st.container(key="brand_row"):
        brand, mode_col = st.columns([0.62, 0.38], vertical_alignment="center", gap="small")
        brand.markdown(f'<div class="side-brand"><img src="{img_src(LOGO)}" alt=""><span>溯光</span><em>2.0</em></div>',
                       unsafe_allow_html=True)
        st.session_state.setdefault("mode_pick", st.session_state.ui_mode)
        mode_col.segmented_control("白天 / 黑夜", options=list(MODES), key="mode_pick", on_change=switch_mode,
                                   format_func=lambda m: ":material/light_mode:" if m == "light" else ":material/dark_mode:",
                                   label_visibility="collapsed")
    uploads = st.file_uploader("上传文档", type=["pdf", "docx", "xlsx"], accept_multiple_files=True,
                               help="支持 PDF、Word、Excel，可以一次选多份",
                               key=f"uploader{st.session_state.get('uploader_gen', 0)}")
    if PERSIST_LIBRARY:
        st.caption("上传的文件会存进本机资料库，下次打开不用重新上传。")
    try:
        library = rag.FileStore().docs() if PERSIST_LIBRARY else []
    except rag.LibraryBroken as e:   # 目录坏了：已经备份，资料库从空的开始，页面照常能用
        st.warning(str(e))
        library = []
    if not uploads and not library and TEST_PDF.exists():
        st.caption("还没上传文件，先用示例论文 test.pdf 演示")

if uploads and DEMO_MODE:
    # 在线体验版：限制文件个数和总大小，防止一次上传太多把内存撑满
    if len(uploads) > MAX_FILES:
        st.sidebar.warning(f"在线体验版一次最多处理 {MAX_FILES} 个文件，只用前 {MAX_FILES} 个。")
        uploads = uploads[:MAX_FILES]
    kept, total = [], 0
    for f in uploads:
        if total + f.size > MAX_TOTAL_MB * 1024 * 1024:
            st.sidebar.warning(f"在线体验版上传总大小最多 {MAX_TOTAL_MB}MB，「{f.name}」及之后的文件已跳过。")
            break
        kept.append(f)
        total += f.size
    uploads = kept
if uploads:
    files = tuple((f.name, f.getvalue()) for f in uploads)
elif TEST_PDF.exists() and not library:
    files = (("test.pdf", TEST_PDF.read_bytes()),)
else:
    files = ()  # 没上传、也没有示例文件：只能闲聊（开了资料库时，用资料库里的文件）
chunks, bm25, vectors, stats, empty_files, failed = [], None, None, [], [], []
lib_version = None   # 资料库里全部资料的编号；载入资料库后才有
try:
    if PERSIST_LIBRARY and (uploads or library):
        if uploads:
            # 已经录入过的文件按内容哈希直接跳过，只有新文件才需要切段、算向量。
            # 读取失败的文件记在 bad_files 里，只试这一次：它还留在上传框里时，之后每次提问不会再重读一遍（大扫描件要好几分钟）
            bad = st.session_state.setdefault("bad_files", {})
            # 按（文件名, 内容, 哈希）一个个录入：两个同名不同内容的文件一起传，失败记录也不会记到对方头上
            uploaded = [(name, data, hashlib.sha256(data).hexdigest()) for name, data in files]
            with st.spinner("正在把新文件录入资料库…（扫描件要逐页识别文字，每页约 2 秒）" if rag.OCR_ENABLED else "正在把新文件录入资料库…"), \
                    ocr_progress_bar():
                for name, data, h in uploaded:
                    if h not in bad:
                        _, why = rag.library_add([(name, data)])
                        if why:
                            bad[h] = why[0]
            failed = [bad[h] for _, _, h in uploaded if h in bad]   # 坏文件只跳过它自己，资料库里其他资料照常能用
            library = rag.FileStore().docs()
        # 处理规则更新过（比如加了 OCR）：用旧规则处理的资料从存好的原文件重新处理一次，不用重新上传
        bad = st.session_state.setdefault("bad_files", {})
        outdated = [d for d in library if rag.library_outdated(d) and d["id"] not in bad]
        if outdated:
            with st.spinner(f"读取规则更新了，正在重新处理 {len(outdated)} 份旧资料（只需要这一次；扫描件要逐页识别，会慢一些）…"), \
                    ocr_progress_bar():
                _, upgrade_failed = rag.library_upgrade(skip=set(bad))
            bad.update({doc_id: (name, why) for doc_id, name, why in upgrade_failed})
            library = rag.FileStore().docs()
        failed = failed + [bad[d["id"]] for d in library if d["id"] in bad and bad[d["id"]] not in failed]
        stale = [d for d in library if d["n_chunks"] and d["embed_model"] != rag.EMBED_MODEL]
        tip = f"嵌入模型换过了，正在为资料库里的 {len(stale)} 份资料重新计算向量（只需要这一次）…" if stale else "正在载入资料库…"
        with st.spinner(tip):
            lib_version = tuple((d["id"], d["n_chunks"]) for d in library)   # 带上段数：扫描件用 OCR 重录后编号不变、段数变了
            chunks, bm25, vectors, empty_files, stats, problems = load_library(lib_version)
        for name, why in problems:   # 只跳过这一份，其他资料照常能用；可以在侧栏把它删掉再重新上传
            st.sidebar.warning(f"「{name}」在资料库里的文件缺失或损坏，这次先跳过（{why}）。可以把它删掉再重新上传。")
        doc_names = [d["name"] for d in library]
    else:
        with st.spinner("正在为文档建立索引…"):
            chunks, bm25, vectors, empty_files, failed = build_index(files)
            stats = excel_stats(files)
        doc_names = [name for name, _ in files]
except Exception as e:
    doc_names = [name for name, _ in files]
    st.error(f"读取文档时出错了：{e}。可以换个文件试试，或者把文件另存一份再上传。")
for name, why in failed:
    st.sidebar.warning(f"「{name}」{why}。")

with st.sidebar:
    if st.session_state.get("deleted_name"):
        st.toast(f"已从资料库删除「{st.session_state.pop('deleted_name')}」")
    if PERSIST_LIBRARY and library:
        # 资料库模式：直接按目录列出（没有文字的扫描件也列出来，方便删除）；资料多时放进可滚动的框里
        st.markdown(f'<div class="files-head"><span>资料库</span><span><b>{len(library)}</b> 份 · '
                    f'{sum(d["n_chunks"] for d in library)} 段</span></div>', unsafe_allow_html=True)
        with (st.container(height=360, border=False, key="library") if len(library) > 8 else st.container(border=False, key="library")):
            library_rows(library)
        # 点了删除图标：弹出确认框。编号用过就清掉，这样点 ✕ 关掉弹窗后不会再弹出来
        pending = st.session_state.pop("confirm_delete", None)
        doc = next((d for d in library if d["id"] == pending), None)
        if doc:
            confirm_delete_dialog(doc)
        # 提问范围：不选 = 全部资料；选了几份，就只在这几份里找（总结也只总结这几份）
        searchable = [d for d in library if d["n_chunks"]]
        # 选过的资料如果被删了，从选择里去掉。每次都要清：删到只剩 1 份时选择框不显示，不清的话旧编号会一直留着
        # 只剩 0–1 份可检索的资料时直接清空：这时选择框不显示，旧选择留着清不掉，之后新传的资料会被排除在外
        if st.session_state.get("scope"):
            valid = {d["id"] for d in searchable} if len(searchable) > 1 else set()
            st.session_state.scope = [x for x in st.session_state.scope if x in valid]
        if len(searchable) > 1:
            labels = doc_labels(library)
            st.multiselect("提问范围", options=[d["id"] for d in searchable], format_func=labels.get, key="scope",
                           placeholder="全部资料",
                           help="不选就是在全部资料里找。选了几份之后，提问、总结都只看这几份。")
            # Streamlit 自带的清空 ⊗ 在下拉菜单开着时常常点不中（菜单一收，框变矮，⊗ 跟着跳走），
            # 所以藏起来、换成这个按钮；CSS 里也让框的高度不再跟着菜单变
            if st.session_state.get("scope"):
                st.button("全部资料", icon=":material/restart_alt:", type="tertiary", key="clear_scope", on_click=clear_scope)
    # 以下是没开资料库时（在线体验版 / 示例 test.pdf）的文件列表：类型标签 + 文件名 + 段数
    counts = {}
    for c in chunks:
        counts[c["file"]] = counts.get(c["file"], 0) + 1
    if counts and not (PERSIST_LIBRARY and library):
        rows = "".join(
            f'<div class="file-item"><span class="kind {FILE_KINDS.get(name.lower().rsplit(".", 1)[-1], "PDF")}">'
            f'{FILE_KINDS.get(name.lower().rsplit(".", 1)[-1], "文件")}</span>'
            f'<span class="name" title="{html.escape(name)}">{html.escape(name)}</span><span class="n">{n} 段</span></div>'
            for name, n in counts.items()
        )
        st.markdown(f'<div class="files-head">已载入 {len(counts)} 个文件</div>{rows}', unsafe_allow_html=True)
    for name in ([] if PERSIST_LIBRARY and library else empty_files):   # 资料库模式下，列表里已经标了“无文字”
        st.warning(f"「{name}」没有读到文字，可能是扫描件（图片），已跳过。")
    st.write("")
    if st.button("清空对话", width="stretch", disabled=not st.session_state.history):
        st.session_state.history = []
        st.rerun()
    quota_slot = st.empty()   # 在线体验版的剩余次数，等这次提问计数之后再写（不然会慢一拍）
    st.markdown('<div class="side-foot"><i></i><span>回答由 DeepSeek 生成，检索在本机完成。<br>'
                '上传的文档片段（问题里点名资料时还有资料名）会发送给 DeepSeek。</span></div>', unsafe_allow_html=True)

scope = (st.session_state.get("scope") or []) if PERSIST_LIBRARY and library and lib_version else []
if scope:
    chunks, bm25, vectors, stats = scoped_index(lib_version, tuple(sorted(scope)))
    doc_names = [d["name"] for d in library if d["id"] in scope]

if not doc_names:
    st.info("还没有文档：请在左边上传 PDF、Word 或 Excel，然后就可以提问了。")
elif not chunks:
    st.info("上传的文件里没有可以读取的文字，暂时只能闲聊。请上传带文字的 PDF、Word 或 Excel。")

# 输入框固定在页面底部；示例按钮点过的问题也当作输入
# 选了提问范围时，在输入框的提示文字里写明，打字前就能看到
hint = ("针对文档提问，比如：这篇文档主要讲了什么？" if not scope else
        f"只在「{doc_labels(library).get(scope[0], '选中的资料')}」里找，请提问…" if len(scope) == 1 else f"只在选中的 {len(scope)} 份资料里找，请提问…")
typed = st.chat_input(hint, max_chars=500 if DEMO_MODE else None)
# 链接里带 ?q=问题 时，打开页面就直接问这个问题（方便分享和演示）；用过就从地址栏去掉，刷新不会重复提问
# 在线体验版默认关闭（防止别人用链接消耗额度），由 ALLOW_Q_PARAM 控制
link_q = (st.query_params.get("q") or "")[:300] if ALLOW_Q_PARAM else ""   # 限制长度，防止超长链接
if "q" in st.query_params:
    del st.query_params["q"]
question = typed or st.session_state.pop("pending", None) or link_q

# ---------- 欢迎区：还没开始聊天时显示 ----------
if not st.session_state.history and not question:
    demo_hit = {"file": "rental_contract.docx", "page": "正文",
                "text": "……押金人民币9000元……7个工作日内将押金无息退还……"}
    # 标题下的功能小标签：按实际开了哪些功能显示（在线体验版没有资料库和 OCR）
    pills = (["本机检索"] if not DEMO_MODE else []) + (["扫描件 OCR"] if rag.OCR_ENABLED else []) + ["表格统计"] \
        + (["整个书架一起问"] if PERSIST_LIBRARY else [])
    st.markdown(
        f'<div class="brand"><img src="{img_src(LOGO)}" alt=""><h1>溯光</h1></div>'
        '<p class="tagline">每一句回答，都能溯回原文。</p>'
        f'<div class="pills">{"".join(f"<span>{p}</span>" for p in pills)}</div><div class="beam"></div>'
        '<div class="demo"><div class="q">押金是多少？什么时候退？</div>'
        '<div class="a">押金为人民币 9000 元，租期结束、费用结清后 7 个工作日内无息退还。<sup class="ref">1</sup></div>'
        f'{cite_card(1, demo_hit)}</div>'
        '<p class="demo-cap">示例：对一份租房合同提问。每个角标都对应下方的原文，可以逐句核对。</p>',
        unsafe_allow_html=True,
    )
    with st.container(key="examples"):
        st.markdown('<p class="ex-head">从一个问题开始</p>', unsafe_allow_html=True)
        cols = st.columns(2)   # 两列卡片；手机上 Streamlit 会自动变成一列
        for i, q in enumerate(pick_examples(doc_names)):
            cols[i % 2].button(q, key=f"ex{i}", icon=EX_ICONS[i % len(EX_ICONS)], on_click=ask_example, args=(q,),
                               width="stretch")

# ---------- 聊天记录（包括每条回答的引用片段） ----------
for msg in st.session_state.history:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg["role"] == "assistant":
            if msg.get("note"):
                st.caption(msg["note"])
            render_reply(msg["content"])
            if msg.get("hits"):
                show_citations(msg["hits"], msg["content"])
        else:
            show_user_text(msg["content"])

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        show_user_text(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("正在查找原文…"):
            no_text = "还没有文档，请先在左边上传 PDF、Word 或 Excel，再来问我。" if not doc_names else rag.NO_TEXT_REPLY
            blocked = check_quota()
            trace = {}
            if blocked:
                reply, hits = blocked, []
            elif PERSIST_LIBRARY and library and lib_version:
                # 问题里点名了某几份资料（“第一个档案”“NIST 那份”）时，这一问只看那几份；“第几个”按侧栏顺序数
                labels = doc_labels(library)
                reply, hits = rag.answer(question, chunks, bm25, vectors, stats, trace=trace, no_text_reply=no_text,
                                         docs=[(d["id"], labels[d["id"]]) for d in library],
                                         narrow=lambda ids: scoped_index(lib_version, tuple(sorted(ids))))
            else:
                reply, hits = rag.answer(question, chunks, bm25, vectors, stats, no_text_reply=no_text)
        note = f"这次按你的问题，只看了：{'、'.join(labels[i] for i in trace['docs'])}" if trace.get("docs") else ""
        if note:
            st.caption(note)
        render_reply(reply)
        if hits:
            show_citations(hits, reply)
    st.session_state.history.append({"role": "assistant", "content": reply, "hits": hits, "note": note})

if DEMO_MODE:
    left = max(0, SESSION_LIMIT - st.session_state.get("asked", 0))
    quota_slot.caption(f"在线体验版：本次还可以问 {left} 个问题；最多 {MAX_FILES} 个文件，单个 20MB、总共 {MAX_TOTAL_MB}MB。")
