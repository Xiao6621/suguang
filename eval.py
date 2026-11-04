"""
知识库助手自动评测脚本
用法（在项目文件夹里）：
    .venv/bin/python eval.py              # 跑全部样本 + 用 AppTest 真实跑一遍 app.py 做对照
    .venv/bin/python eval.py --no-apptest # 只跑样本评测
    .venv/bin/python eval.py --no-apptest --real-only   # 只跑第 6 步的真实资料题（my_eval/questions.md）
    .venv/bin/python eval.py --no-apptest --real-only --repeat 3   # 每道题跑 3 次，分清“不稳定”和“真错”
结果写到 samples/eval_output.md 和 samples/eval_output.json；真实资料题的结果单独写到 my_eval/real_output.md（只在本地）

做法：直接 import rag.py 里的函数（build_index / excel_stats / read_pages / answer），
和网页 app.py 调用的是同一套代码，所以 rag.py 改了，评测也会跟着变。
"""
import json, os, re, sys, time, unicodedata, warnings, logging
from pathlib import Path

PROJECT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(PROJECT, "app.py")
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)

import rag
from rag import build_index, excel_stats, read_pages, load_embedder

# ---------- 1. 跑一道题：记录意图、关键词、检索结果、调用了几次模型 ----------
calls = []
_real_ask = rag.ask
def logged_ask(system, user, **kw):
    t0 = time.time()
    out = _real_ask(system, user, **kw)
    calls.append({"system": system[:20], "out": out, "sec": round(time.time() - t0, 1)})
    return out
rag.ask = logged_ask   # rag.answer 内部调用的是 rag.ask，这样每次调用模型都会被记下来

def run_question(question, index, stats=(), docs=None, narrow=None):
    chunks, bm25, vectors = index[:3]
    calls.clear()
    trace = {}
    t0 = time.time()
    reply, hits = rag.answer(question, chunks, bm25, vectors, list(stats), trace=trace, docs=docs, narrow=narrow)
    return {
        "picked": list(trace.get("docs") or []),
        "intent": trace.get("intent"), "understanding": trace.get("understanding"),
        "route": "ERROR" if trace.get("error") or not trace.get("kind") else trace["kind"], "keywords": trace.get("keywords"), "reply": reply,
        "hits_full": [f"{h['file']} {h['page']}" for h in hits],
        "hits": [f"{h['file']} {h['page']}" for h in hits], "hit_texts": [h["text"] for h in hits],
        "llm_calls": len(calls), "sec": round(time.time() - t0, 1),
    }

# ---------- 2. 测试题（expect 里每一组至少命中一个词才算对；evidence 是正确答案所在原文，用来判断检索有没有找对） ----------
CHAT = [
    {"q": "你好", "type": "chat", "route": "CHAT"},
    {"q": "你能做什么", "type": "chat", "route": "INTRO"},
    {"q": "谢谢，辛苦了", "type": "chat", "route": "CHAT"},
]
SUITES = [
    {"file": "rental_contract.docx", "desc": "租房合同（Word，自制）", "qs": [
        {"q": "押金是多少？", "type": "直接事实", "expect": [["9000", "9,000", "玖仟"]], "evidence": "押金人民币9000元"},
        {"q": "如果我提前搬走，要赔房东多少钱？", "type": "同义改写（赔→违约金）", "expect": [["4500", "一个月租金"]], "evidence": "相当于一个月租金的违约金"},
        {"q": "房东多久会把保证金还给我？", "type": "同义改写（保证金→押金）", "expect": [["7"], ["工作日"]], "evidence": "7个工作日内将押金无息退还"},
        {"q": "How much is the monthly rent?", "type": "跨语言（英问中）", "expect": [["4500", "4,500"]], "evidence": "月租金为人民币4500元"},
        {"q": "房子里交付了几台空调？", "type": "表格内容", "expect": [["2", "两"]], "evidence": "空调（格力1.5匹）|2台"},
        {"q": "这套房子可以养宠物吗？", "type": "文档里没有", "refuse": True},
    ]},
    {"file": "sales_2025Q1.xlsx", "desc": "一季度销售表（Excel，5 个工作表，自制）", "qs": [
        {"q": "2月份华东区的销售额是多少？", "type": "直接查表", "expect": [["136480", "136,480", "13.6"]], "evidence": "2月销售额（元）:136480"},
        {"q": "李娜每卖一单能拿几个点的提成？", "type": "同义改写（几个点→提成比例）", "expect": [["5%", "5 %", "百分之五", "5个点"]], "evidence": "提成比例:5%"},
        {"q": "What is the unit price of the robot vacuum R5?", "type": "跨语言（英问中）", "expect": [["2499", "2,499"]], "evidence": "产品:扫地机器人R5"},
        {"q": "订单 SO202502018 是谁卖的？卖了什么、多少台？", "type": "明细行查找", "expect": [["王强"], ["扫地机器人"], ["17"]], "evidence": "订单号:SO202502018，销售员:王强"},
        {"q": "一季度哪个产品卖出的数量最少？", "type": "需要汇总计算", "expect": [["智能手表"], ["154"]], "evidence": None},
        {"q": "4月份的销售额是多少？", "type": "文档里没有", "refuse": True},
        {"q": "一季度哪个销售员的销售额最高？", "type": "需要汇总计算（新增）", "expect": [["李娜"], ["320247", "320,247", "32.02"]], "evidence": None, "new": True},
        {"q": "3月份平均每单金额是多少？", "type": "需要汇总计算（新增）", "expect": [["12159", "12,159", "1.22"]], "evidence": None, "new": True},
    ]},
    {"file": "philips_hd7000_manual.pdf", "desc": "飞利浦 HD7000 电饭煲说明书（中文 PDF，25 页）", "qs": [
        {"q": "这款电饭煲的额定功率是多少？", "type": "直接事实", "expect": [["1250"]], "evidence": "额定输入功率（W）1250"},
        {"q": "坏了的话能免费修多长时间？", "type": "同义改写（免费修→保修）", "expect": [["两年", "2年", "2 年"]], "evidence": "提供两年保修服务"},
        {"q": "我想下班回家就有饭吃，最长能提前多久定时？", "type": "同义改写（定时→预约）", "expect": [["24"]], "evidence": "预约计时器最长可设置24小时"},
        {"q": "What does error code C2 mean on this rice cooker?", "type": "跨语言（英问中）", "expect": [["内锅", "inner pot", "inner bowl"]], "evidence": "防未正确安放内锅保护"},
        {"q": "这款电饭煲售价多少钱？", "type": "文档里没有", "refuse": True},
    ]},
    {"file": "caict_cloud_bluebook_2025.pdf", "desc": "中国信通院《云计算蓝皮书（2025年）》（中文 PDF，52 页）", "qs": [
        {"q": "2024年我国云计算市场规模是多少？", "type": "直接事实", "expect": [["8288"]], "evidence": "我国云计算市场规模达8288亿元"},
        {"q": "国内私有云这块生意一年有多大？", "type": "同义改写（生意→市场规模）", "expect": [["2072"]], "evidence": "私有云市场规模2072亿元"},
        {"q": "How big was the global cloud computing market in 2024?", "type": "跨语言（英问中）", "expect": [["6929", "692.9"]], "evidence": "全球云计算市场规模为6929亿美元"},
        {"q": "亚马逊云和微软云2024年营收分别是多少？", "type": "直接事实（多个数字）", "expect": [["1076"], ["1024"]], "evidence": "1076亿美元和1024亿美元"},
        {"q": "阿里云2024年的营收是多少？", "type": "文档里没有（但提到阿里云）", "refuse": True},
    ]},
    {"file": "rag_paper_en.pdf", "desc": "RAG 原始论文（英文 PDF，即 test.pdf）", "qs": [
        {"q": "RAG 用的检索器是什么？", "type": "跨语言（中问英）", "expect": [["DPR", "Dense Passage Retriev"]], "evidence": "DensePassageRetriever"},
        {"q": "生成部分用的是哪个预训练模型？", "type": "跨语言（中问英）", "expect": [["BART"]], "evidence": "BART-large"},
        {"q": "外部知识库用的是哪个版本的维基百科，切成了多少篇文档？", "type": "跨语言 + 换说法", "expect": [["2018"], ["21"]], "evidence": "December2018"},
        {"q": "What are the two RAG model variants?", "type": "英文直接问", "expect": [["RAG-Sequence", "RAG-Seq"], ["RAG-Token"]], "evidence": "RAG-Sequence"},
        {"q": "RAG 在 ImageNet 图像分类上的准确率是多少？", "type": "文档里没有", "refuse": True},
        {"q": "这篇论文的核心观点是什么？", "type": "全文总结（新增）", "summary": [["检索", "retriev"], ["生成", "generat", "BART"]], "new": True},
        {"q": "帮我总结成 3 条要点", "type": "全文总结·指定条数（新增）", "summary": [["检索", "retriev"], ["生成", "generat", "BART"]], "points": 3, "new": True},
        {"q": "用一段话概括这篇论文主要讲了什么", "type": "全文总结（新增）", "summary": [["检索", "retriev"], ["维基", "Wikipedia"]], "points": 0, "new": True},
    ]},
    {"file": "scan_notice.pdf", "desc": "书店年度工作安排（自制扫描件：只有图片、没有文字层，make_scan_sample.py 生成；用 OCR 识别文字）", "qs": [
        {"q": "今年的销售额目标是多少？", "type": "扫描件 OCR", "expect": [["1260"]], "evidence": "全年销售额目标为1260万元"},
        {"q": "计划新开几家门店？开在哪些城市？", "type": "扫描件 OCR（新增）", "expect": [["3", "三"], ["杭州"], ["苏州"], ["南京"]], "evidence": "计划新开门店3家", "new": True},
        {"q": "全年至少要办多少场读书会？", "type": "扫描件 OCR（新增）", "expect": [["24"]], "evidence": "全年不少于24场", "new": True},
    ]},
    # 在线体验版默认关闭 OCR：扫描件要友好提示，不能崩溃
    {"file": "scan_notice.pdf", "desc": "书店年度工作安排（自制扫描件，OCR 关闭，模拟在线体验版）", "ocr": False, "qs": [
        {"q": "今年的销售额目标是多少？", "type": "扫描件·OCR 关闭：应友好提示而不是崩溃（新增）", "graceful": True, "new": True},
    ]},
]
# 书架：几份资料放在一起，问题里点名某一份（第 5b 步）。docs 是期望认出的资料（按 files 的顺序编号，和侧栏一样），[] 表示没点名
SHELF = {"files": ["rental_contract.docx", "sales_2025Q1.xlsx", "rag_paper_en.pdf"], "qs": [
    {"q": "第一个档案主要是讲什么", "type": "点名资料·按顺序（新增）", "docs": [1], "summary": [["租"], ["押金", "租金"]], "new": True},
    {"q": "那篇英文论文的核心观点是什么？", "type": "点名资料·按描述（新增）", "docs": [3], "summary": [["检索", "retriev"], ["生成", "generat", "BART"]], "new": True},
    {"q": "第二个文件里，2月份华东区的销售额是多少？", "type": "点名资料·按顺序（新增）", "docs": [2], "expect": [["136480", "136,480", "13.6"]], "new": True},
    {"q": "押金是多少？", "type": "没点名·不该乱选资料（新增）", "docs": [], "expect": [["9000", "9,000", "玖仟"]], "new": True},
]}
norm = lambda s: re.sub(r"\s+", "", s or "")

# ---------- 3. 判分 ----------
# 引用标注：[片段2]、【片段2】、(片段1、3)、[片段1-3]、（见片段2、7）…… 判分前全部去掉，免得“片段2”里的 2 被当成答案
CITE_RE = re.compile(r"[\[【(（]\s*片段\s*[\d\s,，、\-–~至和及]*[\]】)）]|片段\s*\d+(?:\s*[,，、和及\-–~至]\s*\d+)*")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
NUM_U_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(百万|千万|万|亿|million|billion|M\b|B\b)?")
SCALE = {"万": 1e4, "百万": 1e6, "千万": 1e7, "亿": 1e8, "million": 1e6, "billion": 1e9, "m": 1e6, "b": 1e9}
CN_DIGIT = "零一二两三四五六七八九十百千万"

def clean_reply(reply):
    t = CITE_RE.sub(" ", reply or "")
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)   # 136,480 → 136480（只认英文逗号，“17，154”不会被合并）
    t = unicodedata.normalize("NFKC", t)             # 全角转半角：５％ → 5%
    return t.replace("**", "")

def num_hit(kw, text):
    """纯数字按“完整的数”比：2 不会命中 12、2018、1.25、片段2；12159 可命中 12159.77；
    带单位的数会换算后比较（13.65万≈136480，2100万=21M），至少 3 位有效数字才换算，避免粗略数误命中"""
    k = float(kw)
    for n, u in NUM_U_RE.findall(text):
        if n == kw or ("." not in kw and n.split(".")[0] == kw):
            return True
        u = u.lower()
        if u and len(n.replace(".", "")) >= 3 and len(kw.replace(".", "")) >= 2:   # 单个数字（如 2 台）不换算，免得“200万”命中 2
            dec = len(n.split(".")[1]) if "." in n else 0
            step, v = SCALE[u] / 10 ** dec, float(n) * SCALE[u]
            if any(abs(round(k * sc / step) * step - v) < step / 2 for sc in (1, 1e4, 1e6, 1e8)):
                return True
    return False

def hit(keyword, text):
    """keyword 是否出现在 text 里（数字和中文数字都要求边界完整）"""
    kw = unicodedata.normalize("NFKC", re.sub(r"(?<=\d),(?=\d{3})", "", keyword)).lower()
    text = text.lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", kw):
        return num_hit(kw, text)
    if kw == "两":                                   # 只认“两台/两个…”，不认“两个月租金”“两室一厅”
        return re.search(r"两\s*[台个部套件只](?!月)", text) is not None
    if kw.endswith("%"):                             # 5% 也接受“5个百分点”
        alt = re.escape(kw[:-1]) + r"\s*个?百分点"
        if re.search(r"(?<![\d.])" + alt, text):
            return True
    pat = re.escape(kw)
    if kw[0].isdigit():
        pat = r"(?<![\d.])" + pat
    if kw[-1].isdigit():
        pat += r"(?!\d)"
    if kw[-1] in CN_DIGIT:                           # 百分之五 ≠ 百分之五十
        pat += f"(?![{CN_DIGIT}])"
    return re.search(pat, text) is not None

# 拒答：只有“第一句话”就在说文档里没有，才算拒答；正文中间顺带出现“并未”“没有明确”不算
REFUSE_RE = re.compile(
    r"(没有|未|并未|并没有|没|未曾)(明确|具体|直接|详细)?(提到|提及|提供|涉及|给出|包含|说明|记载|找到|列出|出现|相关"
    r"|约定|规定|写明|注明|标明|载明|关于|记录|报告)"
    r"|不包含|无法(回答|确定|得知|判断|从)|找不到|查不到|无相关|并无"
    r"|(只|仅)(有|包含|涵盖|覆盖).{0,15}(数据|内容|信息)")
# “带单位的答案数字”：拒答句前面先出现这种数，说明其实给了答案；年份、型号、月份不算
ANS_NUM = re.compile(r"\d+(?:\.\d+)?\s*(?:元|美元|万|亿|%|台|件|单|小时|分钟|W|瓦)")
GUESS_RE = re.compile(r"(约|大约|估计|一般|通常|市场价|参考价|推测|可能是).{0,10}\d")

def first_sentence(text):
    return re.split(r"[。！？!?\n]", text.strip(), maxsplit=1)[0]

def is_refusal(reply):
    first = first_sentence(clean_reply(reply))
    m = REFUSE_RE.search(first)
    return m is not None and not ANS_NUM.search(first[:m.start()])

def judge(item, r):
    if item.get("graceful"):
        return "✅友好提示" if r["route"] != "ERROR" and "扫描件" in r["reply"] else "❌"
    text = clean_reply(r["reply"])
    if item.get("summary"):
        # 总结题：不能是拒答；要带 [片段N] 引用；要提到关键概念；指定了条数时，编号列表的条数要一致
        if is_refusal(r["reply"]) or r["route"] == "ERROR":
            return "❌总结失败（拒答或出错）"
        if "SUMMARY" not in r["route"].upper():
            return f"🟡没有进入总结模式（路由 {r['route']}）"
        if not re.search(r"\[片段\s*\d+", r["reply"]):
            return "🟡总结没有引用出处"
        missing = [g[0] for g in item["summary"] if not any(hit(k, text) for k in g)]
        if missing:
            return f"🟡总结缺少关键概念（{'、'.join(missing)}）"
        if "points" in item:  # 只数顶层编号（行首、不缩进）：1. / 1、/ 一、/ - / •
            n = len(re.findall(r"(?m)^(?:#+\s*)?(?:\d+[.、)）]|[一二三四五六七八九十]+、|[-*•]\s)", text))
            if n != item["points"]:
                return f"🟡条数不对（要求 {item['points']} 条，实际 {n} 条）"
        return "✅正确"
    if item.get("refuse"):
        if not is_refusal(r["reply"]):
            return "❌没拒答（可能瞎编）"
        if GUESS_RE.search(text):
            return "🔍拒答但又估了数字（需人工复核）"
        return "✅正确拒答"
    if is_refusal(r["reply"]):   # 第一句就说“没有提到”：后面顺带列出的词命中了关键词，也不算答出来
        return "❌没答出来（第一句就说没提到）"
    groups = item["expect"]
    ok = sum(any(hit(k, text) for k in g) for g in groups)
    if ok < len(groups):
        return "🟡部分正确" if ok else "❌错误"
    head = first_sentence(text)
    if not all(any(hit(k, head) for k in g) for g in groups):
        return "🔍正确待复核（答案不在第一句，可能是顺带提到的数字）"
    return "✅正确"

def main():
    results = {"chat": [], "docs": []}
    print("加载嵌入模型…", flush=True)
    load_embedder()
    if "--real-only" in sys.argv:
        run_real()
        return
    # 闲聊（文档用合同，内容无关）
    first_doc = os.path.join("samples", SUITES[0]["file"])
    idx = build_index(((SUITES[0]["file"], Path(first_doc).read_bytes()),))
    for item in CHAT:
        r = run_question(item["q"], idx)
        ok = item["route"] in r["route"].upper()
        r.update(q=item["q"], expect_route=item["route"], verdict="✅" if ok else "❌", hits_shown=bool(r["hits"]))
        results["chat"].append(r)
        print(f"[闲聊] {item['q']} → {r['route']} {r['verdict']}  {r['reply'][:60]!r}", flush=True)

    for suite in SUITES:
        path = os.path.join("samples", suite["file"])
        if not os.path.exists(path):  # 第三方样本没有放进仓库，需要按 README 里的链接自行下载
            print(f"\n[{suite['file']}] 文件不存在，跳过（下载方法见 README「样本与版权」）", flush=True)
            continue
        data = Path(path).read_bytes()
        rag.OCR_ENABLED = suite.get("ocr", True)   # 默认开着 OCR；某组题写了 "ocr": False 就关掉（模拟在线体验版）
        rec = {"file": suite["file"], "desc": suite["desc"], "size_kb": len(data) // 1024, "questions": []}
        t0 = time.time()
        try:
            idx = build_index(((suite["file"], data),))
        except Exception as e:
            rec["index_error"] = f"{type(e).__name__}: {e}"
            rec["chars"] = sum(len(t) for _, t in read_pages(suite["file"], data))
            print(f"[{suite['file']}] 建索引失败：{rec['index_error']}（抽出文字 {rec['chars']} 字）", flush=True)
            rec["questions"] = [{"q": it["q"], "type": it["type"], "verdict": "❌建索引失败", "route": "ERROR", "reply": rec["index_error"],
                                 "hits": [], "sec": 0, "new": it.get("new", False)} for it in suite["qs"]]
            results["docs"].append(rec)
            continue
        rec["index_sec"] = round(time.time() - t0, 1)
        chunks = idx[0]
        rec["empty_files"] = list(idx[3]) if len(idx) > 3 else []
        stats = excel_stats(((suite["file"], data),))
        rec["n_chunks"] = len(chunks)
        rec["chars"] = sum(len(t) for _, t in read_pages(suite["file"], data))
        print(f"\n[{suite['file']}] {len(chunks)} 段，建索引 {rec['index_sec']}s，没读到文字的文件: {rec['empty_files']}", flush=True)
        for item in suite["qs"]:
            try:
                r = run_question(item["q"], idx, stats)
            except Exception as e:
                r = {"route": "ERROR", "reply": f"{type(e).__name__}: {e}", "hits": [], "hit_texts": [], "keywords": None, "llm_calls": 0, "sec": 0}
            ev = item.get("evidence")
            if ev:
                gold = [i for i, c in enumerate(chunks) if norm(ev) in norm(c["text"])]
                r["evidence_found_in_doc"] = bool(gold)
                r["evidence_retrieved"] = any(norm(ev) in norm(t) for t in r["hit_texts"])
            r.update(q=item["q"], type=item["type"], verdict=judge(item, r))
            if r["route"].upper() in ("CHAT", "INTRO"):
                r["verdict"] = f"❌被误判为闲聊（原判 {r['verdict']}）"
            r["new"] = item.get("new", False)
            rec["questions"].append(r)
            print(f"  {r['verdict']}  [{item['type']}] {item['q']}\n     检索到: {r['hits']}  证据命中: {r.get('evidence_retrieved')}\n     回答: {r['reply'][:150]!r}", flush=True)
        results["docs"].append(rec)

    rag.OCR_ENABLED = True
    run_shelf(results)

    if "--no-apptest" not in sys.argv:
        results["apptest"] = run_apptest()

    with open("samples/eval_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    write_md(results)
    print("\n已写入 samples/eval_output.md 和 samples/eval_output.json")
    run_real()

def run_apptest():
    """用 Streamlit 官方的 AppTest 真实跑一遍 app.py（不上传文件时用 test.pdf），确认和上面的评测结果一致"""
    from streamlit.testing.v1 import AppTest
    os.environ["PERSIST_LIBRARY"] = "0"   # 关掉资料库：否则本机 library/ 里有资料时，测的是你的资料库而不是 test.pdf
    out = []
    for q in ["你好", "RAG 用的检索器是什么？"]:
        at = AppTest.from_file(APP, default_timeout=180)
        at.run()
        at.chat_input[0].set_value(q).run()
        msgs = at.chat_message
        reply = msgs[-1].markdown[0].value if len(msgs) else ""
        # 新界面的原文卡片：<div class="cite-src"><b>文件名</b><span>位置</span></div>
        sources = [src for m in at.markdown for src in re.findall(r'class="cite-src"><b>[^<]*</b><span>([^<]+)', m.value)]
        out.append({"q": q, "reply": reply, "sources": sources, "exception": [str(e.value) for e in at.exception]})
        print(f"[AppTest] {q} → {reply[:100]!r} 引用: {sources}", flush=True)
    return out

def run_shelf(results):
    """书架题：几份资料一起建索引（每段标上属于哪份），检查问题里点名的资料有没有认对、回答对不对"""
    files = [(f, Path("samples", f).read_bytes()) for f in SHELF["files"] if os.path.exists(os.path.join("samples", f))]
    if len(files) < len(SHELF["files"]):   # 少了文件，编号就对不上了
        print("\n[书架] 样本不全，跳过", flush=True)
        return
    t0 = time.time()
    chunks = [dict(c, doc=name) for name, data in files for c in rag.file_chunks(name, data)]
    vectors = rag.embed([c["text"] for c in chunks])
    stats = [dict(row, doc=name) for name, data in files for row in excel_stats([(name, data)])]
    idx = (chunks, rag.BM25Plus([rag.tokenize(c["text"]) for c in chunks]), vectors)
    docs = [(name, name) for name, _ in files]
    narrow = lambda ids: rag.library_subset(chunks, vectors, stats, ids)
    rec = {"file": "、".join(SHELF["files"]), "desc": f"书架：{len(files)} 份资料放在一起，问题里点名某一份", "size_kb": sum(len(d) for _, d in files) // 1024,
           "chars": sum(len(c["text"]) for c in chunks), "n_chunks": len(chunks), "index_sec": round(time.time() - t0, 1), "questions": []}
    print(f"\n[书架] {len(chunks)} 段：{SHELF['files']}", flush=True)
    for item in SHELF["qs"]:
        try:
            r = run_question(item["q"], idx, stats, docs=docs, narrow=narrow)
        except Exception as e:
            r = {"picked": [], "route": "ERROR", "reply": f"{type(e).__name__}: {e}", "hits": [], "hit_texts": [], "keywords": None, "llm_calls": 0, "sec": 0}
        want = [SHELF["files"][n - 1] for n in item["docs"]]
        verdict = judge(item, r)
        if r["picked"] != want:
            verdict = f"❌认错资料（认成 {r['picked'] or '没点名'}，应为 {want or '没点名'}；回答 {verdict}）"
        r.update(q=item["q"], type=item["type"], verdict=verdict, new=item.get("new", False))
        rec["questions"].append(r)
        print(f"  {r['verdict']}  [{item['type']}] {item['q']}\n     认出: {r['picked']}  检索到: {r['hits']}\n     回答: {r['reply'][:150]!r}", flush=True)
    results["docs"].append(rec)

# ---------- 第 6 步：真实资料题（题目和结果都只在本地） ----------
# 两套题都在同一个「测试书架」上答：把测试资料文件夹（默认 my_eval/docs/，可用环境变量 SUGUANG_REAL_DOCS 指定）里的资料
# 录进单独的测试资料库 my_eval/test_library/
# （和网页一样走 录入 → 载入 → 提问；第一次要算向量，之后复用），不碰用户自己的资料库。
#   ① my_eval/溯光2.0评测新题.json：Grok 帮用户整理的 33 道，格式同 SUITES；单份资料的题只在那一份里找（相当于提问范围选了它），
#      files = 跨文档（全部资料），scope = 提问范围限定，evidence_any = 多个可选证据
#   ② my_eval/questions.md：一行一题的简单格式，「我出的题」「Claude 出的题」两节，在整个书架上答（会认点名的资料）
REAL_FILE = Path("my_eval/questions.md")
REAL_JSON = Path("my_eval/溯光2.0评测新题.json")
REAL_DOCS = Path(os.getenv("SUGUANG_REAL_DOCS") or "my_eval/docs")
REAL_LIBRARY = Path("my_eval/test_library")
REAL_AUTHORS = {"## 我出的题": "用户", "## Claude 出的题": "Claude"}
REAL_LABELS = Path("my_eval/labels.json")   # 理解率、找到率的标准答案：类型、需要的原文、该看哪些资料
# 现在的“判断类型”只有 DOC / CALC / SUMMARY，对照标准类型时这样换算；以后“理解”这一步输出的是中文类型，直接比
OLD_KIND = {"查事实": "DOC", "对比": "DOC", "计算": "CALC", "比较排名": "CALC", "总结": "SUMMARY"}

def check_understanding(label, r, names):
    """理解对不对（类型；该看哪些资料）、找没找到（需要的每一段原文都在检索结果里）。没有标注的项返回 None"""
    if not label:
        return {}
    kind = (r.get("intent") or r["route"]).upper()
    out = {"intent_ok": label["intent"] == kind if not kind.isascii() else OLD_KIND[label["intent"]] in kind}
    if label.get("scope") is not None and r.get("pick"):
        out["scope_ok"] = set(label["scope"]) == set(r.get("picked_names") or [])
    if label["need"]:
        out["found_ok"] = all(any(norm(e) in norm(t) for t in r["hit_texts"]) for e in label["need"])
    return out
GROK_AUTHOR = "用户（Grok 整理）"

def parse_real():
    """读题目文件：只读「我出的题」「Claude 出的题」两节，一行一题：问题 | 答案里应该有的词 | 答案在哪份资料（可选）"""
    items, author = [], None
    for line in REAL_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            author = REAL_AUTHORS.get(line.strip())
            continue
        line = line.strip()
        if not author or not line or line.startswith("#") or "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|")] + [""]
        q, want, src = cols[0], cols[1], cols[2]
        groups = lambda text: [[w.strip() for w in g.split("/") if w.strip()] for g in re.split(r"[,，]", text) if g.strip()]
        item = {"q": q, "author": author, "src": src}
        if want in ("没提到", "没有提到"):
            item.update(refuse=True, type="应该拒答")
        elif want.startswith(("总结：", "总结:")):
            item.update(summary=groups(want[3:]), type="总结")
        else:
            item.update(expect=groups(want), type="事实 / 统计")
        items.append(item)
    return items

def run_real():
    items = parse_real() if REAL_FILE.exists() else []
    suites = json.loads(REAL_JSON.read_text(encoding="utf-8"))["suites"] if REAL_JSON.exists() else []
    if not items and not suites:
        return
    if not REAL_DOCS.exists():
        print(f"\n[真实资料题] 找不到资料文件夹 {REAL_DOCS}，跳过", flush=True)
        return
    store = rag.FileStore(str(REAL_LIBRARY))
    files = [(f.name, f.read_bytes()) for f in sorted(REAL_DOCS.iterdir()) if f.suffix.lower() in (".pdf", ".docx", ".xlsx")]
    t0 = time.time()
    _, failed = rag.library_add(files, store)   # 录过的直接跳过
    for name, why in failed:
        print(f"[真实资料题] 「{name}」{why}", flush=True)
    chunks, bm25, vectors, _, stats = rag.library_load(store)
    docs = [(d["id"], d["name"]) for d in store.docs()]   # 顺序就是测试书架的“侧栏顺序”
    ids, names = {n: i for i, n in docs}, dict(docs)
    narrow = lambda doc_ids: rag.library_subset(chunks, vectors, stats, doc_ids)
    print(f"\n[真实资料题] 测试书架 {len(docs)} 份、{len(chunks)} 段（{time.time() - t0:.0f} 秒）：{'、'.join(names.values())}", flush=True)

    jobs = []   # (题目, 题集名, 只在哪几份里找 / None = 全部, 是否认点名的资料)
    for suite in suites:
        if "file" in suite:
            jobs += [(dict(q, author=GROK_AUTHOR, group=suite["file"]), [ids.get(suite["file"])], False) for q in suite["qs"]]
        else:
            jobs += [(dict(q, author=GROK_AUTHOR, group="跨文档"), [ids.get(f) for f in q["scope"]] if q.get("scope") else None, True)
                     for q in suite["qs"]]
    jobs += [(dict(it, group="整个书架"), None, True) for it in items]

    labels = json.loads(REAL_LABELS.read_text(encoding="utf-8"))["labels"] if REAL_LABELS.exists() else {}
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1   # 每道题跑几次，分清“不稳定”和“真错”
    rows = []
    for item, scope, pick in jobs:
        if scope is not None and None in scope:
            print(f"  ⚠️ 测试书架里没有这份资料，跳过：{item['q']}", flush=True)
            continue
        idx = narrow(scope) if scope else (chunks, bm25, vectors, stats)
        runs = []
        for _ in range(repeat):
            try:
                r = run_question(item["q"], idx[:3], idx[3], docs=docs if pick else None, narrow=narrow if pick else None)
            except Exception as e:
                r = {"picked": [], "route": "ERROR", "reply": f"{type(e).__name__}: {e}", "hits": [], "hit_texts": [], "keywords": None, "llm_calls": 0, "sec": 0}
            verdict = judge(item, r)
            if r["route"].upper() in ("CHAT", "INTRO"):
                verdict = f"❌被误判为闲聊（原判 {verdict}）"
            # 一行一题的格式里第三栏写了资料：答对了还要看引用是不是来自这份资料（答对但引用来自别处，可能是蒙的）
            elif item.get("src") and verdict.startswith("✅") and not item.get("refuse") \
                    and not any(item["src"].lower() in h.lower() for h in r["hits"]):
                verdict = f"🟡答对但引用不是来自「{item['src']}」"
            r.update(verdict=verdict, pick=pick, picked_names=[names.get(i, i) for i in r.get("picked", [])])
            r.update(check_understanding(labels.get(item["q"]), r, names))
            runs.append(r)
        r = runs[-1]
        count = lambda key: (sum(bool(x.get(key)) for x in runs), len(runs)) if key in r else None
        r.update(q=item["q"], author=item["author"], group=item["group"], type=item["type"], label=labels.get(item["q"]),
                 answer_ok=(sum(x["verdict"].startswith("✅") for x in runs), len(runs)),
                 intent_n=count("intent_ok"), scope_n=count("scope_ok"), found_n=count("found_ok"),
                 verdicts=[x["verdict"] for x in runs])
        rows.append(r)
        show = lambda n: f"{n[0]}/{n[1]}" if n else "-"
        print(f"  {r['verdict']}  [{item['author']}·{item['type']}] {item['q']}\n     答对 {show(r['answer_ok'])}  理解 {show(r['intent_n'])}"
              f"  范围 {show(r['scope_n'])}  找到 {show(r['found_n'])}  只看: {r['picked_names'] or ('范围 ' + str([names[i] for i in scope]) if scope else '全部')}"
              f"\n     回答: {r['reply'][:120]!r}", flush=True)
    write_real_md(rows, docs, len(chunks), repeat)

def write_real_md(rows, docs, n_chunks, repeat=1):
    """报告：每组题的 答对率 / 理解率 / 范围 / 找到率（每道题跑 repeat 次，次数都算进去），再列每道题"""
    L = ["# 第 6 步：真实资料评测结果（eval.py 生成，只在本地）", "",
         f"测试书架（{REAL_DOCS}）{len(docs)} 份、{n_chunks} 段：" + "、".join(n for _, n in docs) + f"。每道题跑 {repeat} 次。", ""]
    rate = lambda part, key: (sum(r[key][0] for r in part if r.get(key)), sum(r[key][1] for r in part if r.get(key)))
    pct = lambda a: f"{a[0]}/{a[1]}（{a[0] / a[1]:.0%}）" if a[1] else "-"
    summary = []
    for author in (GROK_AUTHOR, "用户", "Claude"):
        part = [r for r in rows if r["author"] == author]
        if not part:
            continue
        title = {"用户": "你自己加的题", "Claude": "Claude 出的题"}.get(author, "你的题（Grok 整理的 33 道）")
        line = (f"{title}（{len(part)} 道）：理解 {pct(rate(part, 'intent_n'))}，资料范围 {pct(rate(part, 'scope_n'))}，"
                f"找到原文 {pct(rate(part, 'found_n'))}，答对 {pct(rate(part, 'answer_ok'))}")
        summary.append(line)
        L += [f"## {title}", "", line, "",
              "| 资料 | 标准类型 | 问题 | 答对 | 理解 | 范围 | 找到 | 判分 | 只看了哪些资料 | 检索到的位置 | 回答（最后一次） |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        show = lambda n: f"{n[0]}/{n[1]}" if n else "-"
        for r in part:
            ans = r["reply"].replace("\n", " ").replace("|", "／")
            L.append(f"| {r['group']} | {(r['label'] or {}).get('intent', '-')} | {r['q']} | {show(r['answer_ok'])} | {show(r['intent_n'])} | {show(r['scope_n'])}"
                     f" | {show(r['found_n'])} | {'；'.join(r['verdicts'])} | {'、'.join(r['picked_names']) or '-'} | {'；'.join(r['hits']) or '无'} | {ans} |")
        L.append("")
    L.insert(4, "\n".join(f"- {x}" for x in summary) + "\n")
    REAL_FILE.with_name("real_output.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n[真实资料题] " + "\n[真实资料题] ".join(summary) + "\n已写入 my_eval/real_output.md", flush=True)

def write_md(res):
    L = ["# 自动评测输出（eval.py 生成）", ""]
    L += ["## 闲聊路由", "", "| 输入 | 期望 | 实际 | 结果 | 回复 |", "|---|---|---|---|---|"]
    for r in res["chat"]:
        L.append(f"| {r['q']} | {r['expect_route']} | {r['route']} | {r['verdict']} | {r['reply'][:60].replace(chr(10), ' ')}… |")
    for d in res["docs"]:
        L += ["", f"## {d['desc']} — `{d['file']}`", ""]
        if "index_error" in d:
            L.append(f"**建索引失败**：`{d['index_error']}`（pypdf 抽出文字 {d['chars']} 字）")
            continue
        L.append(f"{d['size_kb']} KB，抽出 {d['chars']} 字，切成 {d['n_chunks']} 段，建索引耗时 {d['index_sec']} 秒，没读到文字被跳过的文件：{d.get('empty_files') or '无'}")
        L += ["", "| 题型 | 问题 | 结果 | 检索到的位置 | 证据片段被检索到 | 耗时 | 回答 |", "|---|---|---|---|---|---|---|"]
        for r in d["questions"]:
            ans = r["reply"].replace("\n", " ").replace("|", "／")
            L.append(f"| {r['type']} | {r['q']} | {r['verdict']} | {'；'.join(r.get('hits_full', r['hits'])) or '无'} | {r.get('evidence_retrieved', '-')} | {r['sec']}s | {ans} |")
    # 汇总：原有题目和新增题目分开统计，方便和上一版对比
    for label, is_new in (("原有题目", False), ("新增题目", True)):
        qs = [r for d in res["docs"] for r in d.get("questions", []) if r.get("new", False) == is_new]
        tally = {}
        for r in qs:
            tally[r["verdict"].split("（")[0]] = tally.get(r["verdict"].split("（")[0], 0) + 1
        L += ["", f"**{label}汇总**：" + "，".join(f"{k} {v}" for k, v in tally.items())]
    if res.get("apptest"):
        L += ["", "## AppTest 真实运行 app.py（默认 test.pdf）", ""]
        for a in res["apptest"]:
            L.append(f"- **{a['q']}** → {a['reply'][:200].replace(chr(10), ' ')}  引用：{a['sources']} 异常：{a['exception'] or '无'}")
    Path("samples/eval_output.md").write_text("\n".join(L) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
