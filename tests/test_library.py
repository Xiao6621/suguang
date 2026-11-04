"""资料库、OCR、点名资料的小测试：不调用 DeepSeek，嵌入模型和 OCR 都换成假的，几秒钟跑完。
运行：.venv/bin/python -m unittest discover tests -v

每个测试对应一次修过的问题（见 docs/2.0/开发记录.md 的「审查与修复」），以后改资料库相关的代码，跑一遍就知道有没有改坏。"""
import hashlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from docx import Document
from PIL import Image
from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DEEPSEEK_API_KEY", "test-dummy")   # 测试不调 DeepSeek；但页面一开始会检查有没有 Key，没有 .env 时（别人 clone 下来）会停住
sys.path.insert(0, str(ROOT))
import rag  # noqa: E402

APP = str(ROOT / "app.py")


def fake_embed(texts):
    """假的嵌入模型：同一段文字永远得到同一个向量"""
    out = [np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).random(8) for t in texts]
    return np.array(out).reshape(len(texts), 8)


class FakeOCR:
    """假的 OCR：每次认出一行固定文字；fail_on 里的第几次调用会报错"""
    def __init__(self, fail_on=()):
        self.calls, self.fail_on = 0, set(fail_on)

    def __call__(self, img):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("假装这一页认字失败")
        # 真实的扫描页每页能认出几百上千字，这里给 100 多字，比水印那一行多得多
        return type("R", (), {"txts": (f"扫描页认出的文字第{self.calls}次。" * 10,)})()


def docx(text):
    b = io.BytesIO()
    d = Document()
    d.add_paragraph(text)
    d.save(b)
    return b.getvalue()


def scan_pdf(pages=2, watermark=False):
    """做一份“扫描件”：每页是一张图片（没有文字层）；watermark=True 时角上加一行真文字，像扫描全能王那样；
    也可以传一段文字，当成 PPT 导出页上的真文字（整页背景图 + 几行字）"""
    imgs = [Image.new("RGB", (600, 850), "white") for _ in range(pages)]
    b = io.BytesIO()
    imgs[0].save(b, "PDF", save_all=True, append_images=imgs[1:], resolution=72)
    if not watermark:
        return b.getvalue()
    r, w = PdfReader(io.BytesIO(b.getvalue())), PdfWriter()
    for page in r.pages:
        wm = PageObject.create_blank_page(width=float(page.mediabox.width), height=float(page.mediabox.height))
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        wm[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        cs = StreamObject()
        line = watermark if isinstance(watermark, str) else "Scanned with CamScanner"
        cs.set_data(f"BT /F1 8 Tf 20 12 Td ({line}) Tj ET".encode())
        wm[NameObject("/Contents")] = w._add_object(cs)
        page.merge_page(wm)
        w.add_page(page)
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


def mixed_pdf():
    """第 1 页是文字（取自样本论文），第 2 页是扫描图片"""
    w = PdfWriter()
    w.add_page(PdfReader(str(ROOT / "samples" / "rag_paper_en.pdf")).pages[0])
    w.add_page(PdfReader(io.BytesIO(scan_pdf(1))).pages[0])
    b = io.BytesIO()
    w.write(b)
    return b.getvalue()


def old_style(store):
    """去掉 pipeline / ocr 字段，模拟第 5 步以前录入的资料"""
    docs = store.docs()
    for d in docs:
        d.pop("pipeline", None)
        d.pop("ocr", None)
    store._write_json("manifest.json", docs)


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.saved = (rag.FileStore.__init__.__defaults__, rag.embed, rag.load_ocr, rag.OCR_ENABLED, rag.EMBED_MODEL,
                      rag.load_embedder, rag.embedder_cached)
        rag.FileStore.__init__.__defaults__ = (self.dir,)
        rag.embed = fake_embed
        rag.load_embedder = lambda: None   # 页面一打开会先加载嵌入模型；没下载过（别人 clone 下来）时要下 2.2GB，测试里跳过
        rag.embedder_cached = lambda: True
        self.ocr = FakeOCR()
        rag.load_ocr = lambda: self.ocr
        rag.OCR_ENABLED = True
        self.store = rag.FileStore()

    def tearDown(self):
        (rag.FileStore.__init__.__defaults__, rag.embed, rag.load_ocr, rag.OCR_ENABLED, rag.EMBED_MODEL,
         rag.load_embedder, rag.embedder_cached) = self.saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def ids(self):
        return [d["id"] for d in self.store.docs()]

    # ---------- 第 1、4 步：存、读、删、提问范围 ----------
    def test_load_matches_build_index(self):
        files = [("A.docx", docx("押金是9000元")), ("B.docx", docx("保修期两年"))]
        rag.library_add(files)
        chunks = rag.library_load()[0]
        self.assertEqual([(c["file"], c["text"]) for c in chunks], [(c["file"], c["text"]) for c in rag.build_index(files)[0]])

    def test_duplicate_is_skipped(self):
        rag.library_add([("A.docx", docx("押金是9000元"))])
        self.assertEqual(rag.library_add([("A.docx", docx("押金是9000元"))]), ([], []))
        self.assertEqual(len(self.store.docs()), 1)

    def test_remove(self):
        rag.library_add([("A.docx", docx("押金")), ("B.docx", docx("保修"))])
        self.assertTrue(rag.library_remove(self.ids()[0]))
        chunks, _, vectors, _, _ = rag.library_load()
        self.assertEqual([c["file"] for c in chunks], ["B.docx"])
        self.assertEqual(len(vectors), 1)

    def test_model_change_reembeds(self):
        rag.library_add([("A.docx", docx("押金"))])
        rag.EMBED_MODEL = "另一个模型"
        rag.library_load()
        self.assertEqual(self.store.docs()[0]["embed_model"], "另一个模型")

    def test_subset_with_deleted_id_is_empty(self):
        rag.library_add([("A.docx", docx("押金"))])
        chunks, _, vectors, _, stats = rag.library_load()
        self.assertEqual(rag.library_subset(chunks, vectors, stats, ["已经删掉的编号"])[0], [])

    def test_scope_delete_does_not_crash_app(self):
        """第一次审查第 1 条：提问范围只选 B，删掉 B 后只剩 1 份，原来整页 KeyError"""
        from streamlit.testing.v1 import AppTest
        rag.library_add([("B.docx", docx("押金是9000元")), ("C.docx", docx("保修期两年"))])
        b = next(d["id"] for d in self.store.docs() if d["name"] == "B.docx")
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        at.multiselect(key="scope").select(b).run()
        rag.library_remove(b)
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.scope, [])

    def test_scope_cleared_when_one_doc_left(self):
        """第二轮审查 N6：选 B、C，删掉 C 只剩 B；原来 scope 一直是 [B]，之后新传的资料会被排除"""
        from streamlit.testing.v1 import AppTest
        rag.library_add([("B.docx", docx("押金是9000元")), ("C.docx", docx("保修期两年"))])
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        at.multiselect(key="scope").select(self.ids()[0]).select(self.ids()[1]).run()
        rag.library_remove(self.ids()[1])
        at.run()
        self.assertEqual(at.session_state.scope, [])

    def test_named_empty_doc_says_which(self):
        """第三轮审查 N6：点名的那份一个字都没读到时，说清楚是哪份，而不是“没有文档”"""
        real_ask = rag.ask
        rag.ask = lambda system, question, **kw: "SUMMARY 2"
        try:
            reply, hits = rag.answer("第二个文件讲了什么", [{"file": "A.docx", "page": "正文", "text": "押金", "doc": "a"}], None, np.zeros((1, 8)),
                                     docs=[("a", "A.docx"), ("b", "空扫描.pdf")], narrow=lambda ids: ([], None, None, []))
        finally:
            rag.ask = real_ask
        self.assertIn("《空扫描.pdf》里没有读到文字", reply)

    def test_upgrade_failure_warning_stays(self):
        """第三轮审查 N5：升级失败的提示原来只显示一次"""
        from streamlit.testing.v1 import AppTest
        rag.library_add([("A.docx", docx("押金"))])
        old_style(self.store)
        os.remove(os.path.join(self.dir, f"{self.ids()[0]}.docx"))   # 原文件没了，升级会失败
        at = AppTest.from_file(APP, default_timeout=60)
        for _ in range(3):
            at.run()
            self.assertTrue(any("原文件读不到" in w.value for w in at.warning))

    # ---------- 坏文件、坏资料库 ----------
    def test_bad_file_does_not_block_others(self):
        added, failed = rag.library_add([("坏.pdf", b"%PDF-1.4 broken"), ("好.docx", docx("押金"))])
        self.assertEqual(added, ["好.docx"])
        self.assertIn("损坏", failed[0][1])

    def test_build_index_skips_bad_file(self):
        failed = []
        chunks = rag.build_index([("坏.pdf", b"%PDF-1.4 broken"), ("好.docx", docx("押金"))], failed)[0]
        self.assertEqual({c["file"] for c in chunks}, {"好.docx"})
        self.assertEqual(failed[0][0], "坏.pdf")

    def test_password_pdf_message(self):
        w = PdfWriter()
        w.add_blank_page(100, 100)
        w.encrypt("secret")
        b = io.BytesIO()
        w.write(b)
        self.assertIn("密码", rag.library_add([("有密码.pdf", b.getvalue())])[1][0][1])

    def test_embed_failure_is_not_blamed_on_file(self):
        def no_net(texts):
            raise ConnectionError("model download failed")
        rag.embed = no_net
        self.assertIn("文件本身没问题", rag.library_add([("好.docx", docx("押金"))])[1][0][1])

    def test_broken_manifest_is_backed_up(self):
        rag.library_add([("A.docx", docx("押金"))])
        Path(self.dir, "manifest.json").write_text("{broken")
        with self.assertRaises(rag.LibraryBroken):
            self.store.docs()
        self.assertEqual(self.store.docs(), [])
        self.assertTrue(any(f.startswith("manifest.json.bad-") for f in os.listdir(self.dir)))

    def test_broken_manifest_app_shows_warning(self):
        from streamlit.testing.v1 import AppTest
        Path(self.dir, "manifest.json").write_text("{broken")
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("损坏" in w.value for w in at.warning))

    def test_missing_doc_file_is_skipped(self):
        rag.library_add([("A.docx", docx("押金")), ("B.docx", docx("保修"))])
        os.remove(os.path.join(self.dir, f"{self.ids()[1]}.json"))
        problems = []
        chunks = rag.library_load(problems=problems)[0]
        self.assertEqual([c["file"] for c in chunks], ["A.docx"])
        self.assertEqual(problems[0][0], "B.docx")

    def test_replace_failure_keeps_old_doc(self):
        """第二轮审查 N4：旧扫描件重新处理时磁盘满，原来会先删掉旧资料"""
        data = scan_pdf(1)
        rag.OCR_ENABLED = False
        rag.library_add([("扫描.pdf", data)])
        old_style(self.store)
        before = self.store.docs()
        rag.OCR_ENABLED = True
        real = rag.FileStore._write
        rag.FileStore._write = lambda self, name, write: (_ for _ in ()).throw(OSError(28, "No space left on device"))
        try:
            added, failed = rag.library_add([("扫描.pdf", data)])
        finally:
            rag.FileStore._write = real
        self.assertEqual(added, [])
        self.assertIn("保存到资料库失败", failed[0][1])
        self.assertEqual(self.store.docs(), before)

    def test_replace_npy_failure_keeps_chunks_and_vectors_aligned(self):
        """第三轮审查 N2：重新处理时段落写好了、向量没写进去，原来两边对不上，后面所有资料的引用都会错位"""
        rag.OCR_ENABLED = False
        rag.library_add([("混合.pdf", mixed_pdf())])
        old_style(self.store)
        rag.OCR_ENABLED = True
        real = rag.FileStore._write
        def npy_fails(store, name, write):
            if name.endswith(".npy"):
                raise OSError(28, "No space left on device")
            return real(store, name, write)
        rag.FileStore._write = npy_fails
        try:
            self.assertEqual(rag.library_upgrade()[0], [])
        finally:
            rag.FileStore._write = real
        rag.library_add([("B.docx", docx("押金是9000元"))])
        problems = []
        chunks, _, vectors, _, _ = rag.library_load(problems=problems)
        self.assertEqual(len(chunks), len(vectors))
        self.assertEqual(problems, [])
        b = [i for i, c in enumerate(chunks) if c["file"] == "B.docx"][0]   # B 的段落对应的就是 B 自己的向量
        self.assertTrue(np.allclose(vectors[b], fake_embed([chunks[b]["text"]])[0]))

    def test_truncated_vectors_are_detected(self):
        rag.library_add([("A.docx", docx("押金是9000元")), ("B.docx", docx("保修期两年"))])
        np.save(os.path.join(self.dir, f"{self.ids()[0]}.npy"), np.zeros((0, 8)))
        problems = []
        chunks = rag.library_load(problems=problems)[0]
        self.assertEqual([c["file"] for c in chunks], ["B.docx"])
        self.assertIn("对不上", problems[0][1])

    def test_upgrade_twice_leaves_no_stale_files(self):
        rag.OCR_ENABLED = False
        rag.library_add([("混合.pdf", mixed_pdf())])
        old_style(self.store)
        rag.OCR_ENABLED = True
        rag.library_upgrade()
        doc = self.store.docs()[0]
        self.assertEqual(sorted(os.listdir(self.dir)), sorted(["manifest.json", f"{doc['id']}.pdf", *rag.FileStore._data_files(doc)]))
        self.assertTrue(rag.library_remove(doc["id"]))
        self.assertEqual(os.listdir(self.dir), ["manifest.json"])

    # ---------- 第 5 步 OCR ----------
    def test_ocr_one_bad_page_only_loses_that_page(self):
        self.ocr = FakeOCR(fail_on={2})
        pages = rag.read_pages("扫描.pdf", scan_pdf(3))
        self.assertEqual([w for w, _ in pages], ["第 1 页（扫描识别）", "第 2 页", "第 3 页（扫描识别）"])

    def test_watermarked_scan_is_ocred(self):
        """扫描全能王会在角上印一行真文字水印，原来超过 10 个字就不识别了"""
        pages = rag.read_pages("水印扫描.pdf", scan_pdf(2, watermark=True))
        self.assertEqual(self.ocr.calls, 2)
        self.assertTrue(all(w.endswith(rag.OCR_TAG) for w, _ in pages))

    def test_ocr_progress(self):
        """只数要识别的扫描页；每认完一页报告一次"""
        seen = []
        with rag.ocr_progress(lambda name, done, total: seen.append((name, done, total))):
            rag.library_add([("混合.pdf", mixed_pdf()), ("扫描.pdf", scan_pdf(2))])
        self.assertEqual(seen, [("混合.pdf", 1, 1), ("扫描.pdf", 1, 2), ("扫描.pdf", 2, 2)])
        rag.read_pages("扫描.pdf", scan_pdf(1))   # with 外面不再报告
        self.assertEqual(len(seen), 3)

    def test_slide_text_layer_is_kept(self):
        """第三轮审查 N3：PPT 导出的页（整页背景图 + 一行 120 字的真文字），OCR 认出的字不比文字层多，就保留原来准确的文字"""
        slide = "Quarterly results overview: revenue grew in every region while costs stayed flat across the whole year 2025 ok"
        where, text = rag.read_pages("slides.pdf", scan_pdf(1, watermark=slide))[0]
        self.assertEqual(where, "第 1 页")
        self.assertIn("Quarterly results overview", text)

    def test_failed_ocr_page_is_reported_and_retried_once(self):
        """第三轮审查 N4：某一页没认出来要提示，下次打开再试一次（只试一次）"""
        self.ocr = FakeOCR(fail_on={2})
        added, failed = rag.library_add([("扫描.pdf", scan_pdf(3))])
        self.assertEqual(added, ["扫描.pdf"])
        self.assertIn("第 2 页没认出来", failed[0][1])
        doc = self.store.docs()[0]
        self.assertEqual(doc["ocr_failed_pages"], [2])
        self.assertTrue(rag.library_outdated(doc))
        self.ocr = FakeOCR(fail_on={2})             # 再试一次，还是第 2 页失败
        done, failed = rag.library_upgrade()
        self.assertIn("已经重试过", failed[0][2])
        self.assertFalse(rag.library_outdated(self.store.docs()[0]))   # 不会每次打开都重试
        self.assertEqual(len(self.store.docs()), 1)

    def test_failed_ocr_page_fixed_on_retry(self):
        self.ocr = FakeOCR(fail_on={1})
        rag.library_add([("扫描.pdf", scan_pdf(2))])
        self.ocr = FakeOCR()                        # 这次都认出来了
        self.assertEqual(rag.library_upgrade(), (["扫描.pdf"], []))
        doc = self.store.docs()[0]
        self.assertNotIn("ocr_failed_pages", doc)
        self.assertEqual(doc["n_chunks"], len(self.store.load(doc["id"])[0]))
        self.assertIn("第 1 页（扫描识别）", [c["page"] for c in self.store.load(doc["id"])[0]])

    def test_text_pdf_is_not_ocred(self):
        rag.read_pages("论文.pdf", (ROOT / "samples" / "rag_paper_en.pdf").read_bytes())
        self.assertEqual(self.ocr.calls, 0)

    def test_ocr_line_join(self):
        real = rag.load_ocr
        rag.load_ocr = lambda: (lambda img: type("R", (), {"txts": ("今年销售额目标增长5%左右，各门店在日常经营", "中努力争取更好业绩。", "Next line")})())
        try:
            text = rag.read_pages("扫描.pdf", scan_pdf(1))[0][1]
        finally:
            rag.load_ocr = real
        self.assertIn("在日常经营中努力", text)
        self.assertIn("。\nNext line", text)

    def test_ocr_headings_not_joined(self):
        """第 5 步审查第 5 条：标题没写满一行，不能和下一行拼在一起"""
        self.assertEqual(rag.join_ocr_lines(["一、总体要求", "二、主要任务", "（一）基本原则", "第一章 总则", "第一条 为了"]),
                         "一、总体要求\n二、主要任务\n（一）基本原则\n第一章 总则\n第一条 为了")
        box = lambda right: [[0, 0], [right, 0], [right, 10], [0, 10]]
        text = rag.join_ocr_lines(["项目概述", "本项目旨在提升门店的服务", "质量。"], [box(40), box(100), box(30)])
        self.assertEqual(text, "项目概述\n本项目旨在提升门店的服务质量。")

    def test_ocr_wide_line_does_not_break_joining(self):
        """第四轮审查 R1：右上角有一行特别宽的日期，原来正文一处都接不上，日期反而和下一行粘在一起"""
        box = lambda left, right: [[left, 0], [right, 0], [right, 10], [left, 10]]
        lines = ["2026年3月5日", "年度报告的正文第一行写满了要接下", "一行中文被截断了这一行也写满了要接", "继续写完。"]
        text = rag.join_ocr_lines(lines, [box(60, 100), box(0, 90), box(0, 90), box(0, 30)])
        self.assertEqual(text, "2026年3月5日\n年度报告的正文第一行写满了要接下一行中文被截断了这一行也写满了要接继续写完。")

    def test_ocr_right_aligned_signature_not_joined(self):
        """靠右对齐的署名右边也到了边界，但不能和下一行接上（“董事长张三各位股东：”）"""
        box = lambda left, right: [[left, 0], [right, 0], [right, 10], [left, 10]]
        lines = ["正文第一行写满了要接", "正文第二行写满了要接", "董事长张三", "各位股东：", "现在我代表董事会作报告。"]
        text = rag.join_ocr_lines(lines, [box(0, 100), box(0, 100), box(70, 100), box(0, 15), box(5, 60)])
        self.assertIn("董事长张三\n各位股东：", text)

    def test_ocr_two_columns(self):
        """两栏排版：左栏的行到不了右栏的边，原来左栏一句都不接"""
        box = lambda left, right: [[left, 0], [right, 0], [right, 10], [left, 10]]
        lines = ["左栏第一行写满了要接", "左栏第二行写满了要接", "左栏写完。", "右栏第一行写满了要接", "右栏第二行写满了要接", "右栏写完。"]
        boxes = [box(0, 45), box(0, 45), box(0, 20), box(55, 100), box(55, 100), box(55, 70)]
        self.assertEqual(rag.join_ocr_lines(lines, boxes), "左栏第一行写满了要接左栏第二行写满了要接左栏写完。\n右栏第一行写满了要接右栏第二行写满了要接右栏写完。")

    # ---------- 处理版本号 ----------
    def test_upgrade_mixed_pdf(self):
        """第 5 步以前录入的混合 PDF（文字页 + 扫描页），升级后补识别扫描页，编号、录入时间不变"""
        rag.OCR_ENABLED = False
        rag.library_add([("混合.pdf", mixed_pdf())])
        old_style(self.store)
        before = self.store.docs()[0]
        rag.OCR_ENABLED = True
        self.assertTrue(rag.library_outdated(before))
        done, failed = rag.library_upgrade()
        after = self.store.docs()[0]
        self.assertEqual((done, failed), (["混合.pdf"], []))
        self.assertEqual((after["id"], after["added"]), (before["id"], before["added"]))
        self.assertIn("第 2 页（扫描识别）", [c["page"] for c in self.store.load(after["id"])[0]])
        self.assertFalse(rag.library_outdated(after))

    def test_upgrade_reuses_vectors_when_unchanged(self):
        rag.library_add([("A.docx", docx("押金是9000元"))])
        old_style(self.store)
        calls = []
        rag.embed = lambda texts: calls.append(len(texts)) or fake_embed(texts)
        rag.library_upgrade()
        self.assertEqual(calls, [])


class PickDocsTest(unittest.TestCase):
    """第 5b 步 + 第三轮审查 N1：本地先判断问题是不是真的在点名资料，是的话才把资料清单发给 DeepSeek"""
    DOCS = [(n, n) for n in ["NIST_AI_RMF_1.0_en.pdf", "2025年行业数据表_人口能源交通.xlsx", "2025年各国货物进出口.xlsx",
                             "智能制造产业发展规划2026-2028.pdf", "test.pdf", "rental_contract.docx"]]

    def test_named(self):
        cases = {"第一个档案主要是讲什么": "all", "第1份讲了什么": "all", "最后一个讲了什么": "all", "第二个和第三个里，2025年出口总额是多少？": "all",
                 "那篇英文论文的核心观点是什么？": "all", "最后那份文件讲了什么": "all", "进出口那张表里对美国出口多少？": [3],
                 "NIST那份文档主要讲了什么": [1], "AI RMF 的四个核心功能是什么": [1], "讲智能制造的那个文件有哪些重点？": [4],
                 "rental contract 里押金多少": [6]}
        for q, want in cases.items():
            self.assertEqual(rag.doc_mentions(q, self.DOCS), want, q)

    def test_not_named(self):
        for q in ["押金是多少？", "2025年末全国人口是多少？", "人工智能有哪些风险？", "一季度哪个产品卖得最少？",
                  "这个产品一季度卖了多少台？", "那个时候的人口是多少？", "销量前3个产品是哪些？", "合同里第二个条件是什么？",
                  "人工智能如何推动创新发展？", "中国对美国出口多少？", "What is the content about attention?", "test 一下，你好"]:
            self.assertIsNone(rag.doc_mentions(q, self.DOCS), q)

    def test_name_words_skip_suffix(self):
        eng, zh = rag.name_words("报告.docx（11-03 19:56 录入）")
        self.assertNotIn("录入", zh)
        self.assertEqual(eng, set())

    def test_parse_pick(self):
        self.assertEqual(rag._parse_pick("DOC 1-3", {1, 2, 3, 4}), ("DOC", [1, 2, 3]))
        self.assertEqual(rag._parse_pick("SUMMARY 2、4", {1, 2, 3, 4}), ("SUMMARY", [2, 4]))
        self.assertEqual(rag._parse_pick("DOC（第2份）", {2}), ("DOC", [2]))
        self.assertEqual(rag._parse_pick("DOC 5", {1, 2}), ("DOC", []))   # 清单里没列出的编号不认
        self.assertEqual(rag._parse_pick("CHAT", set()), ("CHAT", []))


if __name__ == "__main__":
    unittest.main()
