"""生成一份「自制扫描件」：只有图片、没有文字层的 PDF，用来测 OCR（内容为虚构）

做法：先把一页虚构的书店通知画成图片，再加上纸张底色、噪点和模糊，模拟扫描仪扫出来的效果，
最后存成只含图片的 PDF。溯光读这种 PDF 时抽不出文字，会自动交给 OCR 识别。
运行：在 samples 文件夹里 `../.venv/bin/python make_scan_sample.py`（需要 Mac 自带的黑体字体）
"""
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1240, 1754          # A4，150 dpi
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
random.seed(7)

TITLE = "溯光书店 2026 年度工作安排"
SUB = "内部通知 · 2026 年 1 月 8 日"
BODY = [
    ("h", "一、全年目标"),
    ("p", "1. 全年销售额目标为 1260 万元，比上年增长 8%，其中线上小程序销售额不低于 300 万元。"),
    ("p", "2. 会员人数力争达到 5 万人，其中新增会员 1.2 万人，老会员复购率保持在 60% 以上。"),
    ("p", "3. 计划新开门店 3 家，分别位于杭州、苏州和南京，每家面积在 400 平方米左右。"),
    ("h", "二、重点工作"),
    ("p", "1. 每月举办 2 场读书会，全年不少于 24 场，每场邀请一位作者或译者到店分享。"),
    ("p", "2. 儿童绘本区面积扩大到 120 平方米，周末增加亲子共读时段。"),
    ("p", "3. 线上小程序订单占比提高到 30% 以上，同城配送时间缩短到 2 小时以内。"),
    ("h", "三、时间安排"),
    ("p", "第一季度完成杭州门店装修，第二季度开业；苏州、南京门店在第四季度前开业。各门店每月 5 日前报送上月经营数据。"),
]
SIGN = "溯光书店管理办公室"


def wrap(text, font, width, draw):
    """按像素宽度断行（中文不按词断，模拟真实文件里被截断在行尾的词）"""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + ([cur] if cur else [])


page = Image.new("L", (W, H), 255)
d = ImageDraw.Draw(page)
f_title, f_sub, f_h, f_p = (ImageFont.truetype(FONT, s) for s in (46, 24, 32, 29))
y = 150
d.text((W / 2, y), TITLE, font=f_title, fill=20, anchor="mm")
y += 60
d.text((W / 2, y), SUB, font=f_sub, fill=80, anchor="mm")
y += 80
left, width = 130, W - 260
for kind, text in BODY:
    if kind == "h":
        y += 18
        d.text((left, y), text, font=f_h, fill=20)
        y += 58
    else:
        for line in wrap(text, f_p, width, d):
            d.text((left, y), line, font=f_p, fill=30)
            y += 50
        y += 8
y += 40
d.text((W - 130, y), SIGN, font=f_p, fill=30, anchor="ra")

# 扫描效果：纸张底色、噪点、模糊。
# 不加倾斜：试过整页歪 0.3° 以上，「1. 每月举办 2 场读书会……」这一行会被认成乱码（已记进已知问题），
# 评测要测的是「扫描件能不能答对」，所以样本保持端正
noise = Image.effect_noise((W, H), 18)
page = Image.blend(page, noise, 0.08).filter(ImageFilter.GaussianBlur(0.6))
paper = Image.new("RGB", (W, H), (244, 240, 230))
scan = Image.composite(page.convert("RGB"), paper, Image.eval(page, lambda v: 255 - v))
scan = Image.blend(paper, scan, 0.92)
scan.save("scan_notice.pdf", "PDF", resolution=150)
print("写好 scan_notice.pdf")
