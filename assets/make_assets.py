"""生成「溯光」的 logo 和头像（先画 1024 大，再缩小，边缘更平滑）

2.0 界面改版：图标「溯」—— 深蓝底上，一道光沿弧线绕回起点，中间一颗星。
光从暗到亮（越往回走越亮），在白天、黑夜两种界面上都清楚。
运行：在 assets 文件夹里 `../.venv/bin/python make_assets.py`
"""
import math

from PIL import Image, ImageDraw, ImageFilter

S = 1024                      # 画布大小（之后缩小到 256 / 128）
U = S / 100                   # 设计稿按 100×100 画，1 单位 = U 像素
NAVY_TOP, NAVY_BOTTOM = (31, 46, 94), (13, 21, 48)
AMBER, LIGHT = (245, 185, 66), (255, 227, 160)


def gradient(c1, c2):
    """左上 → 右下的对角渐变"""
    img = Image.new("RGB", (S, S))
    px = img.load()
    for y in range(S):
        for x in range(S):
            t = (x + y) / (2 * (S - 1))
            px[x, y] = tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
    return img


def star(d, cx, cy, r, inner, fill):
    """四角星（光点）"""
    pts = []
    for k in range(8):
        ang = math.pi / 4 * k - math.pi / 2
        rr = r if k % 2 == 0 else inner
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    d.polygon(pts, fill=fill)


def mark(scale=1.0, shape="square"):
    """logo 主体：深蓝底 + 一道由暗到亮的弧形光（末端带箭头，表示「溯回」）+ 中间的星"""
    bg = gradient(NAVY_TOP, NAVY_BOTTOM).convert("RGBA")
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    if shape == "square":
        md.rounded_rectangle((0, 0, S - 1, S - 1), radius=int(24 * U), fill=255)
    else:
        md.ellipse((0, 0, S - 1, S - 1), fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(bg, (0, 0), mask)

    c = S / 2
    P = lambda v: c + (v * U - c) * scale        # 设计稿坐标 → 像素（圆形头像整体缩小一点）
    cx, cy, r, w = P(50), P(52), 30 * U * scale, 7.5 * U * scale

    # 弧形光：从右下（28°）顺时针绕到右上（312°）。左边暗、右边亮，像一道光绕回来
    arc = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ad = ImageDraw.Draw(arc)
    a0, a1, steps = 28.0, 312.0, 160
    box = (cx - r, cy - r, cx + r, cy + r)
    for i in range(steps):
        mid = math.radians(a0 + (a1 - a0) * (i + .5) / steps)
        t = (math.cos(mid) + 1) / 2                     # 0 = 最左，1 = 最右
        alpha = .6 * t / .5 if t < .5 else .6 + .4 * (t - .5) / .5
        col = tuple(int(a + (b - a) * t) for a, b in zip(AMBER, LIGHT))
        ad.arc(box, a0 + (a1 - a0) * i / steps, a0 + (a1 - a0) * (i + 1) / steps + .6, fill=col + (int(255 * alpha),), width=int(w))
    for ang in (a0, a1):                                 # 两端画成圆头
        ex, ey = cx + r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang))
        ad.ellipse((ex - w / 2, ey - w / 2, ex + w / 2, ey + w / 2), fill=LIGHT + (255,))
    out.alpha_composite(arc)

    # 箭头：表示「回到起点」
    d = ImageDraw.Draw(out)
    pts = [(P(62), P(22)), (P(74), P(29)), (P(66), P(40))]
    aw = int(6 * U * scale)
    d.line(pts, fill=LIGHT + (255,), width=aw, joint="curve")
    for x, y in (pts[0], pts[2]):
        d.ellipse((x - aw / 2, y - aw / 2, x + aw / 2, y + aw / 2), fill=LIGHT + (255,))

    # 中间的星 + 光晕
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gr = 17 * U * scale
    ImageDraw.Draw(glow).ellipse((cx - gr, cy - gr, cx + gr, cy + gr), fill=AMBER + (110,))
    glow = glow.filter(ImageFilter.GaussianBlur(9 * U * scale))
    out.alpha_composite(glow)
    star(ImageDraw.Draw(out), cx, cy, 15.5 * U * scale, 4 * U * scale, LIGHT + (255,))
    # 圆角之外的部分裁掉（光晕可能溢出）
    final = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    final.paste(out, (0, 0), Image.composite(out.split()[3], Image.new("L", (S, S), 0), mask))
    return final


def user_avatar():
    """用户头像：浅琥珀色圆底 + 琥珀色简笔人像（白天、黑夜底色上都看得清）"""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, S - 1, S - 1), fill=(245, 185, 66, 70))
    person = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    pd = ImageDraw.Draw(person)
    col = (226, 152, 30, 255)
    pd.ellipse((S / 2 - 135, 250, S / 2 + 135, 520), fill=col)       # 头
    pd.ellipse((S / 2 - 280, 600, S / 2 + 280, 1180), fill=col)      # 肩膀
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, S - 1, S - 1), fill=255)
    clipped = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    clipped.paste(person, (0, 0), Image.composite(person.split()[3], Image.new("L", (S, S), 0), mask))
    img.alpha_composite(clipped)
    return img


def save(img, name, size):
    img.resize((size, size), Image.LANCZOS).save(f"{name}", optimize=True)


logo = mark(1.0, "square")
ai = mark(0.92, "circle")
user = user_avatar()
save(logo, "logo.png", 256)
save(ai, "avatar_ai.png", 128)
save(user, "avatar_user.png", 128)

# 预览图：同一组图标分别放在白天、黑夜底色上（大图 + 侧栏大小 + 标签页大小）
W, H = 1000, 420
prev = Image.new("RGBA", (W * 2, H * 2), (250, 250, 247, 255))
ImageDraw.Draw(prev).rectangle((W, 0, W * 2, H * 2), fill=(10, 16, 32, 255))
for ox in (0, W):
    for img, x, y, s in [(logo, 90, 150, 440), (ai, 600, 210, 200), (user, 840, 210, 200), (logo, 600, 520, 68), (logo, 700, 534, 40)]:
        prev.alpha_composite(img.resize((s, s), Image.LANCZOS), (ox + x, y))
prev = prev.resize((W, H), Image.LANCZOS)
prev.save("avatar_preview.png")
print("done")
