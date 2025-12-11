from PIL import Image, ImageDraw, ImageFont

# ================================
# 请在这里填写西夏文字体文件路径（例如："./TangutFont.ttf"）
FONT_PATH = "C:\project\THOCR\enlargeDataset\Tangut N4694 V3.10.ttf"   # TODO: 填入字体路径
# ================================

# 输出图片大小
IMG_SIZE = (100, 100)

# 字符字号（可调整）
FONT_SIZE = 72

# 需要输出的 Tangut Unicode 范围
# 西夏文 Unicode 块：U+17000 - U+187F7
START = 0x17000
END   = 0x187F7

def generate_tangut_images():
    if not FONT_PATH:
        print("请先在脚本中填入字体文件路径！")
        return

    # 加载字体
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception as e:
        print("字体加载失败：", e)
        return

    for codepoint in range(START, END + 1):
        char = chr(codepoint)
        filename = f"U+{codepoint:04X}.png"

        # 创建100x100白色背景图
        img = Image.new("RGB", IMG_SIZE, color="white")
        draw = ImageDraw.Draw(img)

        # 计算文字居中位置
        text_width, text_height = draw.textsize(char, font=font)
        x = (IMG_SIZE[0] - text_width) / 2
        y = (IMG_SIZE[1] - text_height) / 2

        # 绘制字符
        draw.text((x, y), char, font=font, fill="black")

        # 保存图片
        img.save(filename)
        print("生成：", filename)

    print("全部字符生成完毕！")

if __name__ == "__main__":
    generate_tangut_images()
