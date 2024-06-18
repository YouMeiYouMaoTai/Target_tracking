from PIL import Image

# 打开图片
img = Image.open('D:/Desktop/1.png')

# 重置尺寸
img_resized = img.resize((1400, 440))

# 保存图片
img_resized.save('D:/Desktop/2.png')
