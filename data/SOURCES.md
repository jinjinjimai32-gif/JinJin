# 数据来源

四张 PNG 文件直接取自 scikit-image 的 v0.19.3 标签，保留原文件，未把生成的噪声当成原始数据。

源目录：https://github.com/scikit-image/scikit-image/tree/v0.19.3/skimage/data

文件：camera.png、coins.png、moon.png、page.png。

官方说明：https://scikit-image.org/docs/stable/api/skimage.data.html

camera 为摄影师 Stefan van der Walt 公开到公有领域的照片；coins 为希腊硬币示例，moon 为月面示例，page 为扫描文字示例。不同样例的来源说明以 scikit-image 文档为准，不将本项目代码许可扩展到第三方图片。

各文件的 SHA-256、原尺寸及处理后尺寸见 results/summary.json。实验先将长边缩小至 256 像素并保持宽高比，再加噪；没有训练、验证或模型拟合环节。
