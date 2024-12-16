# Here is the example function of perturbation add in image

# 图像向左旋转90度

def rotate_left(input_path, output_path):
    """
    将图像向左旋转 90 度并保存到指定路径。
    
    参数:
    input_path (str): 原始图像的路径。
    output_path (str): 保存旋转后图像的路径。
    """
    # 打开图像
    with Image.open(input_path) as img:
        # 将图像向左旋转 90 度
        rotated_img = img.rotate(90, expand=True)
        # 保存图像到新的路径
        rotated_img.save(output_path)


每个函数的命名以 perturbation的形式，包含两个参数，及输入路径和输出路径

每个function需要加上注释简短说明这个函数的pertubation type 中文，以下面的形式，不用在意是否准确，后续我会统一改成英文

“”“
perturbation 简短说明
”“”


需要实现的perturbation 包括：
1. Image corruptions   paper “Benchmarking Neural Network Robustness to Common Corruptions and Surface Variations ” 中包含的15种
2. 滤镜  ##看是否能调一些美颜相机的api
3. 旋转 ： 向左 90，  向右 90， 180 
4. 左右翻转
5. 背景虚化
5. 加水印（jiawen）

#每种perturbation 都是一个function


