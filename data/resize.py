import os
from PIL import Image
import constants

def fileWalk(directory, destPath):
    os.makedirs(destPath, exist_ok=True)

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}, skipping.")
        return

    # 遍历当前类别目录下的所有文件
    for file in os.listdir(directory):
        # 兼容常见图片扩展名（大写或小写）
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        src_file_path = os.path.join(directory, file)
        if not os.path.isfile(src_file_path):
            continue

        try:
            with Image.open(src_file_path) as pic:
                # 统一转换为 RGB 格式（防止包含 RGBA 或灰度图像）
                pic = pic.convert('RGB')
                width, height = pic.size

                # 如果高度大于宽度，旋转 90 度调整为横向
                if height > width:
                    pic = pic.transpose(Image.Transpose.ROTATE_90)

                # 双三次插值缩放到目标尺寸 (DIM2 为宽，DIM1 为高)
                target_size = (constants.DIM2, constants.DIM1)
                pic_resized = pic.resize(target_size, resample=Image.Resampling.BICUBIC)

                dest_file_path = os.path.join(destPath, file)
                pic_resized.save(dest_file_path, quality=95)
        except Exception as e:
            print(f"Error processing {src_file_path}: {e}")

def main():
    prepath = os.path.join(os.getcwd(), 'dataset-orgaized')
    destPath = os.path.join(os.getcwd(), 'dataset-resized')

    categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

    for category in categories:
        src_dir = os.path.join(prepath, category)
        target_dir = os.path.join(destPath, category)
        print(f"Processing category: {category}...")
        fileWalk(src_dir, target_dir)

    print("Image resizing completed.")

if __name__ == '__main__':
    main()