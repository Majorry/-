import os
import time
from PIL import Image

def delete_small_images(dataset_dir="Dataset", min_area=128*128, output_file="deleted_images.txt"):
    """删除Dataset中面积小于min_area的图片，并将删除的图片路径保存到txt文件"""
    deleted_images = []
    failed_images = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(root, file)
                try:
                    # 强制加载图片信息后立即关闭，避免文件被锁
                    img = Image.open(img_path)
                    width, height = img.size
                    img.close()

                    area = width * height
                    if area < min_area:
                        # 尝试删除文件（增加重试机制）
                        for attempt in range(3):
                            try:
                                os.remove(img_path)
                                deleted_images.append(img_path)
                                print(f"🗑️ 已删除: {img_path} ({width}×{height} = {area})")
                                break
                            except PermissionError:
                                print(f"⚠️ 文件被占用，重试 {attempt+1}/3: {img_path}")
                                time.sleep(0.5)
                        else:
                            failed_images.append(img_path)

                except Exception as e:
                    print(f"⚠️ 无法读取图片: {img_path}，错误: {e}")
                    failed_images.append(img_path)

    # 写入删除记录
    with open(output_file, "w", encoding="utf-8") as f:
        for path in deleted_images:
            f.write(path + "\n")

    # 写入删除失败记录
    if failed_images:
        failed_file = output_file.replace(".txt", "_failed.txt")
        with open(failed_file, "w", encoding="utf-8") as f:
            for path in failed_images:
                f.write(path + "\n")
        print(f"\n⚠️ 有 {len(failed_images)} 张图片无法删除或读取，路径已保存至：{failed_file}")

    print("\n✅ 任务完成！")
    print(f"共删除 {len(deleted_images)} 张面积小于 {min_area} 的图片。")
    print(f"📄 删除记录已保存至：{os.path.abspath(output_file)}")


if __name__ == "__main__":
    delete_small_images("Dataset", min_area=128*128, output_file="deleted_images.txt")
