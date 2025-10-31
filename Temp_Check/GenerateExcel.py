import os
from PIL import Image
import pandas as pd

def process_subfolder(subfolder_path, excel_dir):
    """处理单个子文件夹，统计图片尺寸并导出Excel"""
    records = []  # 存储每张图片的路径和尺寸

    for root, _, files in os.walk(subfolder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        size_str = f"{width}×{height}"
                        records.append({
                            "图片路径": img_path,
                            "图片尺寸": size_str
                        })
                except Exception as e:
                    print(f"⚠️ 无法读取图片: {img_path}，错误: {e}")

    if not records:
        print(f"⚠️ 子文件夹 {subfolder_path} 中未找到 .jpg 图片。")
        return

    # 创建 DataFrame
    df = pd.DataFrame(records)

    # 统计相同尺寸的图片数量
    size_counts = df["图片尺寸"].value_counts().reset_index()
    size_counts.columns = ["图片尺寸", "数量"]

    # 添加分隔行
    df = pd.concat([df, pd.DataFrame([{"图片路径": "", "图片尺寸": ""}]), size_counts], ignore_index=True)

    # Excel 文件名与路径
    folder_name = os.path.basename(subfolder_path.rstrip("/\\"))
    excel_path = os.path.join(excel_dir, f"{folder_name}.xlsx")

    # 导出 Excel
    df.to_excel(excel_path, index=False)
    print(f"✅ 已生成 {excel_path} ，共统计 {len(records)} 张图片。")

def main(dataset_dir="Dataset", excel_dir="Excel"):
    """主函数，遍历Dataset下所有子文件夹"""
    # 如果 Excel 目录不存在，则创建
    os.makedirs(excel_dir, exist_ok=True)

    # 获取 Dataset 下所有子文件夹
    subfolders = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
                  if os.path.isdir(os.path.join(dataset_dir, d))]

    if not subfolders:
        print("⚠️ 未在 Dataset 下找到子文件夹。")
        return

    # 遍历并处理每个子文件夹
    for subfolder in subfolders:
        process_subfolder(subfolder, excel_dir)

if __name__ == "__main__":
    main("Dataset", "Excel")
