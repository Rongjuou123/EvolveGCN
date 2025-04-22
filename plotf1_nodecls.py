import re
import matplotlib.pyplot as plt

# === 设置你的日志文件路径 ===
log_file_path = r"E:\Study\Umich\2024-2025\EECS553\Final_Project\EvolveGCN\log\log_reddit_node_cls_egcn_h_20250421194814_r0.log"  # 修改为实际路径

# 读取日志文件
with open(log_file_path, "r", encoding="utf-8") as f:
    log_data = f.read()

# 正则匹配
epoch_pattern = re.compile(r"TRAIN epoch (\d+)")
f1_class_pattern = re.compile(r"TRAIN measures@1000 for class (\d+) - precision [\d.]+ - recall [\d.]+ - f1 ([\d.]+)")

# 初始化存储结构：每类一个 F1 分数列表
class_f1_dict = {i: [] for i in range(7)}

# 获取每个 epoch 的日志块（跳过开头无用段）
epoch_blocks = re.split(r"################ TRAIN epoch \d+ ###################", log_data)

# 提取前 10 个 epoch 的每类 F1@1000
for epoch_id, block in enumerate(epoch_blocks[1:]):  # 第一块是非 epoch 内容
    # if epoch_id >= 10:
    #     break
    matches = re.findall(f1_class_pattern, block)
    f1_this_epoch = {int(cls): float(f1) for cls, f1 in matches}
    for cls in range(7):
        class_f1_dict[cls].append(f1_this_epoch.get(cls, 0.0))

# 绘图
plt.figure(figsize=(12, 8))
for cls in range(7):
    plt.plot(range(len(class_f1_dict[cls])), class_f1_dict[cls], marker='o', label=f'Class {cls}')
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score per Class")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("node_class_f1_plot.png")  # 保存图片
plt.show()
