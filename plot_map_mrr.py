
import re
import matplotlib.pyplot as plt

# 设置你的日志文件路径
log_file_path = r"log\log_reddit_link_pred_egcn_h_20250419220401_r0.log"  # 修改为实际路径

# 读取日志文件
with open(log_file_path, "r", encoding="utf-8") as f:
    log_data = f.read()

# 提取 MAP 和 MRR
epochs = []
mrrs = []
maps = []

for match in re.finditer(r"################ TRAIN epoch (\d+) ###################.*?mean MRR ([\d.]+) - mean MAP ([\d.]+)", log_data, re.DOTALL):
    epoch = int(match.group(1))
    mrr = float(match.group(2))
    map_ = float(match.group(3))
    epochs.append(epoch)
    mrrs.append(mrr)
    maps.append(map_)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, mrrs, marker='o', label='MRR')
plt.plot(epochs, maps, marker='s', label='MAP')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("MAP and MRR over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("map_mrr_plot.png")  # 保存图像
plt.show()
