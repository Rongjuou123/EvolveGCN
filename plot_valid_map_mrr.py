
import re
import matplotlib.pyplot as plt

# 设置你的日志文件路径
log_file_path = r"log\log_reddit_link_pred_egcn_h_20250419220401_r0.log"  # 修改为实际路径

# 读取日志文件
with open(log_file_path, "r", encoding="utf-8") as f:
    log_data = f.read()

# 提取 VALID 每个 epoch 的 MAP 和 MRR
valid_map_mrr_pattern = re.compile(r"################ VALID epoch (\d+) ###################.*?mean MRR ([\d.]+) - mean MAP ([\d.]+)", re.DOTALL)

valid_epochs = []
valid_mrrs = []
valid_maps = []

for match in re.finditer(valid_map_mrr_pattern, log_data):
    epoch = int(match.group(1))
    mrr = float(match.group(2))
    map_ = float(match.group(3))
    valid_epochs.append(epoch)
    valid_mrrs.append(mrr)
    valid_maps.append(map_)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(valid_epochs, valid_mrrs, marker='o', label='Valid MRR')
plt.plot(valid_epochs, valid_maps, marker='s', label='Valid MAP')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("MAP and MRR over Validation Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("valid_map_mrr_plot.png")  # 保存图像
plt.show()
