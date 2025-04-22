
import re
import matplotlib.pyplot as plt


log_file_path = r"log\log_reddit_link_pred_egcn_h_20250419220401_r0.log"  


with open(log_file_path, "r", encoding="utf-8") as f:
    log_data = f.read()


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


plt.figure(figsize=(10, 6))
plt.plot(valid_epochs, valid_mrrs, marker='o', label='Valid MRR')
plt.plot(valid_epochs, valid_maps, marker='s', label='Valid MAP')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("MAP and MRR over Validation Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("valid_map_mrr_plot.png")  
plt.show()
