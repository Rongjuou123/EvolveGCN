
import re
import matplotlib.pyplot as plt


log_file_path = r"log\log_reddit_link_pred_egcn_h_20250419220401_r0.log"  


with open(log_file_path, "r", encoding="utf-8") as f:
    log_data = f.read()


test_map_mrr_pattern = re.compile(r"TEST batch \d+ / \d+ - .*?MRR\s+([\d.]+)\s+- partial MAP\s+([\d.]+)")

test_mrrs = []
test_maps = []
test_steps = []

for idx, match in enumerate(re.finditer(test_map_mrr_pattern, log_data)):
    mrr = float(match.group(1))
    map_ = float(match.group(2))
    test_steps.append(idx + 1)
    test_mrrs.append(mrr)
    test_maps.append(map_)


plt.figure(figsize=(10, 6))
plt.plot(test_steps, test_mrrs, marker='o', label='Test MRR')
plt.plot(test_steps, test_maps, marker='s', label='Test MAP')
plt.xlabel("Test Batch")
plt.ylabel("Score")
plt.title("MAP and MRR over Test Batches")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_map_mrr_plot.png")  
plt.show()
