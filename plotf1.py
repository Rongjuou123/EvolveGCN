import re
import matplotlib.pyplot as plt


log_file_path = r'E:\Study\Umich\2024-2025\EECS553\Final_Project\log_reddit_edge_cls_egcn_h_20250413185135_r0.log'


with open(log_file_path, "r", encoding="utf-8") as f:
    log_data = f.read()


epoch_pattern = re.compile(r"TRAIN epoch (\d+)")
train_f1_pattern = re.compile(r"TRAIN measures microavg - precision [\d.]+ - recall [\d.]+ - f1 ([\d.]+)")
valid_f1_pattern = re.compile(r"VALID measures microavg - precision [\d.]+ - recall [\d.]+ - f1 ([\d.]+)")


epochs = []
train_f1s = []
valid_f1s = []


epoch_matches = epoch_pattern.findall(log_data)
train_f1_matches = train_f1_pattern.findall(log_data)
valid_f1_matches = valid_f1_pattern.findall(log_data)

for i in range(min(len(epoch_matches), len(train_f1_matches), len(valid_f1_matches))):
    epochs.append(int(epoch_matches[i]))
    train_f1s.append(float(train_f1_matches[i]))
    valid_f1s.append(float(valid_f1_matches[i]))


plt.figure(figsize=(10, 6))
plt.plot(epochs, train_f1s, label="Train F1", marker='o')
plt.plot(epochs, valid_f1s, label="Valid F1", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Microavg F1 Score")
plt.title("Train vs Valid Microavg F1 Score per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("f1_plot.png")  
plt.show()
