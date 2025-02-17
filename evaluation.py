import os
import json
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file JSON
with open("training_results.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

# Tạo figure với kích thước lớn hơn
plt.figure(figsize=(12, 5))

# Biểu đồ Train Loss vs Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", linestyle="--")
plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="s", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

# Biểu đồ Train Accuracy vs Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o", linestyle="--")
plt.plot(epochs, history["val_acc"], label="Validation Accuracy", marker="s", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

# Hiển thị biểu đồ
plt.tight_layout()
plt.savefig(os.path.join("images", "result_chart.png"))
plt.show()
