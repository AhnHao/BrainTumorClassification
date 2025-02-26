import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import BrainTumorCNN
from dataset import test_loader
from sklearn.metrics import classification_report

# Khởi tạo thiết bị (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load lại mô hình đã lưu
model = BrainTumorCNN(4)  # Số lớp output (4 classes)
model.load_state_dict(torch.load("model.pth"))
model.to(device)
model.eval()  # Đưa model về chế độ đánh giá

# Lấy class labels từ dataset
class_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]  # Thay đổi theo dataset của bạn

# Lấy một batch từ test_loader
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Đưa ảnh và nhãn vào device
images, labels = images.to(device), labels.to(device)

# Dự đoán
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# Hiển thị 10 ảnh đầu tiên
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    img = images[i].cpu().numpy().transpose((1, 2, 0))  # Chuyển đổi định dạng ảnh
    img = np.clip(img, 0, 1)  # Chuẩn hóa giá trị pixel

    # Kiểm tra dự đoán đúng hay sai
    color = "green" if preds[i] == labels[i] else "red"

    axes[i].imshow(img)
    axes[i].set_title(
        f"True: {class_labels[labels[i].item()]}\nPred: {class_labels[preds[i].item()]}",
        color=color, fontsize=10
    )
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join("images", "predict_result.png"))
plt.show()

# Dự đoán toàn bộ tập test
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Tạo classification report
report = classification_report(true_labels, pred_labels, target_names=class_labels)

# Lưu báo cáo vào file
with open("classification_report.txt", "w") as f:
    f.write(report)

print(report)
print("Đã lưu classification report vào classification_report.txt")