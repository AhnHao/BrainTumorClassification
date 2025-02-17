import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Đường dẫn đến tập dữ liệu
train_dir = "dataset/Training"
test_dir = "dataset/Testing"

# Đếm số lượng ảnh trong mỗi lớp
def count_images(directory):
    classes = os.listdir(directory)
    class_counts = {cls: len(os.listdir(os.path.join(directory, cls))) for cls in classes}
    return class_counts

# Đếm số ảnh trong tập train & test
train_counts = count_images(train_dir)
test_counts = count_images(test_dir)

# Tổng số ảnh trong mỗi tập
total_train = sum(train_counts.values())
total_test = sum(test_counts.values())

# Vẽ biểu đồ phân bố số lượng ảnh trong tập huấn luyện
def plot_image_distribution(data_counts, title):
    plt.figure(figsize=(8, 6))
    bars = plt.bar(data_counts.keys(), data_counts.values(), color=['blue', 'red', 'green', 'yellow'])
    # Thêm nhãn số lượng ảnh lên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, str(height), ha='center', va='bottom', fontsize=12, color='black')
    plt.ylim(0, max(data_counts.values()) * 1.1)
    plt.xlabel("Lớp")
    plt.ylabel("Số lượng ảnh")
    plt.title("Phân bố số lượng ảnh trong tập " + title)
    plt.savefig(os.path.join("images", title))
    plt.show()

plot_image_distribution(train_counts, "huấn luyện")
plot_image_distribution(test_counts, "kiểm tra")

# Vẽ biểu đồ tròn (tỷ lệ giữa tập train & test)
def plot_pie_chart(total_train, total_test, total_val=0):
    plt.figure(figsize=(6, 6))
    if total_val > 0:
        labels = ['Train', 'Test' , 'Validation']
        sizes = [total_train, total_test, total_val]
        colors = ['blue', 'red', 'yellow']
        title = "pie_chart_3_dataset.png"
    else:
        labels = ['Train', 'Test']
        sizes = [total_train, total_test]
        colors = ['blue', 'red']
        title = "pie_chart.png"
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title("Tỷ lệ số lượng ảnh giữa các tập dữ liệu")
    plt.savefig(os.path.join("images", title))
    plt.show()

plot_pie_chart(total_train, total_test)

# Hiển thị ảnh
def show_random_images(directory, num_images=16):
    classes = os.listdir(directory)
    plt.figure(figsize=(12, 6))

    for i in range(num_images):
        cls = random.choice(classes)  # Chọn ngẫu nhiên lớp
        class_folder = os.path.join(directory, cls)
        img_name = random.choice(os.listdir(class_folder))  # Chọn ngẫu nhiên ảnh
        img_path = os.path.join(class_folder, img_name)

        image = Image.open(img_path)

        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        plt.title(cls)
        plt.axis("off")
    plt.savefig(os.path.join("images", "show_random_images.png"))
    plt.show()

# Hiển thị ảnh từ tập huấn luyện
show_random_images(train_dir)

