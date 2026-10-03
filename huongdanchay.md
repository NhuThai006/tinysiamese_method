# Hướng Dẫn Chạy TinySiamese Network

Dự án này sử dụng mạng TinySiamese (kết hợp với **ACGAN**) để phân loại/xác thực vân tay. Thay vì đưa trực tiếp ảnh vào mạng Siamese, nó trích xuất đặc trưng (feature vectors) từ ảnh bằng Discriminator của GAN trước, sau đó đưa các vector này vào mạng TinySiamese để tính độ tương đồng.

## Yêu cầu môi trường
Đảm bảo bạn đã cài đặt các thư viện cơ bản của Python (đặc biệt là PyTorch, OpenCV, NumPy, Pandas, Scikit-learn):
```bash
pip install torch torchvision opencv-python numpy pandas scikit-learn pillow
```

## Các bước chạy (Train & Test)

### Bước 0: Train ACGAN (thay cho VGG16)
Chạy `GANTrain.py` để train Generator + Discriminator trên 4 lớp vân tay (Arch, LeftLoop, RightLoop, Whorl).
Discriminator sẽ được lưu tại `model/GANFVC.h5` và dùng để trích xuất vector đặc trưng 4096 chiều.
**Lệnh:**
```bash
python GANTrain.py
```

### Bước 1: Trích xuất đặc trưng tập Train
Chạy file `FeatureVectorsTrain.py` để trích xuất đặc trưng từ ảnh tập train.
File này sẽ sử dụng Discriminator GAN (được lưu trong thư mục `model`) để tạo ra các file CSV chứa:
- Đặc trưng (features): `csv_train.csv`
- Nhãn (labels): `csv_trainLabels.csv`
- Đường dẫn ảnh (paths): `csv_trainPaths.csv`
**Lệnh:**
```bash
python FeatureVectorsTrain.py
```

### Bước 2: Trích xuất đặc trưng tập Test
Tương tự như Bước 1, chạy file `FeatureVectorsTest.py` để trích xuất đặc trưng cho tập test.
File này sẽ tạo ra:
- `csv_test.csv`
- `csv_testLabels.csv`
- `csv_testPaths.csv`
**Lệnh:**
```bash
python FeatureVectorsTest.py
```

### Bước 3: Train mô hình TinySiamese
Sau khi đã có các file CSV từ Bước 1 và 2, tiến hành train mạng TinySiamese.
Mô hình sẽ đọc dữ liệu từ CSV và bắt đầu học. Sau khi train xong, mô hình (trọng số) sẽ được lưu lại trong thư mục `model`.
**Lệnh:**
```bash
python TinySiameseTrain.py
```

### Bước 4: Test và Đánh giá Mô hình
Chạy file `TinySiameseTest.py` để test mô hình đã train. File này sẽ load mô hình đã lưu và tập test CSV để đưa ra các chỉ số đánh giá (độ chính xác, loss, etc.).
**Lệnh:**
```bash
python TinySiameseTest.py
```

## Lưu ý về Dữ liệu
- Trong thư mục `Database`, có chứa sẵn một số ảnh mẫu (thuộc bộ FVC2002 và FVC2004).
- Nếu bạn có tập dữ liệu vân tay riêng, hãy để đúng cấu trúc thư mục giống với mẫu trong `Database` và sửa lại đường dẫn trong code nếu cần thiết.
- Cần train ACGAN trước (`python GANTrain.py`) để có `model/GANFVC.h5` dùng trích xuất đặc trưng.
