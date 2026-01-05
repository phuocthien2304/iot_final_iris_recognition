# Nhận dạng mống mắt bằng CNN

Kho lưu trữ này cung cấp mã huấn luyện/đánh giá hệ thống nhận dạng mống mắt dựa trên CNN và một ứng dụng demo GUI (Tkinter) có thể phân đoạn (segment), chuẩn hóa (normalize) và xếp chồng (stack) ảnh mống mắt thô theo thời gian thực.

## 1) Cài đặt môi trường

Bạn có thể dùng Conda hoặc Python venv. Demo cần OpenCV cho tiền xử lý.

- Tùy chọn A  Conda (khuyến nghị)
`
conda create -n iris-recognition-cnn python=3.9 -y
conda activate iris-recognition-cnn
pip install torch torchvision numpy pillow scikit-learn opencv-python
`

- Tùy chọn B  Dùng file environment.yml
`
conda env create -f environment.yml
conda activate iris-recognition-cnn
`

- Tùy chọn C  Python venv
`
python -m venv .venv
. .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision numpy pillow scikit-learn opencv-python matplotlib tqdm
`

Ghi chú:
- Hỗ trợ chạy trên máy chỉ có CPU (không bắt buộc CUDA). Checkpoint sẽ được nạp trên CPU khi cần.
- Demo dùng OpenCV cho segment/normalize nên cần gói opencv-python.

## 2) Mô hình đã huấn luyện và dữ liệu

- Checkpoint pretrained: https://drive.google.com/file/d/1pf-X13Dpsj-Q8p5Xzl0-uCoB1IeFmYD7/view?usp=sharing
- Ảnh đã tiền xử lý (tùy chọn): https://drive.google.com/file/d/172GZgrAzNrA146BjFpF1vXtAH-zWymlA/view?usp=sharing

Đặt các file .pth vào thư mục models/ (ví dụ: models/best_iris_cnn_improved.pth). Các thư mục lớn như /models, /data đã nằm trong .gitignore.

## 3) Chạy demo GUI

`
python demo_gui.py
`

Trong giao diện:
- Chọn kiến trúc: 
esnet101 hoặc densenet201, duyệt tới checkpoint .pth trong models/.
- Thêm 5 ảnh ghi danh (enrollment): nhập tên + chọn ảnh. Hỗ trợ PNG/JPG/BMP/TIF/TIFF.
- Chọn 1 ảnh truy vấn (query).
- Tùy chọn: bật Preprocess raw image (segment + normalize + stack) để chuẩn hóa ảnh thô ngay khi chạy demo.
- Nhấn Run Identification. Sử dụng ngưỡng (threshold) để từ chối người lạ (open-set).

GUI hiển thị lưới 6 cột thumbnail (5 người + 1 query):
- Hàng trên: ảnh gốc (nhỏ).
- Hàng dưới: ảnh sau tiền xử lý (nhỏ).
- Dòng tên nằm dưới mỗi cột. Kết quả và điểm tương đồng hiển thị trong khung log.

## 4) Chạy bằng dòng lệnh (CLI)

Cần dữ liệu đã tiền xử lý theo chuẩn ImageFolder. Tham khảo các script trong thư mục /scripts để segment/normalize/stack trước khi huấn luyện/đánh giá.

- Huấn luyện (closed set):
`
python train.py --arch resnet101 --data_dir <PATH_TO_DATA> --epochs 80 --lr 2e-5
`
- Đánh giá closed set:
`
python eval_closed_set.py --arch resnet101 --data_dir <PATH_TO_DATA> --checkpoint models/best_iris_cnn_improved.pth
`
- Đánh giá open set:
`
python eval_open_set.py --arch resnet101 --enrollment_dir <PATH_ENROLL> --test_dir <PATH_TEST> --checkpoint models/best_iris_cnn_improved.pth
`
- Cross-validation (open set):
`
python cv_open_set.py --arch resnet101 --data_dir <PATH_TO_DATA> --checkpoint models/best_iris_cnn_improved.pth
`

## 5) Tiền xử lý dữ liệu

Các script tiền xử lý nằm trong thư mục /scripts (segment, normalize, stack). Nên dùng pipeline này để tạo bộ dữ liệu đồng nhất cho huấn luyện/đánh giá. Trong demo GUI đã có bản rút gọn để xử lý nhanh ảnh thô.

## 6) Khắc phục sự cố

- ModuleNotFoundError: cv2: cài đặt pip install opencv-python.
- TIFF nhiều khung: demo sẽ đọc khung đầu tiên.
- Tải checkpoint trên CPU: mã đã hỗ trợ map_location='cpu'.
- Điểm tương đồng không phải 1.0 dù cùng mắt: do khác biệt nhỏ trong segment/normalize và số học dấu phẩy động; hãy đặt ngưỡng thực tế (vd. 0.900.95).

## 7) Tệp quan trọng

- 	rain.py  huấn luyện closed set (xem --help).
- eval_closed_set.py  đánh giá closed set.
- eval_open_set.py  đánh giá open set (cosine similarity).
- cv_open_set.py  cross-validation open set.
- Thư mục /scripts  công cụ tiền xử lý dữ liệu.

## 8) Giấy phép

Dùng cho mục đích học thuật/demo. Vui lòng tuân thủ giấy phép của bộ dữ liệu bạn sử dụng.
