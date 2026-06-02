# Lab 5 - K-Means Clustering

Project này chỉ làm phần **K-Means** trong bài học máy thống kê. Phần Gaussian Mixture Model chưa làm theo yêu cầu.

## Cấu trúc thư mục

```text
kmeans_lab5_project/
├── run_kmeans_assignments.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── data.py
    ├── experiments.py
    ├── kmeans.py
    ├── metrics.py
    └── plotting.py
```

## Nội dung chính

- `src/kmeans.py`: class `KMeans`, tự cài bằng NumPy theo EM:
  - E-step: gán mỗi điểm vào centroid gần nhất.
  - M-step: cập nhật centroid bằng trung bình các điểm trong cụm.
- `src/data.py`: sinh dữ liệu Gaussian cho Assignment 1, 2, 3.
- `src/metrics.py`: tính accuracy sau khi ánh xạ lại nhãn cụm, vì nhãn của K-Means là tùy ý.
- `src/plotting.py`: vẽ kết quả phân cụm và đường giảm inertia.
- `src/experiments.py`: chạy toàn bộ ba assignment và lưu kết quả.

## Cách chạy

```bash
pip install -r requirements.txt
python run_kmeans_assignments.py
```

Sau khi chạy, xem kết quả trong thư mục `outputs/`:

- `summary.csv`: bảng kết quả.
- `*_clusters.png`: biểu đồ phân cụm.
- `*_inertia.png`: biểu đồ inertia giảm qua từng vòng lặp.

## Nhận xét cho báo cáo

### Assignment 1: ảnh hưởng của khởi tạo centroid ngẫu nhiên

K-Means phụ thuộc vào vị trí centroid ban đầu. Nếu khởi tạo tốt, thuật toán hội tụ nhanh và tìm đúng ba cụm. Nếu khởi tạo kém, có thể hội tụ vào nghiệm cục bộ, inertia cao hơn hoặc một centroid bị kéo về vùng không tối ưu. Vì vậy trong thực tế nên chạy K-Means nhiều lần với nhiều seed khác nhau rồi chọn mô hình có inertia nhỏ nhất.

### Assignment 2: ảnh hưởng của kích thước cụm không cân bằng

Khi số lượng điểm giữa các cụm chênh lệch lớn, cụm lớn có ảnh hưởng mạnh hơn lên vị trí centroid. K-Means có xu hướng tối thiểu hóa tổng bình phương khoảng cách nên có thể ưu tiên mô tả tốt cụm đông điểm hơn cụm ít điểm. Điều này làm cụm nhỏ dễ bị phân sai hơn so với trường hợp các cụm cân bằng.

### Assignment 3: ảnh hưởng của phân phối có phương sai lớn theo một chiều

Cụm thứ ba dùng ma trận hiệp phương sai `[[10, 0], [0, 1]]`, nên dữ liệu bị kéo dài theo trục x. K-Means giả định cụm có dạng gần cầu vì dùng khoảng cách Euclidean đến centroid. Khi một cụm bị kéo dài, ranh giới K-Means có thể cắt cụm này thành nhiều phần hoặc trộn nó với cụm khác, làm hiệu năng giảm so với dữ liệu Gaussian tròn đều.
