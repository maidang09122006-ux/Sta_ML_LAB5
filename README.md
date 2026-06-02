# Lab 5 - K-Means Clustering

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

K-Means phụ thuộc vào vị trí centroid ban đầu. Trong thí nghiệm này, các seed khởi tạo khác nhau đều hội tụ về nghiệm có cùng inertia và accuracy, cho thấy dữ liệu có ba cụm khá rõ ràng. Tuy nhiên, số vòng lặp để hội tụ khác nhau giữa các seed, ví dụ có seed hội tụ nhanh hơn và có seed cần nhiều vòng lặp hơn. Vì vậy trong thực tế vẫn nên chạy K-Means nhiều lần với nhiều seed khác nhau rồi chọn kết quả có inertia nhỏ nhất.

### Assignment 2: ảnh hưởng của kích thước cụm không cân bằng

Khi số lượng điểm giữa các cụm chênh lệch lớn, cụm lớn có thể ảnh hưởng mạnh hơn đến vị trí centroid vì K-Means tối thiểu hóa tổng bình phương khoảng cách trên toàn bộ dữ liệu. Trong thí nghiệm này, mặc dù kích thước cụm không cân bằng, các cụm vẫn tách khá rõ nên K-Means vẫn đạt accuracy cao. Điều này cho thấy mất cân bằng kích thước có thể gây khó khăn, nhưng mức độ ảnh hưởng còn phụ thuộc vào khoảng cách và độ chồng lấn giữa các cụm.

### Assignment 3: ảnh hưởng của phân phối có phương sai lớn theo một chiều

Cụm thứ ba dùng ma trận hiệp phương sai `[[10, 0], [0, 1]]`, nên dữ liệu bị kéo dài theo trục x. K-Means dùng khoảng cách Euclidean đến centroid nên phù hợp hơn với các cụm có dạng gần tròn. Khi một cụm bị kéo dài, thuật toán dễ phân cụm kém chính xác hơn so với trường hợp các cụm Gaussian tròn đều. Kết quả thực nghiệm cũng cho thấy Assignment 3 có accuracy thấp hơn Assignment 1 và Assignment 2.
