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

Assignment 1 được chạy với nhiều seed khởi tạo centroid khác nhau từ `0` đến `4`. Kết quả cho thấy các seed đều hội tụ về cùng giá trị inertia `1119.9859` và accuracy `0.9817`. Điều này cho thấy dữ liệu trong Assignment 1 có ba cụm khá rõ ràng, nên K-Means có thể tìm được nghiệm ổn định dù centroid ban đầu khác nhau.

Tuy nhiên, số vòng lặp để hội tụ lại khác nhau giữa các seed. Ví dụ seed `0` và seed `2` hội tụ sau `4` vòng lặp, trong khi seed `1` cần đến `11` vòng lặp. Vì vậy, khởi tạo centroid ngẫu nhiên không làm thay đổi nhiều kết quả cuối trong thí nghiệm này, nhưng có ảnh hưởng đến tốc độ hội tụ của thuật toán.

### Assignment 2: ảnh hưởng của kích thước cụm không cân bằng

Assignment 2 sử dụng ba cụm có kích thước không cân bằng: `1200`, `200` và `1000` điểm. Kết quả thực nghiệm cho thấy K-Means vẫn phân cụm tốt với accuracy `0.9808` và inertia `4662.0401`. Điều này xảy ra vì mặc dù số lượng điểm giữa các cụm chênh lệch lớn, các cụm trong dữ liệu vẫn tương đối tách biệt.

Tuy nhiên, về mặt bản chất, K-Means tối thiểu hóa tổng bình phương khoảng cách từ các điểm đến centroid, nên các cụm có nhiều điểm có thể ảnh hưởng mạnh hơn đến vị trí centroid. Do đó, dữ liệu không cân bằng có thể làm thuật toán nhạy hơn, đặc biệt khi các cụm nằm gần nhau hoặc bị chồng lấn nhiều.

### Assignment 3: ảnh hưởng của phân phối có phương sai lớn theo một chiều

Assignment 3 có một cụm sử dụng ma trận hiệp phương sai `[[10, 0], [0, 1]]`, làm dữ liệu bị kéo dài theo trục x. Kết quả thực nghiệm cho thấy Assignment 3 có accuracy `0.9317`, thấp hơn Assignment 1 và Assignment 2. Điều này phù hợp với đặc điểm của K-Means vì thuật toán sử dụng khoảng cách Euclidean và thường hoạt động tốt hơn với các cụm có dạng gần tròn.

Khi một cụm bị kéo dài, ranh giới phân cụm của K-Means có thể không còn phù hợp hoàn toàn với hình dạng thật của dữ liệu. Vì vậy, một số điểm ở cụm kéo dài dễ bị gán sang cụm khác, làm hiệu năng giảm so với trường hợp các cụm Gaussian tròn đều.

