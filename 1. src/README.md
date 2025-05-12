Dự Án Đề Xuất Cây Trồng / Crop Recommendation Project

Mô tả / Description

Dự án này xây dựng một mô hình máy học để đề xuất loại cây trồng phù hợp dựa trên các đặc trưng đất và khí hậu (N, P, K, nhiệt độ, độ ẩm, pH, lượng mưa). Dự án sử dụng:


ydata-profiling để trực quan hóa và phân tích dữ liệu, tạo báo cáo chi tiết về phân phối đặc trưng, tương quan, và giá trị thiếu.



LazyPredict để so sánh hiệu suất của nhiều thuật toán phân loại.



RandomForestClassifier được tối ưu hóa bằng RandomizedSearchCV để dự đoán cây trồng.

This project builds a machine learning model to recommend suitable crops based on soil and climate features (N, P, K, temperature, humidity, pH, rainfall). The project uses:

ydata-profiling for data visualization and analysis, generating a detailed report on feature distributions, correlations, and missing values.



LazyPredict to compare the performance of multiple classification algorithms.



RandomForestClassifier optimized with RandomizedSearchCV for crop prediction.



Techniques to check for overfitting, including learning curves and comparison with a simpler model (Logistic Regression)

Dự án sử dụng ydata-profiling để tạo báo cáo trực quan hóa dữ liệu tự động. Báo cáo (crop_recommendation_report.html) bao gồm:

Phân phối của từng đặc trưng (histogram, thống kê cơ bản).
Tương quan giữa các đặc trưng (ma trận tương quan).
phát hiện giá trị thiếu và giá trị bất thường.

Phân tích tương tác giữa các đặc trưng.

Để xem báo cáo, mở file crop_recommendation_report.html trong trình duyệt sau khi chạy script.

The project uses ydata-profiling to generate an automated data visualization report. The report (crop_recommendation_report.html) includes:

Distribution of each feature (histograms, basic statistics).



Correlations between features (correlation matrix).



Detection of missing values and outliers.
Analysis of feature interactions.

To view the report, open crop_recommendation_report.html in a browser after running the script.

Kết quả / Results

Độ chính xác kiểm tra: ~99% (RandomForestClassifier).

Báo cáo phân tích dữ liệu chi tiết từ ydata-profiling.

Dữ liệu / Dataset

Tập dữ liệu: Crop Recommendation Dataset
Bao gồm 2200 mẫu, 22 loại cây trồng, 7 đặc trưng.
