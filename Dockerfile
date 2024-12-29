FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /

# Sao chép file Python vào container
COPY api.py .
COPY model.py .
# Sao chép model.pkl vào container (nếu có)
COPY model.pkl .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir Flask torch numpy

# Mở cổng 5000
EXPOSE 5000

# Thiết lập biến môi trường FLASK_APP
ENV FLASK_APP=api.py

# Lệnh để chạy ứng dụng
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]