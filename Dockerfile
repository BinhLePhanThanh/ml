FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /

# Sao chép file Python vào container
COPY api.py .
# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir Flask torch numpy

# Sao chép model.pkl vào container (nếu có)
COPY model.pkl .

# Mở cổng 5000
EXPOSE 5000

# Lệnh để chạy ứng dụng
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]