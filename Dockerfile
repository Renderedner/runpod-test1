FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 1.py .

# (find-man "python")
# (find-man "python" " -u")
cmd ["python", "-u", "/1.py"]

