FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04

# Install Python
RUN apt-get update && apt-get install -y python3.12 python3-pip

WORKDIR /

# 0.525 × This environment is externally managed                                                                         
COPY requirements.txt .
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt
RUN pip install --break-system-packages --no-cache-dir sentencepiece 
RUN pip install --break-system-packages --no-cache-dir torch
RUN pip install --break-system-packages --no-cache-dir accelerate

COPY 1.py .

# (find-man "python")
# (find-man "python" " -u")
CMD ["python3", "-u", "/1.py"]
# CMD ["python3", "-u", "/1.py", "--test_input", "'{\"input\": {\"prompt\": \"Hello, world!\"}}'"]
# CMD ["python3", "-u", "/1.py", "--test_input", "{\"input\": {\"prompt\": \"Hello, world!\", \"max_tokens\":15}}"]

