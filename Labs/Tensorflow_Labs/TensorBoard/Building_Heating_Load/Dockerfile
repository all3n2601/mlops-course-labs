FROM python:3.11-slim
WORKDIR /lab
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
COPY . .
CMD ["python", "execute_notebook.py"]
