FROM python:3.8.8
WORKDIR /auto_scoring

COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["gunicorn", "server:app", "-c", "./gunicorn.conf.py"]
