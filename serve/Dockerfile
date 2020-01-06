FROM python:3.7-slim
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
ADD serve.py /
ENTRYPOINT [ "python", "./serve.py" ]
