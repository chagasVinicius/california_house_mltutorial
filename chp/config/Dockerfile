FROM totvslabs/pycarol:2.41.2

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN pip install -r requirements.txt

ADD . /app

CMD ["python3", "run.py"]
