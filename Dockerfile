FROM python:3.8

COPY ./requires.txt ./requires.txt

RUN pip install -r requires.txt

COPY . .

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
