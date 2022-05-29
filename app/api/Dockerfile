FROM python:3.10

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
