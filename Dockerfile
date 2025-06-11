FROM hub.cs.upb.de/dice-research/images/python:3.11
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "9000", "server:app"]
