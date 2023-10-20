FROM python:3.10.13

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN --mount=type=secret,id=OPENAI_API_KEY \
    export OPENAI_API_KEY=$(cat /run/secrets/OPENAI_API_KEY) && \
    yarn gen

COPY ./serving ./serving

COPY ./lore ./lore

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "80"]