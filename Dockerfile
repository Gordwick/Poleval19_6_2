FROM huggingface/transformers-pytorch-cpu:4.18.0

LABEL maintainer="Sebastian Sitko"

WORKDIR /application

COPY ./requirements.cfg .
RUN pip --no-cache-dir install -r requirements.cfg

COPY ./model.bin ..
COPY ./app .

COPY ./start.sh .
RUN chmod +x ./start.sh

EXPOSE 80
CMD ["./start.sh"]