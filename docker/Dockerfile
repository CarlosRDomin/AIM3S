FROM python

RUN pip install --upgrade pip \
    && pip install --no-cache-dir numpy \
    && pip install --no-cache-dir opencv-python \
    && pip install --no-cache-dir matplotlib \
    && pip install --no-cache-dir -U ptvsd==3.0.0

CMD ["/bin/bash"]