# Container for Hugging Face Spaces (Docker SDK) or any Docker host. Serves API + frontend on port 7860.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user static ./static

# Model to serve: a Hugging Face Hub repo id or a local folder. Override in the Space settings.
ENV MODEL_ID=YOUR_HF_USERNAME/nllb-600m-kidawida-swahili
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
