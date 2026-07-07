FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS hecktor2026-task

ENV PYTHONUNBUFFERED=1
ENV PATH="/home/user/.local/bin:${PATH}"

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

USER user
WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Safety cleanup:
# If pip installed a user-site PyTorch wheel, remove it.
# We want to use the CUDA-compatible PyTorch that comes from the base image.
RUN rm -rf /home/user/.local/lib/python*/site-packages/torch \
           /home/user/.local/lib/python*/site-packages/torch-* \
           /home/user/.local/lib/python*/site-packages/torchvision \
           /home/user/.local/lib/python*/site-packages/torchvision-* \
           /home/user/.local/lib/python*/site-packages/torchaudio \
           /home/user/.local/lib/python*/site-packages/torchaudio-* \
           /home/user/.local/lib/python*/site-packages/nvidia \
           /home/user/.local/lib/python*/site-packages/nvidia-* || true

RUN python -c "import torch; print('TORCH_FILE', torch.__file__); print('TORCH_VERSION', torch.__version__); print('TORCH_CUDA', torch.version.cuda); assert '/home/user/.local' not in torch.__file__, torch.__file__"

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
