# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Label
LABEL maintainer="mlfullstack@gmail.com"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /train_eval_test_on_docker_microservices

# Copy all project files first
COPY . .

# Install dependencies
RUN poetry install

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set Python path
ENV PYTHONPATH=/train_eval_test_on_docker_microservices

WORKDIR /train_eval_test_on_docker_microservices/src/train_eval_test_on_docker_microservices

CMD ["python", "main.py", "--mode", "train", "--num-processes", "2", "--epochs", "1", "--save_model"]