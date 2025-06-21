# 🚀 Train Eval and Test a ML model on docker micro services

# 🏗️ From Scratch
- Create a conda environment using command
    ```
    conda create -n train_eval_test_on_docker_microservices python=3.10
    ```
- Activate the environment
    ```
    conda activate train_eval_test_on_docker_microservices
    ```
- Install poetry inside conda environment
    ```
    python -m pip install poetry
    ```
- Create a poetry project for seameless dependency management
    ```
    python -m poetry new train_eval_test_on_docker_microservices
    ```
- Project structure would look like this:
    ```
    .
    ├── README.md
    └── train_eval_test_on_docker_microservices
        ├── pyproject.toml
        ├── README.md
        ├── src
        │   └── train_eval_test_on_docker_microservices
        │       └── __init__.py
        └── tests
            └── __init__.py
    ```
- Keep adding the packages into poetry "pyproject.toml" as and when required. You can also follow the pyproject.toml file of this repo to identify the required dependencies. Example command to add a dependency:
    ```
    python -m poetry add pandas
    ```
- Since poetry package management does not efficiently work with PyTorch installation, install PyTorch by referring to: [PyTorch Installation Guide](https://pytorch.org)
    ```
    python -m pip install torch torchvision torchaudio
    ```
- Since focus of this project is to train, eval and test the ML model using docker micro services, Because of which, instead of investing time on writing own Model config and training code, Let's use the Pytorch hogwilds example on mnist dataset. You can get the code at: [Pytorch MNIST Hogwild](https://github.com/pytorch/examples/tree/main/mnist_hogwild)

# ⚡ Quick Start Guide

> 🎯 **Perfect for users who want to jump straight into training, evaluation, and testing without development setup!**

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:
- 🐳 **Docker** (version 20.0 or higher)
- 🐙 **Docker Compose** (version 2.0 or higher)

### 🔍 Check Your Installation
```bash
# Verify Docker installation
docker --version

# Verify Docker Compose installation
docker compose version
```

## 🚀 Getting Started

### 1. 📥 Clone the Repository
```bash
git clone https://github.com/fullstackml-hub/docker-ml-microservices.git
cd docker_compose_for_train_eval_test
```

### 2. 📁 Directory Structure
After cloning, your project structure should look like this:
```
docker_compose_for_train_eval_test/
├── 📂 data/                    # Dataset storage (auto-created)
├── 📂 models/                  # Trained models storage
├── 📂 train_eval_test_on_docker_microservices/
│   ├── 📦 Dockerfile
│   ├── 🐙 compose.yaml
│   └── 📂 src/
└── 📖 README.md
```

### 3. 🏗️ Build Docker Services
```bash
cd train_eval_test_on_docker_microservices
docker compose build
```
> ⏱️ **Note**: Initial build may take 5-10 minutes as it downloads dependencies.

## 🎯 Running the ML Pipeline

### 🚄 Option 1: Run Individual Services

#### 🔥 Train the Model
```bash
docker compose up train
```
- ⚡ Trains MNIST model using **2 parallel processes**
- 📊 Trains for **1 epoch** (configurable)
- 💾 Automatically saves trained model to `../models/MNIST_hogwild.pt`
- 📈 Shows real-time training progress and loss

#### 📊 Evaluate the Model
```bash
docker compose up eval
```
- 🔍 Loads the trained model
- 📋 Evaluates performance on test dataset
- 📊 Displays accuracy and loss metrics

#### 🧪 Test the Model
```bash
docker compose up test
```
- 🎯 Performs final testing on test dataset
- 📈 Shows detailed test results and accuracy

### 🔄 Option 2: Run Complete Pipeline
```bash
# Run all services sequentially
docker compose up train && docker compose up eval && docker compose up test
```

### 🎛️ Option 3: Custom Training Parameters
```bash
# Train with custom parameters
docker compose run --rm train python main.py --mode train --epochs 5 --num-processes 4 --save_model

# Evaluate with custom batch size
docker compose run --rm eval python main.py --mode eval --test-batch-size 500

# Test with specific settings
docker compose run --rm test python main.py --mode test --test-batch-size 1000
```

## 📊 Expected Output

### 🚄 Training Output
```
Train Epoch: 1 [0/60000 (0%)]    Loss: 2.300000
Train Epoch: 1 [640/60000 (1%)]  Loss: 1.950000
...
Model saved to ../models/MNIST_hogwild.pt
```

### 📊 Evaluation/Test Output
```
Test set: Average loss: 0.0500, Accuracy: 9800/10000 (98%)
```

## 🛠️ Useful Commands

### 🔍 Monitoring and Debugging
```bash
# View running containers
docker compose ps

# View logs from a specific service
docker compose logs train
docker compose logs eval
docker compose logs test

# Follow logs in real-time
docker compose logs -f train
```

### 🧹 Cleanup Commands
```bash
# Stop all services
docker compose down

# Remove all containers, networks, and images
docker compose down --rmi all --volumes --remove-orphans

# Remove only containers and networks (keep images)
docker compose down --volumes
```

### 🔧 Troubleshooting
```bash
# Rebuild services (if you made changes)
docker compose build --no-cache

# Start with fresh containers
docker compose down && docker compose up train

# Check system resources
docker system df
docker system prune  # Clean up unused resources
```

## 📂 Data and Model Management

### 📥 Dataset Location
- **Host Path**: `./data/`
- **Container Path**: `/train_eval_test_on_docker_microservices/../data`
- 🎯 MNIST dataset is **automatically downloaded** on first run

### 💾 Model Storage
- **Host Path**: `./models/`
- **Container Path**: `/train_eval_test_on_docker_microservices/../models`
- 📋 Trained models persist between container runs
- 🔄 Models are shared across all services (train/eval/test)

## ⚙️ Configuration Options

### 🎛️ Available Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 1 | Number of training epochs |
| `--batch-size` | 64 | Training batch size |
| `--test-batch-size` | 1000 | Testing batch size |
| `--lr` | 0.01 | Learning rate |
| `--num-processes` | 2 | Number of parallel processes |
| `--save_model` | False | Save trained model |

### 🔧 Modify Default Settings
Edit the `compose.yaml` file to change default parameters:
```yaml
command: python main.py --mode train --num-processes 4 --epochs 10 --save_model
```

## 🎉 Quick Success Check

After running the pipeline, you should see:
- ✅ **Training**: Model trained and saved to `./models/MNIST_hogwild.pt`
- ✅ **Evaluation**: Test accuracy around **95-98%**
- ✅ **Testing**: Final test results displayed

**🎊 Congratulations! You've successfully set up and run the ML pipeline using Docker microservices!**

# 📖 Code Explanation

### 💻 main.py

The `main.py` file serves as the entry point for the MNIST Hogwild distributed training application. Here's a breakdown of its key components:

#### 1. ⚙️ **Argument Parser Configuration**
```python
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
```
The script defines multiple command-line arguments for configuring training:
- `--batch-size`: Input batch size for training (default: 64)
- `--test-batch-size`: Input batch size for testing (default: 1000)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.01)
- `--momentum`: SGD momentum (default: 0.5)
- `--num-processes`: Number of parallel training processes (default: 2)
- `--cuda`, `--mps`: Enable GPU training on CUDA or Apple Silicon
- `--save_model`: Flag to save the trained model
- `--dry-run`: Quick single-pass check for debugging

#### 2. 🧠 **Neural Network Architecture (Net Class)**
```python
class Net(nn.Module):
```
Defines a Convolutional Neural Network with:
- **conv1**: First convolutional layer (1→10 channels, 5x5 kernel)
- **conv2**: Second convolutional layer (10→20 channels, 5x5 kernel)
- **conv2_drop**: Dropout layer for regularization
- **fc1**: First fully connected layer (320→50 neurons)
- **fc2**: Output layer (50→10 neurons for 10 digit classes)

The forward pass applies:
1. Conv1 → ReLU → MaxPool
2. Conv2 → Dropout → ReLU → MaxPool
3. Flatten → FC1 → ReLU → Dropout → FC2 → LogSoftmax

#### 3. 🖥️ **Device Selection and Data Loading**
```python
use_cuda = args.cuda and torch.cuda.is_available()
use_mps = args.mps and torch.backends.mps.is_available()
```
- Automatically detects and configures the appropriate device (CPU, CUDA, or MPS)
- Downloads and preprocesses MNIST dataset with normalization
- Creates data loaders with specified batch sizes

#### 4. ⚡ **Multiprocessing Training (Hogwild Algorithm)**
```python
processes = []
for rank in range(args.num_processes):
    p = mp.Process(target=train, args=(rank, args, model, device, dataset1, kwargs))
```
- Creates multiple processes for parallel training
- Each process trains on the shared model simultaneously
- Uses `model.share_memory()` to enable gradient sharing across processes
- Implements the Hogwild algorithm for asynchronous parallel SGD

#### 5. 📊 **Model Evaluation**
```python
test(args, model, device, dataset2, kwargs)
```
After training completion, evaluates the model on the test dataset.

### 🏋️ train.py

The `train.py` file contains the core training and testing logic with four main functions:

#### 1. **train(rank, args, model, device, dataset, dataloader_kwargs)**
```python
def train(rank, args, model, device, dataset, dataloader_kwargs):
```
- **Purpose**: Main training function called by each process
- **Parameters**:
  - `rank`: Process identifier for seeding
  - `args`: Training configuration arguments
  - `model`: Shared neural network model
  - `device`: Computing device (CPU/GPU)
  - `dataset`: Training dataset
  - `dataloader_kwargs`: DataLoader configuration
- **Functionality**:
  - Sets unique random seed per process (`args.seed + rank`)
  - Creates DataLoader for the process
  - Initializes SGD optimizer
  - Executes training for specified epochs

#### 2. **test(args, model, device, dataset, dataloader_kwargs)**
```python
def test(args, model, device, dataset, dataloader_kwargs):
```
- **Purpose**: Evaluates trained model on test dataset
- **Parameters**: Similar to train function but uses test dataset
- **Functionality**:
  - Creates test DataLoader
  - Calls `test_epoch` for model evaluation

#### 3. **train_epoch(epoch, args, model, device, data_loader, optimizer)**
```python
def train_epoch(epoch, args, model, device, data_loader, optimizer):
```
- **Purpose**: Executes one training epoch
- **Key Operations**:
  - Sets model to training mode (`model.train()`)
  - Iterates through batches in data_loader
  - For each batch:
    - Zeros gradients (`optimizer.zero_grad()`)
    - Forward pass through model
    - Computes negative log-likelihood loss
    - Backpropagation (`loss.backward()`)
    - Updates parameters (`optimizer.step()`)
  - Logs training progress every `log_interval` batches
  - Shows process ID, epoch, batch progress, and loss value

#### 4. **test_epoch(model, device, data_loader)**
```python
def test_epoch(model, device, data_loader):
```
- **Purpose**: Evaluates model performance on test data
- **Key Operations**:
  - Sets model to evaluation mode (`model.eval()`)
  - Disables gradient computation (`torch.no_grad()`)
  - Iterates through test batches:
    - Computes predictions
    - Accumulates loss and correct predictions
  - Calculates and prints:
    - Average test loss
    - Accuracy (correct predictions / total samples)
    - Accuracy percentage

#### ✨ **Key Features of the Implementation**:
- ⚡ **Asynchronous Training**: Multiple processes train simultaneously on shared model
- 💾 **Memory Efficiency**: Uses shared memory for model parameters
- 🖥️ **Device Agnostic**: Supports CPU, CUDA, and Apple Silicon (MPS)
- 📊 **Progress Monitoring**: Real-time logging of training progress and test accuracy
- 🔄 **Reproducibility**: Configurable random seeding for consistent results

This implementation demonstrates distributed training using PyTorch's multiprocessing capabilities, making it suitable for scaling training across multiple CPU cores or processes.

## 🐳 Docker Configuration

### 📦 Dockerfile

The `Dockerfile` creates a containerized environment for running the MNIST training application. Here's a detailed breakdown:

#### 1. 🏗️ **Base Image Selection**
```dockerfile
FROM python:3.10-slim
```
- Uses **Python 3.10-slim** as the base image
- `slim` variant provides a minimal Python installation, reducing image size
- Ensures consistent Python environment across different systems

#### 2. 🔧 **Metadata and Environment Variables**
```dockerfile
LABEL maintainer="mlfullstack@gmail.com"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1
```
- **PYTHONUNBUFFERED=1**: Ensures Python output is sent directly to terminal (no buffering)
- **PYTHONDONTWRITEBYTECODE=1**: Prevents Python from creating `.pyc` files
- **POETRY_VERSION=2.1.3**: Specifies Poetry version for dependency management
- **POETRY_HOME**: Sets Poetry installation directory
- **POETRY_VIRTUALENVS_CREATE=false**: Disables virtual environment creation (using container isolation)
- **POETRY_NO_INTERACTION=1**: Runs Poetry in non-interactive mode

#### 3. ⬇️ **System Dependencies Installation**
```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
```
- **curl**: Required for downloading Poetry installer
- **build-essential**: Provides compilers and build tools for Python packages with C extensions
- **--no-install-recommends**: Installs only essential packages, reducing image size
- **rm -rf /var/lib/apt/lists/***: Cleans up package lists to reduce image size

#### 4. 📝 **Poetry Installation**
```dockerfile
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
```
- Downloads and installs Poetry using the official installer
- Adds Poetry to the system PATH for global access

#### 5. ⚙️ **Application Setup**
```dockerfile
WORKDIR /train_eval_test_on_docker_microservices
COPY . .
RUN poetry install
```
- Sets working directory inside the container
- Copies all project files from host to container
- Installs project dependencies using Poetry

#### 6. 🔥 **PyTorch Installation**
```dockerfile
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
- Installs PyTorch CPU version directly with pip (bypassing Poetry for efficiency)
- **--no-cache-dir**: Prevents pip from caching packages, reducing image size
- **--index-url**: Uses PyTorch's official CPU-only package repository

#### 7. 🎯 **Final Configuration**
```dockerfile
ENV PYTHONPATH=/train_eval_test_on_docker_microservices
WORKDIR /train_eval_test_on_docker_microservices/src/train_eval_test_on_docker_microservices
CMD ["python", "main.py", "--mode", "train", "--num-processes", "2", "--epochs", "1", "--save_model"]
```
- **PYTHONPATH**: Ensures Python can find project modules
- **WORKDIR**: Changes to the source code directory
- **CMD**: Default command to run training mode with 2 processes for 1 epoch

### 🐙 Docker Compose Configuration (compose.yaml)

The `compose.yaml` file orchestrates multiple microservices for training, evaluation, and testing. Here's the detailed explanation:

#### 🏗️ **Service Architecture**
The configuration defines three microservices that represent different phases of the ML pipeline:

#### 1. 🚄 **Train Service**
```yaml
train:
  build:
    context: .
    dockerfile: Dockerfile
  command: python main.py --mode train --num-processes 2 --epochs 1 --save_model
  volumes:
    - ./../data:/train_eval_test_on_docker_microservices/../data
    - ./../models:/train_eval_test_on_docker_microservices/../models
```
- **Purpose**: Handles model training phase
- **Build Context**: Uses current directory (.) as build context
- **Command Override**: Runs training mode with specific parameters:
  - `--mode train`: Sets application to training mode
  - `--num-processes 2`: Uses 2 parallel processes for distributed training
  - `--epochs 1`: Trains for 1 epoch (configurable)
  - `--save_model`: Saves trained model after completion
- **Volume Mounts**:
  - **Data Volume**: `../data` (host) → `/train_eval_test_on_docker_microservices/../data` (container)
  - **Models Volume**: `../models` (host) → `/train_eval_test_on_docker_microservices/../models` (container)

#### 2. 📊 **Eval Service**
```yaml
eval:
  build:
    context: .
    dockerfile: Dockerfile
  command: python main.py --mode eval
  volumes:
    - ./../data:/train_eval_test_on_docker_microservices/../data
    - ./../models:/train_eval_test_on_docker_microservices/../models
```
- **Purpose**: Handles model evaluation phase
- **Command**: Runs evaluation mode (`--mode eval`)
- **Shared Volumes**: Accesses same data and models directories as training service
- **Functionality**: Loads trained model and evaluates performance on validation/test data

#### 3. 🧪 **Test Service**
```yaml
test:
  build:
    context: .
    dockerfile: Dockerfile
  command: python main.py --mode test
  volumes:
    - ./../data:/train_eval_test_on_docker_microservices/../data
    - ./../models:/train_eval_test_on_docker_microservices/../models
```
- **Purpose**: Handles model testing phase
- **Command**: Runs testing mode (`--mode test`)
- **Shared Volumes**: Accesses same data and models directories
- **Functionality**: Performs final model testing and generates test metrics

#### 🎯 **Key Benefits of This Architecture**:

1. ✅ **Microservices Pattern**: Each phase (train/eval/test) runs as an independent service
2. ✅ **Shared Data**: All services share common data and model storage through volume mounts
3. ✅ **Scalability**: Services can be scaled independently based on computational needs
4. ✅ **Isolation**: Each service runs in its own container, ensuring environment consistency
5. ✅ **Pipeline Orchestration**: Services can be run sequentially or in parallel as needed

#### 💾 **Volume Management Strategy**:
- **Host-Container Mapping**: Data and models persist on the host filesystem
- **Data Persistence**: Training data and model files survive container restarts
- **Cross-Service Sharing**: All services access the same datasets and trained models
- **Development Workflow**: Changes to data/models on host are immediately available to containers

#### 🚀 **Usage Patterns**:
```bash
# Run individual services
docker compose up train    # Train the model
docker compose up eval     # Evaluate the model
docker compose up test     # Test the model

# Run all services sequentially
docker compose up train && docker compose up eval && docker compose up test

# Build all services
docker compose build

# Clean up
docker compose down --rmi all --volumes
```

This containerized microservices approach enables reproducible, scalable, and maintainable ML workflows that can be easily deployed across different environments.

