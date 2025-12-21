# Python 3.14 Installation Guide for Ubuntu

This guide provides step-by-step instructions for installing Python 3.14 on Ubuntu and setting up the FLUX2 LoRA Training Toolkit.

## Prerequisites

- Ubuntu 20.04 or later
- sudo access
- Internet connection

## Step 1: Remove Existing Python 3.14 Installation

If you have a broken Python 3.14 installation, remove it first:

```bash
# Remove any existing Python 3.14 packages
sudo apt remove --purge python3.14 python3.14-venv python3.14-dev -y

# Remove any manually installed Python 3.14
sudo rm -rf /usr/local/lib/python3.14* /usr/local/bin/python3.14* /usr/local/include/python3.14*

# Clean up package cache
sudo apt autoremove -y
sudo apt autoclean
```

## Step 2: Add the DeadSnakes PPA

The official Ubuntu repositories may not have Python 3.14 yet. Add the DeadSnakes PPA for newer Python versions:

```bash
# Update package list
sudo apt update

# Install software-properties-common if not already installed
sudo apt install -y software-properties-common

# Add the DeadSnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Update package list again
sudo apt update
```

## Step 3: Install Python 3.14 with All Components

Install Python 3.14 with all necessary components:

```bash
# Install Python 3.14 and essential packages
sudo apt install -y python3.14 python3.14-venv python3.14-dev python3.14-distutils python3.14-lib2to3 python3.14-gdbm python3.14-tk

# Verify installation
python3.14 --version
which python3.14
```

## Step 4: Install pip for Python 3.14

Install pip manually if it's not included:

```bash
# Download get-pip.py
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip for Python 3.14
python3.14 get-pip.py

# Clean up
rm get-pip.py

# Verify pip installation
python3.14 -m pip --version
```

## Step 5: Set Up the FLUX2 LoRA Toolkit

Navigate to your project directory and set up the environment:

```bash
# Navigate to project directory
cd ~/flux2-lora-toolkit

# Create virtual environment with Python 3.14
python3.14 -m venv venv314

# Activate the virtual environment
source venv314/bin/activate

# Verify Python version in virtual environment
python --version  # Should show Python 3.14.x

# Upgrade pip in virtual environment
python -m pip install --upgrade pip

# Install the toolkit with all dependencies
pip install --no-cache-dir -e ".[dev]"
```

## Step 6: Verify Installation

Test that everything is working:

```bash
# Activate virtual environment
source venv314/bin/activate

# Test imports
python -c "from diffusers import FluxPipeline; print('FluxPipeline imported successfully')"

# Test app launch (this will run indefinitely, use Ctrl+C to stop)
timeout 10 python app.py || echo "App launched successfully"
```

## Step 7: Download FLUX2 Model (Required for Training)

Before training, you need to download the FLUX2-dev model:

```bash
# Install huggingface_hub if not already installed
pip install huggingface-cli

# Login to HuggingFace (you'll need an account and to accept the model terms)
huggingface-cli login

# Create models directory
mkdir -p ~/models

# Download the FLUX2-dev model
huggingface-cli download black-forest-labs/FLUX.2-dev --local-dir ~/models/black-forest-labs/FLUX.2-dev
```

## Step 8: Download FLUX2-dev Model

**IMPORTANT**: This toolkit is designed specifically for **FLUX2-dev**, NOT FLUX1. FLUX2-dev has different components and will not work with FLUX1 model files.

```bash
# Create models directory
mkdir -p ~/models

# Download FLUX2-dev (requires HuggingFace authentication)
huggingface-cli login  # Login to HuggingFace first
huggingface-cli download black-forest-labs/FLUX.2-dev --local-dir ~/models/black-forest-labs/FLUX.2-dev

# Verify download (should contain text_encoder_2, tokenizer_2, etc.)
ls ~/models/black-forest-labs/FLUX.2-dev/
```

## Step 9: Usage Instructions

Once everything is set up:

```bash
# Activate environment
cd ~/flux2-lora-toolkit
source venv314/bin/activate

# Launch web interface
python app.py

# Or use command line
python cli.py train --preset character --dataset /path/to/dataset --output ./output
```

## Troubleshooting

### Virtual Environment Issues
If you get pip-related errors during venv creation:
```bash
# Create without pip, then install manually
python3.14 -m venv --without-pip venv314
source venv314/bin/activate
curl -sSL https://bootstrap.pypa.io/get-pip.py | python
```

### Permission Issues
If you get permission errors:
```bash
# Install to user directory instead of system-wide
pip install --user --no-cache-dir -e ".[dev]"
```

### Python Path Issues
If Python 3.14 isn't found:
```bash
# Check if it's installed
python3.14 --version

# If not found, check installation location
find /usr -name "python3.14" 2>/dev/null
```

### Alternative Installation Method

If the PPA method doesn't work, you can install Python 3.14 from source:

```bash
# Install build dependencies
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev

# Download Python 3.14 source
cd /tmp
wget https://www.python.org/ftp/python/3.14.2/Python-3.14.2.tgz
tar xzf Python-3.14.2.tgz
cd Python-3.14.2

# Configure and build
./configure --enable-optimizations --with-ensurepip=install
make -j$(nproc)
sudo make altinstall

# Verify installation
python3.14 --version
```

## Environment Variables

Consider adding these to your `~/.bashrc` or `~/.profile`:

```bash
# Python 3.14
export PATH="/usr/bin:$PATH"  # Adjust if Python is installed elsewhere

# Virtual environment activation helper
alias activate-flux="cd ~/flux2-lora-toolkit && source venv314/bin/activate"
```

## Next Steps

After successful installation:

1. Download and prepare your training dataset
2. Configure the model path in the web interface (point to `~/models/black-forest-labs/FLUX.2-dev`)
3. Start training your LoRA models!

For any issues, check the logs and ensure all dependencies are properly installed.