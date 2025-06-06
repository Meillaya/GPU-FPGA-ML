#!/usr/bin/env python3
"""
Development Setup Script for GPU/FPGA ML Kernel Project

This script sets up the development environment, installs dependencies,
and validates the installation.
"""

import subprocess
import sys
import os
from pathlib import Path
import platform


def run_command(cmd, check=True, capture_output=False):
    """Run a command and optionally capture output"""
    print(f"Running: {cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, 
                                   capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e}")
        if not check:
            return None
        sys.exit(1)


def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        print(f"ERROR: Python 3.9+ required, found {python_version.major}.{python_version.minor}")
        sys.exit(1)
    else:
        print(f"OK: Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check operating system
    os_name = platform.system()
    print(f"OK: Operating System: {os_name}")
    
    # Check if CUDA is available
    try:
        cuda_version = run_command("nvcc --version", check=False, capture_output=True)
        if cuda_version:
            print(f"OK: CUDA available")
        else:
            print("WARNING: CUDA not found - GPU features will be limited")
    except:
        print("WARNING: CUDA not found - GPU features will be limited")
    
    # Check if uv is available
    try:
        uv_version = run_command("uv --version", check=False, capture_output=True)
        if uv_version:
            print(f"OK: uv found: {uv_version}")
        else:
            print("ERROR: uv not found - please install uv first")
            print("Installation: https://github.com/astral-sh/uv")
            sys.exit(1)
    except:
        print("ERROR: uv not found - please install uv first")
        sys.exit(1)


def setup_virtual_environment():
    """Set up virtual environment with uv"""
    print("\nSetting up virtual environment...")
    
    # Create virtual environment
    run_command("uv venv")
    
    # Activate virtual environment (instructions for user)
    if platform.system() == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
    else:
        activate_cmd = "source .venv/bin/activate"
    
    print(f"OK: Virtual environment created")
    print(f"   To activate: {activate_cmd}")


def install_dependencies():
    """Install project dependencies"""
    print("\nInstalling dependencies...")
    
    # Install main dependencies
    run_command("uv pip install -e .")
    
    # Install development dependencies
    run_command("uv pip install -e \".[dev]\"")
    
    # Try to install FPGA dependencies (optional)
    try:
        run_command("uv pip install -e \".[fpga]\"", check=False)
        print("OK: FPGA dependencies installed")
    except:
        print("WARNING: FPGA dependencies not available - skipping")
    
    print("OK: Dependencies installed")


def validate_installation():
    """Validate the installation"""
    print("\nValidating installation...")
    
    # Test basic imports
    test_imports = [
        "torch",
        "triton", 
        "numpy",
        "matplotlib",
        "pytest"
    ]
    
    for package in test_imports:
        try:
            run_command(f"python -c \"import {package}; print(f'{package}: OK')\"")
        except:
            print(f"ERROR: Failed to import {package}")
            return False
    
    # Test CUDA availability
    try:
        result = run_command(
            "python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"",
            capture_output=True
        )
        print(f"OK: {result}")
    except:
        print("WARNING: Could not check CUDA availability")
    
    # Test project imports
    try:
        run_command("python -c \"from src.kernels.flash_attention import FlashAttention; print('Project imports: OK')\"")
        print("OK: Project imports working")
    except:
        print("ERROR: Project imports failed")
        return False
    
    return True


def run_quick_test():
    """Run a quick functionality test"""
    print("\nRunning quick functionality test...")
    
    test_script = """
import torch
from src.kernels.flash_attention import FlashAttention

# Test FlashAttention
device = 'cuda' if torch.cuda.is_available() else 'cpu'
flash_attn = FlashAttention(device=device, block_size=32)

# Create test data
q = torch.randn(2, 64, 32, device=device)
k = torch.randn(2, 64, 32, device=device)
v = torch.randn(2, 64, 32, device=device)

# Run forward pass
output = flash_attn.forward(q, k, v)

 print(f"OK: FlashAttention test passed")
 print(f"   Input shape: {q.shape}")
 print(f"   Output shape: {output.shape}")
 print(f"   Device: {device}")
    """
    
    # Write test script to temporary file
    with open("temp_test.py", "w") as f:
        f.write(test_script)
    
    try:
        run_command("python temp_test.py")
        print("OK: Quick test passed")
    except:
        print("ERROR: Quick test failed")
    finally:
        # Clean up
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")


def setup_development_tools():
    """Set up development tools and pre-commit hooks"""
    print("\nSetting up development tools...")
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.trace
*.prof
*.ncu-rep
*.nsys-rep
flashattention_trace/
kernel_trace/

# Jupyter
.ipynb_checkpoints/

# CUDA
*.cubin
*.fatbin
*.ptx
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    print("OK: .gitignore created")
    
    # Set up pre-commit (optional)
    try:
        run_command("uv pip install pre-commit", check=False)
        
        pre_commit_config = """
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
"""
        
        with open(".pre-commit-config.yaml", "w") as f:
            f.write(pre_commit_config.strip())
        
        run_command("pre-commit install", check=False)
        print("OK: Pre-commit hooks configured")
    except:
        print("WARNING: Pre-commit setup skipped")


def create_run_scripts():
    """Create convenient run scripts"""
    print("\nCreating run scripts...")
    
    # Create run_demo.sh for Unix systems
    if platform.system() != "Windows":
        run_demo_sh = """#!/bin/bash
source .venv/bin/activate
python run_demo.py "$@"
"""
        with open("run_demo.sh", "w") as f:
            f.write(run_demo_sh)
        os.chmod("run_demo.sh", 0o755)
        print("OK: run_demo.sh created")
    
    # Create run_demo.bat for Windows
    if platform.system() == "Windows":
        run_demo_bat = """@echo off
.venv\\Scripts\\activate
python run_demo.py %*
"""
        with open("run_demo.bat", "w") as f:
            f.write(run_demo_bat)
        print("OK: run_demo.bat created")
    
    # Create jupyter startup script
    jupyter_script = """#!/usr/bin/env python3
import subprocess
import sys

def main():
    try:
        subprocess.run([sys.executable, "-m", "jupyter", "lab", "notebooks/"], check=True)
    except KeyboardInterrupt:
        print("\\nJupyter Lab stopped")
    except subprocess.CalledProcessError:
        print("Error starting Jupyter Lab")
        print("Make sure it's installed: uv pip install jupyterlab")

if __name__ == "__main__":
    main()
"""
    
    with open("start_jupyter.py", "w") as f:
        f.write(jupyter_script)
    
    if platform.system() != "Windows":
        os.chmod("start_jupyter.py", 0o755)
    
    print("OK: start_jupyter.py created")


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("Setup complete! Next steps:")
    print("="*60)
    
    if platform.system() == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
        demo_cmd = "python run_demo.py"
    else:
        activate_cmd = "source .venv/bin/activate"
        demo_cmd = "./run_demo.sh"
    
    print(f"1. Activate virtual environment:")
    print(f"   {activate_cmd}")
    print()
    print(f"2. Run the demo:")
    print(f"   {demo_cmd}")
    print()
    print(f"3. Start Jupyter Lab:")
    print(f"   python start_jupyter.py")
    print()
    print(f"4. Run tests:")
    print(f"   pytest tests/")
    print()
    print("Key files to explore:")
    print("   - notebooks/flashattention_demo.ipynb")
    print("   - src/kernels/flash_attention.py")
    print("   - src/kernels/moe_routing.py")
    print("   - benchmarks/profiler.py")
    print()
    print("Useful commands:")
    print("   - python run_demo.py --help")
    print("   - python run_demo.py --seq-len 2048")
    print("   - python run_demo.py --num-experts 16")


def main():
    """Main setup function"""
    print("GPU/FPGA ML Kernel Project Setup")
    print("="*50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("ERROR: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    check_system_requirements()
    setup_virtual_environment()
    install_dependencies()
    
    if validate_installation():
        run_quick_test()
        setup_development_tools()
        create_run_scripts()
        print_next_steps()
        print("\nSetup completed successfully!")
    else:
        print("\nSetup failed during validation")
        sys.exit(1)


if __name__ == "__main__":
    main() 