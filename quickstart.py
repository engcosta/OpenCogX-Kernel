#!/usr/bin/env python
"""
Quick Start Script for AGI Kernel POC
=====================================

This script helps you get started quickly by:
1. Checking prerequisites
2. Setting up the environment
3. Running a demo
"""

import subprocess
import sys
import os


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_ollama():
    """Check if Ollama is running."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running ({len(models)} models)")
            return True
    except:
        pass
    print("⚠️  Ollama not running (optional, but recommended)")
    print("   Install: https://ollama.ai/download")
    print("   Then run: ollama pull llama3.2:3b")
    return False


def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"✅ Docker available")
            return True
    except:
        pass
    print("⚠️  Docker not found (optional, for Qdrant and Neo4j)")
    return False


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])


def setup_env():
    """Set up environment file."""
    if not os.path.exists(".env"):
        print("\n📝 Creating .env file...")
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Copied .env.example to .env")
        else:
            print("⚠️  No .env.example found")


def start_infrastructure():
    """Start Docker services if available."""
    print("\n🐳 Starting infrastructure...")
    try:
        subprocess.run(
            ["docker-compose", "up", "-d"],
            check=True,
        )
        print("✅ Qdrant and Neo4j started")
        return True
    except:
        print("⚠️  Could not start Docker services")
        print("   Run manually: docker-compose up -d")
        return False


def run_demo():
    """Run the demo."""
    print("\n🚀 Running demo...")
    subprocess.run([sys.executable, "-m", "agi_kernel", "demo"])


def main():
    """Main entry point."""
    print("=" * 50)
    print("🧠 AGI Kernel POC - Quick Start")
    print("=" * 50)
    
    print("\n1️⃣ Checking prerequisites...")
    
    if not check_python():
        sys.exit(1)
    
    check_ollama()
    has_docker = check_docker()
    
    print("\n2️⃣ Setting up environment...")
    install_dependencies()
    setup_env()
    
    if has_docker:
        print("\n3️⃣ Starting infrastructure...")
        start_infrastructure()
    
    print("\n4️⃣ Running demo...")
    run_demo()
    
    print("\n" + "=" * 50)
    print("🎉 Quick start complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("  • Start API server: python -m agi_kernel serve")
    print("  • Ingest documents: python -m agi_kernel ingest ./corpus")
    print("  • Run learning: python -m agi_kernel learn --iterations 20")
    print("  • View evaluation: python -m agi_kernel evaluate")


if __name__ == "__main__":
    main()
