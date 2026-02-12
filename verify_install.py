"""Quick start script to verify installation and run the app."""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.11 or higher."""
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required (you have {sys.version_info.major}.{sys.version_info.minor})")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'sklearn',
        'requests',
        'yaml'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✅ {package:20} OK")
        except ImportError:
            print(f"❌ {package:20} MISSING")
            missing.append(package)
    
    return len(missing) == 0, missing


def check_config_files():
    """Check if required configuration files exist."""
    project_dir = Path(__file__).parent
    required_files = [
        'configs/config.yaml',
        'app/streamlit_app.py',
        'launcher.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path:30} OK")
        else:
            print(f"❌ {file_path:30} MISSING")
            all_exist = False
    
    return all_exist


def check_api_key():
    """Check if API key is set."""
    api_key = os.getenv('MASSIVE_API_KEY')
    if api_key:
        masked = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
        print(f"✅ MASSIVE_API_KEY set: {masked}")
        return True
    else:
        print(f"⚠️  MASSIVE_API_KEY not set (will use mock data)")
        return False


def install_dependencies():
    """Install missing dependencies."""
    print("\n" + "=" * 50)
    print("Installing dependencies...")
    print("=" * 50 + "\n")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("\n✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies")
        return False


def main():
    """Run all checks."""
    print("=" * 50)
    print("Massive 40-Minute Forecaster - Verification")
    print("=" * 50 + "\n")
    
    print("📋 Checking Python environment...")
    print("-" * 50)
    if not check_python_version():
        print("\n❌ Python version check failed")
        return False
    
    print("\n📦 Checking dependencies...")
    print("-" * 50)
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nWould you like to install them now? (y/n)")
        if input().lower() == 'y':
            if not install_dependencies():
                return False
            # Re-check after installation
            deps_ok, missing = check_dependencies()
            if not deps_ok:
                print(f"\n❌ Still missing: {', '.join(missing)}")
                return False
        else:
            print("⚠️  Skipping installation. The app may not work without dependencies.")
    
    print("\n📁 Checking configuration files...")
    print("-" * 50)
    if not check_config_files():
        print("\n❌ Missing configuration files")
        return False
    
    print("\n🔐 Checking API key...")
    print("-" * 50)
    has_api_key = check_api_key()
    
    print("\n" + "=" * 50)
    print("✅ Verification complete!")
    print("=" * 50 + "\n")
    
    if has_api_key:
        print("You're all set! Run the app:")
        print("  python launcher.py")
    else:
        print("App is ready to run in MOCK MODE (demo data).")
        print("To use live data, set your API key:")
        print("  $env:MASSIVE_API_KEY=\"YOUR_KEY_HERE\"")
        print("\nThen run:")
        print("  python launcher.py")
    
    print("\nFor full documentation, see: README.md")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
