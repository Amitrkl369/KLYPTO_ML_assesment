"""
Setup and Verification Script
Run this script to verify your installation and setup
"""

import sys
import os

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version {version.major}.{version.minor} is too old. Need 3.8+")
        return False

def check_directories():
    """Check if all required directories exist"""
    dirs = ['data', 'notebooks', 'src', 'models', 'results', 'plots']
    all_exist = True
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory exists: {dir_name}/")
        else:
            print(f"✗ Directory missing: {dir_name}/")
            all_exist = False
    
    return all_exist

def check_modules():
    """Check if required modules are installed"""
    required_modules = [
        'pandas', 'numpy', 'scipy', 'sklearn', 'xgboost', 
        'tensorflow', 'hmmlearn', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ Module installed: {module}")
        except ImportError:
            print(f"✗ Module missing: {module}")
            missing.append(module)
    
    if missing:
        print(f"\n⚠ Missing modules: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_notebooks():
    """Check if all notebooks exist"""
    notebooks = [
        '01_data_acquisition.ipynb',
        '02_data_cleaning.ipynb',
        '03_feature_engineering.ipynb',
        '04_regime_detection.ipynb',
        '05_baseline_strategy.ipynb',
        '06_ml_models.ipynb',
        '07_outlier_analysis.ipynb'
    ]
    
    all_exist = True
    
    for nb in notebooks:
        path = os.path.join('notebooks', nb)
        if os.path.exists(path):
            print(f"✓ Notebook exists: {nb}")
        else:
            print(f"✗ Notebook missing: {nb}")
            all_exist = False
    
    return all_exist

def check_source_files():
    """Check if all source modules exist"""
    modules = [
        'data_utils.py', 'features.py', 'greeks.py',
        'regime.py', 'strategy.py', 'backtest.py', 'ml_models.py'
    ]
    
    all_exist = True
    
    for module in modules:
        path = os.path.join('src', module)
        if os.path.exists(path):
            print(f"✓ Module exists: {module}")
        else:
            print(f"✗ Module missing: {module}")
            all_exist = False
    
    return all_exist

def main():
    """Run all checks"""
    print("=" * 80)
    print("QUANTITATIVE TRADING SYSTEM - SETUP VERIFICATION")
    print("=" * 80)
    
    print("\n1. Checking Python Version...")
    print("-" * 80)
    py_ok = check_python_version()
    
    print("\n2. Checking Directory Structure...")
    print("-" * 80)
    dirs_ok = check_directories()
    
    print("\n3. Checking Installed Modules...")
    print("-" * 80)
    modules_ok = check_modules()
    
    print("\n4. Checking Jupyter Notebooks...")
    print("-" * 80)
    notebooks_ok = check_notebooks()
    
    print("\n5. Checking Source Code Modules...")
    print("-" * 80)
    source_ok = check_source_files()
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_checks = [
        ("Python Version", py_ok),
        ("Directory Structure", dirs_ok),
        ("Required Modules", modules_ok),
        ("Jupyter Notebooks", notebooks_ok),
        ("Source Code", source_ok)
    ]
    
    for name, status in all_checks:
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}: {'PASS' if status else 'FAIL'}")
    
    if all(status for _, status in all_checks):
        print("\n🎉 All checks passed! You're ready to start.")
        print("\nNext steps:")
        print("1. Update API credentials in notebooks/01_data_acquisition.ipynb")
        print("2. Run: jupyter notebook")
        print("3. Execute notebooks in sequence (01 through 07)")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing modules: pip install -r requirements.txt")
        print("- Ensure you're in the project root directory")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
