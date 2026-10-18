"""Environment Verification Script.

Checks that all core packages for Phase 1 are installed properly,
prints their exact versions, and validates Python environment health.
"""

import sys
import platform

def check_environment() -> bool:
    print("=" * 60)
    print("CUSTOMER INTELLIGENCE PLATFORM - ENVIRONMENT HEALTH CHECK")
    print("=" * 60)
    print(f"Python Version : {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Platform       : {platform.platform()}")
    print(f"Interpreter    : {sys.executable}")
    print("-" * 60)

    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
        ("shap", "shap"),
        ("joblib", "joblib"),
        ("pytest", "pytest"),
    ]

    all_passed = True

    for import_name, display_name in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "Installed")
            print(f"[OK]   {display_name:<18} -> Version: {version}")
        except ImportError as exc:
            print(f"[FAIL] {display_name:<18} -> Error: {exc}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("RESULT: ALL REQUIRED PACKAGES ARE INSTALLED AND READY FOR PHASE 2!")
    else:
        print("RESULT: SOME PACKAGES FAILED TO IMPORT. PLEASE REVIEW ERRORS ABOVE.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
