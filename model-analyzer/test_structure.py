#!/usr/bin/env python3
"""Test project structure without installing dependencies."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")

    try:
        # Test basic imports
        print("✓ Testing __init__.py")
        import model_analyzer
        print(f"  - Version: {model_analyzer.__version__}")

        # Test parser module structure
        print("✓ Testing model_parser.py structure")
        with open(os.path.join(os.path.dirname(__file__), 'src/model_analyzer/model_parser.py')) as f:
            content = f.read()
            assert 'class ModelParser' in content
            assert 'class TFLiteParser' in content
            assert 'class ONNXParser' in content
            assert 'class OperatorInfo' in content
            print("  - All required classes found")

        # Test stats module structure
        print("✓ Testing operator_stats.py structure")
        with open(os.path.join(os.path.dirname(__file__), 'src/model_analyzer/operator_stats.py')) as f:
            content = f.read()
            assert 'class OperatorStats' in content
            assert 'class OperatorCategory' in content
            print("  - All required classes found")

        # Test exporter module structure
        print("✓ Testing stats_exporter.py structure")
        with open(os.path.join(os.path.dirname(__file__), 'src/model_analyzer/stats_exporter.py')) as f:
            content = f.read()
            assert 'class StatsExporter' in content
            print("  - StatsExporter class found")

        # Test CLI structure
        print("✓ Testing cli.py structure")
        with open(os.path.join(os.path.dirname(__file__), 'src/model_analyzer/cli.py')) as f:
            content = f.read()
            assert 'def main' in content
            assert 'def find_model_files' in content
            print("  - All required functions found")

        print("\n✅ All structure tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_files_exist():
    """Test that all required files exist."""
    print("\nTesting file existence...")

    required_files = [
        'src/model_analyzer/__init__.py',
        'src/model_analyzer/model_parser.py',
        'src/model_analyzer/operator_stats.py',
        'src/model_analyzer/stats_exporter.py',
        'src/model_analyzer/cli.py',
        'requirements.txt',
        'setup.py',
        'README.md',
        'PROJECT_GUIDE.md',
        'run.py',
        'create_demo_models.py',
        'tests/test_model_analyzer.py',
        'examples/analyze_models.py',
    ]

    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False

    return all_exist


def print_usage_examples():
    """Print usage examples."""
    print("\n" + "=" * 60)
    print("QUICK START EXAMPLES")
    print("=" * 60)

    print("\n1. Install dependencies:")
    print("   pip install tensorflow onnx onnxruntime pandas openpyxl")

    print("\n2. Analyze a single model:")
    print("   python3 run.py model.tflite")

    print("\n3. Analyze a directory:")
    print("   python3 run.py /path/to/models")

    print("\n4. Create demo models (requires TensorFlow):")
    print("   python3 create_demo_models.py")
    print("   python3 run.py demo_models")

    print("\n5. Run tests:")
    print("   python3 tests/test_model_analyzer.py")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Model Analyzer - Structure Test")
    print("=" * 60)

    # Test file existence
    files_ok = test_files_exist()

    # Test imports
    import_ok = test_imports()

    # Print usage
    print_usage_examples()

    if files_ok and import_ok:
        print("\n🎉 Project structure is OK!")
    else:
        print("\n⚠️  Some issues detected. Please check the errors above.")
        sys.exit(1)
