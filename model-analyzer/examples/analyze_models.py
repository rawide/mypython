#!/usr/bin/env python3
"""Example script demonstrating how to use the Model Analyzer library."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from model_analyzer import ModelParser, OperatorStats, StatsExporter


def example_single_model():
    """Example: Analyze a single model."""
    print("=" * 60)
    print("Example 1: Analyzing a single model")
    print("=" * 60)

    # Initialize parser
    parser = ModelParser()

    # Example: Analyze a .tflite model
    try:
        model_path = "path/to/your/model.tflite"
        print(f"Parsing: {model_path}")

        operators = parser.parse(model_path)
        print(f"Found {len(operators)} operators")

        # Collect statistics
        stats = OperatorStats()
        stats.add_model("my_model", operators)

        # Print summary
        print("\nOperator Summary:")
        for op in operators[:5]:  # Show first 5 operators
            print(f"  - {op.op_type}: {op.get_shape_signature()}")

        # Export to Excel
        exporter = StatsExporter(stats)
        exporter.export_to_excel("single_model_stats.xlsx")
        print("\nExported to: single_model_stats.xlsx")

    except FileNotFoundError:
        print("Model file not found! Skipping this example.")


def example_multiple_models():
    """Example: Analyze multiple models and compare."""
    print("\n" + "=" * 60)
    print("Example 2: Analyzing multiple models")
    print("=" * 60)

    parser = ModelParser()
    stats = OperatorStats()

    # Analyze multiple models (assuming they exist)
    model_files = [
        "model1.tflite",
        "model2.onnx",
        "model3.tflite"
    ]

    for model_file in model_files:
        try:
            model_name = os.path.basename(model_file)
            print(f"Analyzing: {model_file}")

            operators = parser.parse(model_file)
            stats.add_model(model_name, operators)
            print(f"  Found {len(operators)} operators")

        except FileNotFoundError:
            print(f"  Skipping {model_file}: File not found")
        except Exception as e:
            print(f"  Error analyzing {model_file}: {e}")

    if stats.get_total_models() > 0:
        print(f"\nAnalyzed {stats.get_total_models()} models")
        print(f"Total operators: {stats.get_total_operators()}")

        # Print global operator breakdown
        breakdown = stats.get_operator_breakdown()
        print("\nOperator Breakdown:")
        for op_type, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {op_type}: {count}")

        # Export comprehensive report
        exporter = StatsExporter(stats)
        exporter.export_to_excel("multi_model_comparison.xlsx")
        print("\nExported comprehensive report to: multi_model_comparison.xlsx")


def example_directory_analysis():
    """Example: Analyze all models in a directory."""
    print("\n" + "=" * 60)
    print("Example 3: Analyzing a directory of models")
    print("=" * 60)

    # Find and analyze all models in a directory
    directory = "path/to/model/directory"

    if os.path.exists(directory):
        parser = ModelParser()
        stats = OperatorStats()

        # Find all .tflite and .onnx files
        extensions = parser.supported_extensions()
        model_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    model_files.append(os.path.join(root, file))

        print(f"Found {len(model_files)} model files")

        for model_file in model_files:
            try:
                model_name = os.path.basename(model_file)
                operators = parser.parse(model_file)
                stats.add_model(model_name, operators)
                print(f"  {model_name}: {len(operators)} operators")
            except Exception as e:
                print(f"  Error with {model_file}: {e}")

        # Export results
        if stats.get_total_models() > 0:
            exporter = StatsExporter(stats)

            # Export to Excel
            exporter.export_to_excel("directory_analysis.xlsx")

            # Also export to CSV
            exporter.export_to_csv("directory_csv_output")

            print("\nExported analysis to:")
            print("  - directory_analysis.xlsx")
            print("  - directory_csv_output/")
    else:
        print(f"Directory '{directory}' does not exist. Skipping this example.")


def example_custom_analysis():
    """Example: Custom analysis using the API."""
    print("\n" + "=" * 60)
    print("Example 4: Custom analysis")
    print("=" * 60)

    parser = ModelParser()

    # Analyze a model
    try:
        model_path = "example_model.tflite"
        operators = parser.parse(model_path)

        # Custom analysis: Count specific operator types
        conv_ops = [op for op in operators if 'conv' in op.op_type.lower()]
        fc_ops = [op for op in operators if 'dense' in op.op_type.lower() or 'fc' in op.op_type.lower()]
        activation_ops = [op for op in operators if op.op_type.lower() in ['relu', 'relu6', 'sigmoid', 'tanh']]

        print(f"Convolution ops: {len(conv_ops)}")
        print(f"Fully connected ops: {len(fc_ops)}")
        print(f"Activation ops: {len(activation_ops)}")

        # Analyze input shapes
        input_shapes = {}
        for op in operators:
            for shape in op.input_shapes:
                if shape:
                    shape_key = str(shape)
                    input_shapes[shape_key] = input_shapes.get(shape_key, 0) + 1

        print("\nInput Shape Distribution:")
        for shape, count in sorted(input_shapes.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {shape}: {count} times")

    except FileNotFoundError:
        print("Example model not found. Skipping custom analysis.")


if __name__ == "__main__":
    print("Model Analyzer - Examples")
    print("========================\n")

    # Run examples
    example_single_model()
    example_multiple_models()
    example_directory_analysis()
    example_custom_analysis()

    print("\n" + "=" * 60)
    print("Example script complete!")
    print("\nNote: Please update the model paths in this script")
    print("      to point to your actual model files.")
    print("=" * 60)
