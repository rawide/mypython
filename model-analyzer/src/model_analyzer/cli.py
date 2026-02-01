"""Command-line interface for Model Analyzer."""

import os
import sys
import argparse
from pathlib import Path
from typing import List

from .model_parser import ModelParser
from .operator_stats import OperatorStats
from .stats_exporter import StatsExporter


def find_model_files(directory: str, extensions: List[str]) -> List[str]:
    """Recursively find all model files with given extensions."""
    model_files = []
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for ext in extensions:
        # Case-insensitive search
        pattern = f"**/*{ext.lower()}"
        model_files.extend([str(f) for f in directory.glob(pattern)])

        if ext.lower() != ext.upper():
            pattern = f"**/*{ext.upper()}"
            model_files.extend([str(f) for f in directory.glob(pattern)])

    return sorted(list(set(model_files)))  # Remove duplicates


def analyze_model(parser: ModelParser, model_path: str, model_name: str = None):
    """Analyze a single model file."""
    if model_name is None:
        model_name = os.path.basename(model_path)

    print(f"Analyzing: {model_path}")

    try:
        operators = parser.parse(model_path)
        print(f"  Found {len(operators)} operators")
        return model_name, operators
    except Exception as e:
        print(f"  Error analyzing {model_path}: {str(e)}")
        return model_name, []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze TensorFlow Lite and ONNX models to extract operator statistics"
    )

    parser.add_argument(
        "input",
        help="Path to model file (.tflite or .onnx) or directory containing models"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output path (default: operator_stats.xlsx in current directory)",
        default="operator_stats.xlsx"
    )

    parser.add_argument(
        "-f", "--format",
        help="Output format: excel (default) or csv",
        choices=["excel", "csv"],
        default="excel"
    )

    parser.add_argument(
        "--csv-dir",
        help="Directory for CSV output (default: output_csv)",
        default="output_csv"
    )

    args = parser.parse_args()

    # Initialize parser
    model_parser = ModelParser()
    stats = OperatorStats()

    # Check if input is a file or directory
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    # Collect model files
    model_files = []
    if input_path.is_file():
        if input_path.suffix.lower() in model_parser.supported_extensions():
            model_files = [(args.input, input_path.name)]
        else:
            print(f"Error: Unsupported file format: {input_path.suffix}")
            print(f"Supported formats: {', '.join(model_parser.supported_extensions())}")
            sys.exit(1)
    else:
        # Directory - find all model files
        print(f"Scanning directory: {args.input}")
        extensions = model_parser.supported_extensions()
        found_files = find_model_files(args.input, extensions)

        if not found_files:
            print(f"No model files found in: {args.input}")
            print(f"Looking for extensions: {', '.join(extensions)}")
            sys.exit(1)

        model_files = [(f, Path(f).name) for f in found_files]
        print(f"Found {len(model_files)} model files\n")

    # Analyze each model
    for model_path, model_name in model_files:
        name, operators = analyze_model(model_parser, model_path, model_name)
        if operators:
            stats.add_model(name, operators)

    # Check if we have any data
    if stats.get_total_models() == 0:
        print("\nNo models were successfully analyzed. Exiting.")
        sys.exit(1)

    # Generate report
    print(f"\nAnalyzed {stats.get_total_models()} models")
    print(f"Found {stats.get_total_operators()} total operators")
    print(f"Found {stats.get_unique_operator_types()} unique operator types")
    print(f"Found {stats.get_unique_shape_combinations()} unique operator+shape combinations\n")

    exporter = StatsExporter(stats)

    if args.format == "csv":
        output_dir = args.csv_dir
        exporter.export_to_csv(output_dir)
    else:
        output_path = args.output
        exporter.export_to_excel(output_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
