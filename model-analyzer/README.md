# Model Analyzer

A Python tool for analyzing TensorFlow Lite (.tflite) and ONNX (.onnx) models to extract detailed operator statistics based on operator **attributes** (not just shapes).

## Features

- **Attribute-Based Classification**: Statistics based on operator parameters:
  - Conv2D: classified by ci, co, kw, kh, stride (not by input spatial dimensions)
  - Pooling: MaxPool vs AvgPool are different major categories
  - Pooling: different pool sizes are different sub-categories
  - Dense: classified by input/output feature counts
- **Multi-format Support**: Parse both TensorFlow Lite (.tflite) and ONNX (.onnx) models
- **Batch Processing**: Analyze single files or entire directories of models
- **Detailed Statistics**: Extract operator types, counts, and key attributes
- **Excel Export**: Generate multi-sheet Excel reports with:
  - Global summary across all models
  - Per-model operator statistics
  - Operator breakdown by type
  - Metadata summary
- **CSV Export**: Alternative CSV output for further analysis

## Installation

1. Clone or download the project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Usage

### Command Line Interface

#### Analyze a single model:

```bash
# Analyze a single TFLite model
python -m model_analyzer.cli model.tflite

# Analyze a single ONNX model
python -m model_analyzer.cli model.onnx

# Specify custom output path
python -m model_analyzer.cli model.tflite -o my_stats.xlsx
```

#### Analyze a directory of models:

```bash
# Analyze all .tflite and .onnx files in a directory
python -m model_analyzer.cli /path/to/model/directory

# The tool will recursively find all supported model files
```

#### Export to CSV instead of Excel:

```bash
# Export to CSV format
python -m model_analyzer.cli model.tflite -f csv

# Specify custom CSV output directory
python -m model_analyzer.cli /path/to/models -f csv --csv-dir my_output
```

### Programmatic Usage

```python
from model_analyzer import ModelParser, OperatorStats, StatsExporter

# Parse a model
parser = ModelParser()
operators = parser.parse("model.tflite")

# Collect statistics
stats = OperatorStats()
stats.add_model("model_name", operators)

# Export to Excel
exporter = StatsExporter(stats)
exporter.export_to_excel("stats.xlsx")

# Export to CSV
exporter.export_to_csv("output_dir")
```

## Output Format

### Excel Output (.xlsx)

The Excel file contains multiple sheets:

1. **Global Summary**: Statistics across all analyzed models
   - Operator Type: Major operator category (e.g., Conv2D, MaxPool, Dense)
   - Attributes: Key operator attributes (ci, co, kw, kh, stride for Conv2D; etc.)
   - Count: Number of occurrences with these specific attributes
   - Total Count (Type): Total occurrences of this operator type (sum across all attribute variations)
   - Models Using: Number of models containing this operator+attributes combination
   - Category: Full attribute signature (unique identifier)

2. **Per-Model Sheets**: One sheet for each model (named by model filename)
   - Similar columns as Global Summary but focused on a single model
   - Shows all operator categories present in that specific model

3. **Operator Breakdown**: Summary count per operator type
   - Operator Type: Operator major category
   - Total Count: Total occurrences across all models (sum of all Count values)

4. **Metadata**: Analysis metadata including:
   - Total models analyzed
   - Total operators found
   - Unique operator types
   - Unique operator+attribute combinations

### CSV Output

When using CSV format, multiple files are created:

- `global_summary.csv`: Global statistics with all columns
- `{model_name}.csv`: Per-model statistics (one file per model)
- `operator_breakdown.csv`: Operator type counts across all models

## Operator Classification

### Conv2D Operators
Classified by: **input channels (ci), output channels (co), kernel width (kw), kernel height (kh), stride**

Example signatures:
- `ci=3,co=64,kw=3,kh=3,sh=2,sw=2` - Standard 3x3 Conv2D with stride 2
- `ci=64,co=128,kw=1,kh=1,sh=1,sw=1` - 1x1 bottleneck Conv2D

**Note**: Input spatial dimensions (e.g., 224x224 vs 112x112) are NOT used for classification. Conv2D with same ci, co, kw, kh, stride are grouped together regardless of input size.

### Pooling Operators
**MaxPool** and **AvgPool** are different major categories (not grouped together).

Each pool type is further classified by: **pool size (kw, kh) and stride**

Example signatures:
- `type=max,kw=2,kh=2,sh=2,sw=2` - 2x2 MaxPool with stride 2
- `type=max,kw=4,kh=4,sh=4,sw=4` - 4x4 MaxPool with stride 4
- `type=avg,kw=2,kh=2,sh=2,sw=2` - 2x2 AvgPool with stride 2

### Dense/Fully Connected Operators
Classified by: **input features (in), output features (out)**

Example signatures:
- `in=128,out=64` - Dense layer reducing from 128 to 64 features
- `in=64,out=10` - Final classification layer (64 to 10 classes)

## Examples

### Analyze MobileNet and ResNet models

```bash
# Assuming you have .tflite files
python -m model_analyzer.cli /path/to/pretrained_models -o pretrained_stats.xlsx
```

### Batch analysis with CSV output

```bash
python -m model_analyzer.cli /path/to/model_zoo -f csv --csv-dir zoo_stats
```

## Supported Formats

- TensorFlow Lite: `.tflite`
- ONNX: `.onnx`

## Requirements

- Python 3.7+
- TensorFlow 2.8+ (for TFLite support)
- ONNX 1.12+ (for ONNX support)
- pandas, openpyxl (for Excel export)

## License

MIT License
