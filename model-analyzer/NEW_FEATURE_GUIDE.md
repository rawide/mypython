# New Attribute-Based Operator Classification

## Overview

The Model Analyzer has been updated to use a new operator classification system based on operator **attributes** rather than input/output shapes.

## Classification Logic

### 1. Operator Categories

#### **Convolution Operators (Conv2D)**
- **Major Category**: `Conv2D`
- **Classification Attributes**:
  - `ci` (input channels)
  - `co` (output channels)
  - `kw` (kernel width)
  - `kh` (kernel height)
  - `stride_h` (horizontal stride)
  - `stride_w` (vertical stride)

**Example:**
```
Conv2D(ci=3, co=64, kw=3, kh=3, sh=2, sw=2) → One category
Conv2D(ci=64, co=128, kw=3, kh=3, sh=2, sw=2) → Different category (ci/co different)
Conv2D(ci=3, co=64, kw=5, kh=5, sh=2, sw=2) → Different category (kernel size different)
```

#### **Pooling Operators**
- **Major Category**: `MaxPool` or `AvgPool`
- **Classification Attributes**:
  - `type` (pooling type: max/avg)
  - `kw` (pool width)
  - `kh` (pool height)
  - `stride_h` (horizontal stride)
  - `stride_w` (vertical stride)

**Example:**
```
MaxPool(type=max, kw=2, kh=2, sh=2, sw=2) → One category
MaxPool(type=max, kw=4, kh=4, sh=4, sw=4) → Different category (pool size different)
AvgPool(type=avg, kw=2, kh=2, sh=2, sw=2) → Different major category (avg vs max)
```

#### **Dense/Fully Connected Operators**
- **Major Category**: `Dense`
- **Classification Attributes**:
  - `in` (input features)
  - `out` (output features)

**Example:**
```
Dense(in=128, out=64) → One category
Dense(in=128, out=10) → Different category (output size different)
```

#### **Other Operators**
- For operators like ReLU, Sigmoid, etc., full input/output shapes are used as fallback
- These typically don't have variable parameters

### 2. Signature Format

The **Shape Signature** (now more accurately called **Attribute Signature**) is generated as:

**For Conv2D:**
```
ci=3,co=64,kw=3,kh=3,sh=2,sw=2
```

**For Pooling:**
```
type=max,kw=2,kh=2,sh=2,sw=2
```

**For Dense:**
```
in=128,out=64
```

### 3. Output Format

The Excel/CSV output has been updated with new columns:

| Column | Description |
|--------|-------------|
| **Operator Type** | The operator major category (e.g., Conv2D, MaxPool) |
| **Attributes** | Key operator attributes formatted for display |
| **Count** | Number of occurrences of this specific attribute combination |
| **Total Count (Type)** | Total occurrences of this operator type across all models |
| **Models Using** | (Global Summary only) Number of models containing this operator+attributes |
| **Category** | The full attribute signature |

#### Example Output:

| Operator Type | Attributes | Count | Total Count (Type) | Category |
|--------------|------------|-------|-------------------|----------|
| Conv2D | ci=3,co=64,kw=3,kh=3,sh=2,sw=2 | 5 | 15 | ci=3,co=64,kw=3,kh=3,sh=2,sw=2 |
| Conv2D | ci=64,co=128,kw=3,kh=3,sh=2,sw=2 | 10 | 15 | ci=64,co=128,kw=3,kh=3,sh=2,sw=2 |
| MaxPool | type=max,kw=2,kh=2,sh=2,sw=2 | 3 | 8 | type=max,kw=2,kh=2,sh=2,sw=2 |
| MaxPool | type=max,kw=4,kh=4,sh=4,sw=4 | 5 | 8 | type=max,kw=4,kh=4,sh=4,sw=4 |
| AvgPool | type=avg,kw=2,kh=2,sh=2,sw=2 | 2 | 2 | type=avg,kw=2,kh=2,sh=2,sw=2 |

### 4. Key Changes from Previous Version

#### Before (Shape-based):
- Classified based on input/output tensor dimensions
- Conv2D with different spatial dimensions were different categories
- Hard to compare architectural decisions

#### After (Attribute-based):
- Classified based on operator parameters
- Conv2D with same kernel/stride but different spatial sizes are same category
- Better for analyzing model design patterns

### 5. Usage Examples

#### Find all Conv2D with 3x3 kernel:
```python
import pandas as pd

df = pd.read_excel("stats.xlsx", sheet_name="Global Summary")
conv3x3 = df[df['Category'].str.contains('kw=3,kh=3')]
print(f"Found {len(conv3x3)} different 3x3 Conv2D configurations")
```

#### Compare pooling strategies:
```python
maxpool = df[df['Operator Type'] == 'MaxPool']
avgpool = df[df['Operator Type'] == 'AvgPool']

print(f"MaxPool occurrences: {maxpool['Count'].sum()}")
print(f"AvgPool occurrences: {avgpool['Count'].sum()}")
```

#### Find bottleneck layers (large channel reduction):
```python
# Conv2D where input channels > 3 * output channels
bottlenecks = df[
    (df['Operator Type'] == 'Conv2D') &
    df['Category'].str.contains('ci=(\d+)') &
    df['Category'].str.contains('co=(\d+)')
]
# Parse ci and co values to find bottlenecks
```

### 6. Testing

Run the test script to verify the new classification:
```bash
cd /home/rawide/code/python/model-analyzer
python3 test_new_features.py
```

### 7. Backward Compatibility

- The core API remains the same
- Output file format has new columns but maintains the same multi-sheet structure
- Old scripts using the CLI will continue to work
- Programmatic users may need to update column references

### 8. Future Enhancements

Potential extensions to the classification system:
- Add dilation parameter for Conv2D
- Add padding mode information
- Add activation functions fused with Conv2D
- Support for grouped convolutions
- Depthwise separable convolution tracking

## Summary

This new attribute-based classification provides:
- ✅ More meaningful operator categories
- ✅ Better insight into model architecture patterns
- ✅ Easier comparison across different models
- ✅ Focus on design decisions rather than input sizes
- ✅ Support for all major operator types
