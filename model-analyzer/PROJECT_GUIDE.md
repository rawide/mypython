# Model Analyzer 项目使用指南

## 项目概述

Model Analyzer 是一个用于解析和分析 TensorFlow Lite (.tflite) 和 ONNX (.onnx) 模型文件的 Python 工具。它能够提取模型中所有算子的详细信息，包括算子类型、出现次数、输入输出形状等，并生成详细的统计报告。

## 核心特性

✅ **支持多种模型格式**
- TensorFlow Lite (.tflite)
- ONNX (.onnx)

✅ **详细的统计信息**
- 算子名称和类型
- 算子出现次数统计
- 输入输出形状分析
- 按形状区分的算子统计

✅ **批量处理**
- 支持单个文件或整个文件夹
- 自动递归查找模型文件
- 批量处理多个模型

✅ **丰富的输出格式**
- Excel 多 sheet 报告
- CSV 格式导出
- 每个模型独立 sheet
- 全局汇总 sheet

✅ **智能分类**
- 相同算子，不同形状 → 分别统计
- 算子大类 + shape 小类的分类方式
- 跨模型对比分析

## 项目结构

```
model-analyzer/
├── src/model_analyzer/
│   ├── __init__.py          # 包初始化
│   ├── model_parser.py      # 模型解析器（TFLite/ONNX）
│   ├── operator_stats.py    # 算子统计类
│   ├── stats_exporter.py    # 数据导出器
│   └── cli.py               # 命令行接口
├── tests/
│   └── test_model_analyzer.py  # 测试文件
├── examples/
│   └── analyze_models.py    # 使用示例
├── README.md                # 英文文档
├── PROJECT_GUIDE.md         # 项目指南（本文件）
├── requirements.txt         # 依赖包
├── setup.py                 # 安装脚本
├── run.py                   # 快速运行脚本
└── create_demo_models.py    # 创建演示模型
```

## 安装方法

### 1. 安装依赖

```bash
cd /home/rawide/code/model-analyzer
pip install -r requirements.txt
```

### 2. 安装为可执行程序（可选）

```bash
pip install -e .
```

## 快速开始

### 方法 1：使用命令行工具

```bash
# 分析单个模型
python run.py path/to/model.tflite

# 分析文件夹中的所有模型
python run.py path/to/model/directory

# 指定输出文件名
python run.py model.tflite -o my_stats.xlsx

# 输出为 CSV 格式
python run.py models/ -f csv --csv-dir output_csv/
```

### 方法 2：使用库 API

```python
from model_analyzer import ModelParser, OperatorStats, StatsExporter

# 解析模型
parser = ModelParser()
operators = parser.parse("model.tflite")

# 收集统计
stats = OperatorStats()
stats.add_model("my_model", operators)

# 导出报告
exporter = StatsExporter(stats)
exporter.export_to_excel("stats.xlsx")
```

## 输入输出说明

### 输入

- **单个模型文件**：`.tflite` 或 `.onnx` 文件
- **文件夹路径**：包含多个模型文件的目录（自动递归查找）

### 输出（Excel 格式）

生成的 Excel 文件包含多个 sheet：

#### 1. Global Summary（全局汇总）
- Operator Type: 算子类型（如 Conv2D、ReLU）
- Count: 该形状组合的出现次数
- Total Count (Type): 该算子类型的总次数
- Models Using: 使用该算子+形状的模型数量
- Example Input Shapes: 示例输入形状
- Example Output Shapes: 示例输出形状
- Shape Signature: 形状签名（唯一标识）

#### 2. Per-Model Sheets（按模型分sheet）
- Sheet 名称：模型文件名（截断到31字符）
- 内容：该模型的算子统计信息
- 格式：与 Global Summary 相同

#### 3. Operator Breakdown（算子分类汇总）
- Operator Type: 算子类型
- Total Count: 所有模型中该算子的总次数

#### 4. Metadata（元数据）
- Total Models Analyzed: 分析的模型总数
- Total Operators: 算子总数
- Unique Operator Types: 唯一算子类型数
- Unique Operator+Shape Combinations: 唯一算子+形状组合数

### 输出（CSV 格式）

- `global_summary.csv`: 全局统计
- `{model_name}.csv`: 每个模型的统计
- `operator_breakdown.csv`: 算子分类汇总

## 算子分类规则

工具对相同类型的算子会根据形状进行细分统计：

**示例：**
- Conv2D 算子，输入形状 `[1, 224, 224, 3]` → 输出 `[1, 112, 112, 64]`
- Conv2D 算子，输入形状 `[1, 112, 112, 64]` → 输出 `[1, 56, 56, 128]`

**分类结果：**
- 虽然都是 Conv2D 类型，但由于输入输出形状不同，会被统计为两个独立的类别
- 在 Excel 中显示为两行记录
- Category Name 分别为：`Conv2D_[1, 224, 224, 3]->[1, 112, 112, 64]` 和 `Conv2D_[1, 112, 112, 64]->[1, 56, 56, 128]`

## 进阶使用

### 1. 创建演示模型（需要 TensorFlow）

```bash
python create_demo_models.py
```

这会创建两个简单的演示模型：
- `demo_models/convnet.tflite` - 卷积神经网络
- `demo_models/mlp.tflite` - 多层感知机

### 2. 使用 Python API 进行自定义分析

```python
import os
from model_analyzer import ModelParser, OperatorStats

# 分析模型文件
parser = ModelParser()
model_path = "model.tflite"
operators = parser.parse(model_path)

# 自定义统计
print(f"Total operators: {len(operators)}")

# 按类型分组
op_types = {}
for op in operators:
    op_type = op.op_type
    if op_type not in op_types:
        op_types[op_type] = []
    op_types[op_type].append(op)

# 输出统计
for op_type, ops in sorted(op_types.items(),
                         key=lambda x: len(x[1]),
                         reverse=True):
    print(f"{op_type}: {len(ops)} occurrences")
```

### 3. 批量处理大量模型

```python
import os
from pathlib import Path
from model_analyzer import ModelParser, OperatorStats, StatsExporter

def batch_analyze_models(model_directory):
    """批量分析目录中的所有模型"""

    parser = ModelParser()
    stats = OperatorStats()

    # 查找所有模型文件
    extensions = parser.supported_extensions()
    model_files = []

    for ext in extensions:
        model_files.extend(Path(model_directory).rglob(f"*{ext}"))

    print(f"Found {len(model_files)} model files")

    # 分析每个模型
    for model_file in model_files:
        try:
            model_name = model_file.name
            operators = parser.parse(str(model_file))
            stats.add_model(model_name, operators)
            print(f"✓ {model_name}: {len(operators)} operators")
        except Exception as e:
            print(f"✗ {model_file}: {e}")

    # 导出结果
    exporter = StatsExporter(stats)
    exporter.export_to_excel("batch_analysis.xlsx")
    print(f"\nAnalysis complete! Processed {stats.get_total_models()} models")

# 使用
batch_analyze_models("/path/to/your/models")
```

## 常见问题

### Q1: 如何安装依赖？

```bash
pip install tensorflow onnx onnxruntime pandas openpyxl
```

### Q2: 支持哪些模型格式？

目前支持：
- TensorFlow Lite (.tflite)
- ONNX (.onnx)

### Q3: 如何处理大型模型？

工具会加载整个模型到内存中进行分析。对于大型模型：
- 确保有足够的内存
- 可以分批处理（将模型分目录存放）

### Q4: 如何解释输出结果？

主要关注：
1. **Operator Breakdown**：最常见的算子类型
2. **Models Using**：哪些算子在多个模型中出现
3. **Shape patterns**：输入输出的形状模式

### Q5: 如何获取形状的详细信息？

查看每个 sheet 中的：
- `Example Input Shapes` 列
- `Example Output Shapes` 列
- `Shape Signature` 列（唯一标识）

## 测试

运行测试脚本验证安装：

```bash
python tests/test_model_analyzer.py
```

## 故障排除

### 问题 1: ImportError: No module named 'tensorflow'

**解决方案**：
```bash
pip install tensorflow
```

### 问题 2: ImportError: No module named 'onnx'

**解决方案**：
```bash
pip install onnx onnxruntime
```

### 问题 3: 解析模型时出错

**可能原因**：
- 模型文件损坏
- 不支持的算子类型
- 内存不足

**解决方案**：
- 验证模型文件完整性
- 检查模型格式是否正确
- 尝试单个模型分析

### 问题 4: Excel 文件打开错误

**可能原因**：
- Sheet 名称过长（超过 31 字符）
- 文件名包含非法字符

**解决方案**：
- 工具会自动处理，如有问题请报告 issue
- 使用 CSV 格式作为替代

## 扩展开发

### 添加新的模型格式支持

1. 在 `model_parser.py` 中创建新的 Parser 类：

```python
class MyFormatParser(BaseParser):
    def parse(self, model_path: str) -> List[OperatorInfo]:
        # 实现解析逻辑
        pass
```

2. 在 `ModelParser` 类中注册：

```python
self.parsers = {
    '.tflite': TFLiteParser(),
    '.onnx': ONNXParser(),
    '.myformat': MyFormatParser(),  # 添加新格式
}
```

### 添加新的输出格式

1. 在 `stats_exporter.py` 中添加新方法
2. 在 `cli.py` 中添加格式选项

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 总结

Model Analyzer 是一个强大的模型分析工具，能够：

1. ✅ 解析 .tflite 和 .onnx 模型
2. ✅ 详细统计每个算子的类型、数量和形状
3. ✅ 区分相同算子的不同形状变体
4. ✅ 支持批量处理文件夹
5. ✅ 生成 Excel 多 sheet 报告
6. ✅ 提供 CSV 导出选项

使用简单，功能强大，是模型分析和优化的好帮手！
