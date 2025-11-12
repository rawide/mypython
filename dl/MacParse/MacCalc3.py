#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tflite_mac_analyzer.py
解析 .tflite 模型，导出每层（算子）的输入/权重/输出形状与 MAC 到 CSV。

依赖：
  - flatbuffers
  - 使用 flatc --python 由 schema.fbs 生成的 tflite/ 包（与脚本同目录或在 PYTHONPATH 中）

用法：
  python tflite_mac_analyzer.py model.tflite -o report.csv
可选：
  --include-noncompute  也输出激活/逐元素/池化等非乘加算子（MAC=0）
"""

import os
import sys
import csv
import argparse
from typing import List, Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import flatbuffers  # noqa: F401
    from tflite.Model import Model
    from tflite.BuiltinOperator import BuiltinOperator as BO
    from tflite.BuiltinOptions import BuiltinOptions as BOT
    from tflite.Conv2DOptions import Conv2DOptions
    from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
    from tflite.FullyConnectedOptions import FullyConnectedOptions
    from tflite.Pool2DOptions import Pool2DOptions
except Exception as e:
    print("导入 tflite 生成代码失败，请确认已用 flatc --python 生成 tflite/ 包。错误：", e)
    sys.exit(1)

# --- 可读的算子名映射 ---
OP_NAME = {
    BO.CONV_2D: "Conv2D",
    BO.DEPTHWISE_CONV_2D: "DepthwiseConv2D",
    BO.FULLY_CONNECTED: "FullyConnected",
    BO.MAX_POOL_2D: "MaxPool2D",
    BO.AVERAGE_POOL_2D: "AveragePool2D",
    BO.L2_POOL_2D: "L2Pool2D",
    BO.ADD: "Add",
    BO.MUL: "Mul",
    BO.RELU: "ReLU",
    BO.RELU6: "ReLU6",
    BO.PRELU: "PReLU",
    BO.LOGISTIC: "Sigmoid",
    BO.TANH: "Tanh",
    BO.HARD_SWISH: "HardSwish",
    BO.BATCH_TO_SPACE_ND: "BatchToSpaceND",
    BO.SPACE_TO_BATCH_ND: "SpaceToBatchND",
    BO.SPACE_TO_DEPTH: "SpaceToDepth",
    BO.DEPTH_TO_SPACE: "DepthToSpace",
    BO.CONCATENATION: "Concat",
    BO.RESHAPE: "Reshape",
    BO.SQUEEZE: "Squeeze",
    BO.PAD: "Pad",
    BO.PADV2: "PadV2",
    BO.STRIDED_SLICE: "StridedSlice",
    BO.SLICE: "Slice",
    BO.TRANSPOSE: "Transpose",
    BO.MEAN: "Mean",
    BO.SUM: "Sum",
    BO.MAXIMUM: "Maximum",
    BO.MINIMUM: "Minimum",
    BO.SOFTMAX: "Softmax",
}

# 默认不计 MAC、也不输出的类型（可用 --include-noncompute 开关包含）
# 注意：ADD 在此移除，由专门的 Shortcut 规则处理
ACTIVATION_OR_NONCOMPUTE = {
    "ReLU", "ReLU6", "PReLU", "Sigmoid", "Tanh", "HardSwish",
    "Mul", "Maximum", "Minimum",
    "Reshape", "Squeeze", "Concat", "Pad", "PadV2", "Slice", "StridedSlice",
    "Transpose", "DepthToSpace", "SpaceToDepth", "BatchToSpaceND", "SpaceToBatchND",
    "Mean", "Sum", "Softmax",
    # 池化通常不算 MAC；多数人仍希望看到其尺寸，这里默认视为 noncompute
    "MaxPool2D", "AveragePool2D", "L2Pool2D",
}

def tshape(tensor) -> List[int]:
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]

def get_tensor(subgraph, idx):
    return subgraph.Tensors(idx) if (idx is not None and idx >= 0) else None

def shape_by_index(subgraph, idx) -> List[int]:
    t = get_tensor(subgraph, idx)
    return tshape(t) if t is not None else []

def as_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def nhwc_from_shape(shape: List[int]) -> Tuple[int, int, int, int]:
    # 以 NHWC 解读
    if len(shape) >= 4:
        n, h, w, c = shape[-4], shape[-3], shape[-2], shape[-1]
    elif len(shape) == 3:
        n, h, w, c = 1, shape[0], shape[1], shape[2]
    elif len(shape) == 2:
        n, h, w, c = shape[0], 1, 1, shape[1]
    elif len(shape) == 1:
        n, h, w, c = 1, 1, 1, shape[0]
    else:
        n, h, w, c = 1, 1, 1, 1
    return n, h, w, c

def stride_to_str(sh: int, sw: int) -> str:
    return f"{sh}" if sh == sw else f"{sh}x{sw}"

def compute_mac_conv2d(cin, cout, hout, wout, kh, kw) -> int:
    return as_int(hout) * as_int(wout) * as_int(cout) * as_int(cin) * as_int(kh) * as_int(kw)

def compute_mac_depthwise(cin, cout, hout, wout, kh, kw) -> int:
    # cout = cin * depth_multiplier
    return as_int(hout) * as_int(wout) * as_int(cout) * as_int(kh) * as_int(kw)

def compute_mac_fc(nin, nout) -> int:
    return as_int(nin) * as_int(nout)

def is_shortcut_add(op, subgraph, out_shape: List[int]) -> bool:
    """判断 ADD 是否为残差 shortcut：两输入与输出形状一致"""
    if op.InputsLength() < 2:
        return False
    in0 = shape_by_index(subgraph, op.Inputs(0))
    in1 = shape_by_index(subgraph, op.Inputs(1))
    return (in0 == in1 == out_shape) and (len(out_shape) >= 3)

def main():
    ap = argparse.ArgumentParser(description="TFLite 模型算子与 MAC 统计导出 CSV")
    ap.add_argument("tflite_model", type=str, help="模型路径 .tflite")
    ap.add_argument("-o", "--output", type=str, default="mac_report.csv", help="输出 CSV 路径")
    ap.add_argument("--include-noncompute", action="store_true",
                    help="将激活/逐元素/池化等非乘加算子也输出（MAC=0）")
    args = ap.parse_args()

    with open(args.tflite_model, "rb") as f:
        buf = f.read()

    model = Model.GetRootAsModel(buf, 0)
    if model.SubgraphsLength() <= 0:
        print("模型中没有子图。")
        sys.exit(1)

    sub = model.Subgraphs(0)

    rows = []
    running_idx = 1

    for oi in range(sub.OperatorsLength()):
        op = sub.Operators(oi)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        code = op_code.BuiltinCode()
        op_readable = OP_NAME.get(code, f"OP_{int(code)}")

        # I/O
        in_idx  = op.Inputs(0) if op.InputsLength() > 0 else None
        out_idx = op.Outputs(0) if op.OutputsLength() > 0 else None
        in_shape  = shape_by_index(sub, in_idx)
        out_shape = shape_by_index(sub, out_idx)

        _, Hin, Win, Cin = nhwc_from_shape(in_shape)
        _, Hout, Wout, Cout = nhwc_from_shape(out_shape)

        # 默认值
        kw_val = 1
        kh_val = 1
        sh = 1
        sw = 1
        mac = 0
        emit = True  # 是否输出该行

        bot = op.BuiltinOptionsType()

        # ---- Conv2D ----
        if code == BO.CONV_2D and bot == BOT.Conv2DOptions:
            opts = Conv2DOptions()
            opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            sh, sw = opts.StrideH(), opts.StrideW()

            # 权重：输入序 0=input,1=weights,2=bias
            w_idx = op.Inputs(1) if op.InputsLength() > 1 else None
            w_shape = shape_by_index(sub, w_idx)
            # 明确使用 TFLite Conv2D 布局： [kh, kw, Cin, Cout]
            if len(w_shape) == 4:
                kh_val = w_shape[1]
                kw_val = w_shape[1]
                # 补 Cin/Cout（在 FC 展平路径下可能拿不到）
                if Cin <= 0:
                    Cin = w_shape[2]
                if Cout <= 0:
                    Cout = w_shape[3]

            mac = compute_mac_conv2d(Cin, Cout, Hout, Wout, kh_val, kw_val)

        # ---- DepthwiseConv2D ----
        elif code == BO.DEPTHWISE_CONV_2D and bot == BOT.DepthwiseConv2DOptions:
            opts = DepthwiseConv2DOptions()
            opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            sh, sw = opts.StrideH(), opts.StrideW()
            depth_multiplier = 1
            try:
                depth_multiplier = max(1, opts.DepthMultiplier())
            except Exception:
                pass

            w_idx = op.Inputs(1) if op.InputsLength() > 1 else None
            w_shape = shape_by_index(sub, w_idx)
            # TFLite DepthwiseConv2D: [kh, kw, Cin, depth_multiplier]
            if len(w_shape) == 4:
                kh_val = w_shape[1]
                kw_val = w_shape[1]
                if Cin <= 0:
                    Cin = w_shape[2]
                if Cout <= 0:
                    Cout = Cin * (w_shape[3] if w_shape[3] > 0 else depth_multiplier)
            else:
                # 兜底 Cout
                if Cout <= 0 and Cin > 0:
                    Cout = Cin * depth_multiplier

            mac = compute_mac_depthwise(Cin, Cout, Hout, Wout, kh_val, kw_val)

        # ---- FullyConnected ----
        elif code == BO.FULLY_CONNECTED and bot == BOT.FullyConnectedOptions:
            opts = FullyConnectedOptions()
            opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            sh = sw = 1
            Nin = None
            Nout = Cout if Cout else None

            w_idx = op.Inputs(1) if op.InputsLength() > 1 else None
            w_shape = shape_by_index(sub, w_idx)
            if len(w_shape) == 2:
                Nout_w, Nin_w = w_shape[0], w_shape[1]
                Nin, Nout = Nin_w, Nout_w
            if Nin is None:
                Nin = 1
                if len(in_shape) >= 1:
                    for d in in_shape[1:]:
                        Nin *= max(1, d)
            if Nout is None:
                Nout = 1 if Cout is None else Cout

            kw_val = kh_val = 1
            Hin = Hin if Hin else 1
            Win = Win if Win else 1
            Hout = 1
            Wout = 1
            Cin = Nin
            Cout = Nout
            mac = compute_mac_fc(Nin, Nout)

        # ---- Pooling（不计 MAC，但保留 stride / kernel）----
        elif code in (BO.MAX_POOL_2D, BO.AVERAGE_POOL_2D, BO.L2_POOL_2D) and bot == BOT.Pool2DOptions:
            opts = Pool2DOptions()
            opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            sh, sw = opts.StrideH(), opts.StrideW()
            kh_val, kw_val = opts.FilterHeight(), opts.FilterWidth()
            mac = 0
            # 默认视为 noncompute；若未加 --include-noncompute 则跳过
            emit = args.include_noncompute

        # ---- Shortcut（ADD，且两输入与输出形状一致）----
        elif code == BO.ADD:
            if is_shortcut_add(op, sub, out_shape):
                op_readable = "Shortcut"
                sh = sw = 1
                kw_val = kh_val = 0  # 按你的要求：kernel size 用 0
                # MAC = Hout * Wout * Cout
                mac = as_int(Hout) * as_int(Wout) * as_int(Cout)
                emit = True
            else:
                # 其他 ADD（非残差）当作 noncompute
                mac = 0
                emit = args.include_noncompute

        else:
            # 其他算子：默认 MAC=0；仅在 --include-noncompute 时输出
            mac = 0
            emit = args.include_noncompute

        # 激活/逐元素等：除 Shortcut 外，默认不输出
        if (op_readable in ACTIVATION_OR_NONCOMPUTE) and (not args.include_noncompute):
            emit = False

        if not emit:
            continue

        row = {
            "index": running_idx,
            "op": op_readable,
            "Cin": as_int(Cin),
            "Hin": as_int(Hin),
            "Win": as_int(Win),
            "stride": stride_to_str(as_int(sh, 1), as_int(sw, 1)),
            "kw": as_int(kw_val),
            "kh": as_int(kh_val),
            "Cout": as_int(Cout),
            "Hout": as_int(Hout),
            "Wout": as_int(Wout),
            "MAC": int(mac)
        }
        rows.append(row)
        running_idx += 1

    headers = ["index", "op", "Cin", "Hin", "Win", "stride", "kw", "kh", "Cout", "Hout", "Wout", "MAC"]
    with open(args.output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"完成：共导出 {len(rows)} 行到 {args.output}")
    if not args.include_noncompute:
        print("（提示）默认跳过了激活/逐元素/池化等非乘加算子。若需要一并列出，请使用 --include-noncompute。")

if __name__ == "__main__":
    main()

