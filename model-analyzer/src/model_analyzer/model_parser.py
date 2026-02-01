"""Model parser for TensorFlow Lite and ONNX models."""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


def simplify_shape(shape: List[int]) -> str:
    """Convert shape to a simplified string representation."""
    if shape is None:
        return "Unknown"

    shape_str = "[" + ", ".join([str(dim) if dim is not None else "?" for dim in shape]) + "]"
    return shape_str


class OperatorInfo:
    """Container for operator information."""

    def __init__(self, name: str, op_type: str, inputs: List[str],
                 input_shapes: List[List[int]], output_shapes: List[List[int]],
                 attributes: Dict[str, Any] = None):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.attributes = attributes or {}

    def get_shape_signature(self) -> str:
        """Get a unique signature based on operator attributes."""
        # Generate signature based on operator type and key attributes
        if self.op_type in ['CONV_2D', 'Conv2D', 'Conv']:
            # For Conv2D: ci, co, kw, kh, stride_h, stride_w
            ci = self.attributes.get('ci', '?')
            co = self.attributes.get('co', '?')
            kw = self.attributes.get('kw', '?')
            kh = self.attributes.get('kh', '?')
            stride_h = self.attributes.get('stride_h', '?')
            stride_w = self.attributes.get('stride_w', '?')
            return f"ci={ci},co={co},kw={kw},kh={kh},sh={stride_h},sw={stride_w}"

        elif self.op_type in ['MAX_POOL_2D', 'MaxPool', 'AveragePool', 'AvgPool']:
            # For Pooling: type, kw, kh, stride_h, stride_w
            kw = self.attributes.get('kw', '?')
            kh = self.attributes.get('kh', '?')
            stride_h = self.attributes.get('stride_h', '?')
            stride_w = self.attributes.get('stride_w', '?')
            pool_type = 'max' if 'max' in self.op_type.lower() else 'avg'
            return f"type={pool_type},kw={kw},kh={kh},sh={stride_h},sw={stride_w}"

        elif self.op_type in ['FULLY_CONNECTED', 'Dense', 'Gemm']:
            # For Dense: input_features, output_features
            in_feat = self.attributes.get('input_features', '?')
            out_feat = self.attributes.get('output_features', '?')
            return f"in={in_feat},out={out_feat}"

        else:
            # For other ops: use input/output shape as fallback
            input_shapes_str = "|".join([simplify_shape(s) for s in self.input_shapes])
            output_shapes_str = "|".join([simplify_shape(s) for s in self.output_shapes])
            return f"{input_shapes_str}->{output_shapes_str}"


class BaseParser(ABC):
    """Base class for model parsers."""

    @abstractmethod
    def parse(self, model_path: str) -> List[OperatorInfo]:
        """Parse a model file and return list of operators."""
        pass


class TFLiteParser(BaseParser):
    """Parser for TensorFlow Lite models."""

    def parse(self, model_path: str) -> List[OperatorInfo]:
        """Parse a TensorFlow Lite model."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required to parse .tflite files. Install with: pip install tensorflow")

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get model details
        tensor_details = interpreter.get_tensor_details()
        op_details = interpreter._get_ops_details()

        # Create mapping from tensor index to shape
        tensor_shapes = {}
        for tensor in tensor_details:
            tensor_idx = tensor['index']
            tensor_shapes[tensor_idx] = tensor['shape'].tolist()

        # Extract operators
        operators = []
        for op in op_details:
            op_idx = op['index']
            op_type = op['op_name']

            # Get input/output tensor indices
            input_tensor_idxs = op['inputs']
            output_tensor_idxs = op['outputs']

            # Get input/output shapes
            input_shapes = [tensor_shapes.get(idx, []) for idx in input_tensor_idxs]
            output_shapes = [tensor_shapes.get(idx, []) for idx in output_tensor_idxs]

            # Extract operator attributes
            attributes = self._extract_attributes(op, input_shapes, output_shapes)

            # Create operator name
            op_name = f"{op_type}_{op_idx}"

            operator = OperatorInfo(
                name=op_name,
                op_type=op_type,
                inputs=[f"tensor_{idx}" for idx in input_tensor_idxs],
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                attributes=attributes
            )
            operators.append(operator)

        return operators

    def _extract_attributes(self, op: Dict, input_shapes: List[List[int]],
                           output_shapes: List[List[int]]) -> Dict[str, Any]:
        """Extract key attributes from TFLite operator."""
        attributes = {}
        op_type = op['op_name']

        # Get builtin options if available
        builtin_options = op.get('builtin_options', {})
        custom_options = op.get('custom_options', {})

        if op_type in ['CONV_2D', 'DEPTHWISE_CONV_2D']:
            # Conv2D attributes
            if len(input_shapes) >= 2:
                # Input shape: [batch, height, width, channels_in]
                # Weight shape: [out_channels, height, width, channels_in]
                input_shape = input_shapes[0]
                weight_shape = input_shapes[1] if len(input_shapes) > 1 else []

                attributes['ci'] = input_shape[3] if len(input_shape) > 3 else '?'
                attributes['co'] = weight_shape[0] if len(weight_shape) > 0 else '?'
                attributes['kw'] = weight_shape[2] if len(weight_shape) > 2 else '?'
                attributes['kh'] = weight_shape[1] if len(weight_shape) > 1 else '?'

            # Strides
            attributes['stride_h'] = builtin_options.get('stride_h', '?')
            attributes['stride_w'] = builtin_options.get('stride_w', '?')

        elif op_type in ['AVERAGE_POOL_2D', 'MAX_POOL_2D']:
            # Pooling attributes
            if len(input_shapes) > 0:
                input_shape = input_shapes[0]
                # Try to infer pool size from output shape
                if len(output_shapes) > 0 and len(input_shape) >= 2 and len(output_shapes[0]) >= 2:
                    out_h, out_w = output_shapes[0][1], output_shapes[0][2]
                    in_h, in_w = input_shape[1], input_shape[2]
                    # Approximate pool size (may not be exact without padding info)
                    attributes['kw'] = (in_w + out_w - 1) // out_w if out_w > 0 else '?'
                    attributes['kh'] = (in_h + out_h - 1) // out_h if out_h > 0 else '?'
                else:
                    attributes['kw'] = '?'
                    attributes['kh'] = '?'

            # Strides
            attributes['stride_h'] = builtin_options.get('stride_h', '?')
            attributes['stride_w'] = builtin_options.get('stride_w', '?')

        elif op_type in ['FULLY_CONNECTED', 'DENSE']:
            # Dense layer attributes
            if len(input_shapes) >= 1 and len(output_shapes) >= 1:
                input_shape = input_shapes[0]
                output_shape = output_shapes[0]
                attributes['input_features'] = input_shape[-1] if len(input_shape) > 0 else '?'
                attributes['output_features'] = output_shape[-1] if len(output_shape) > 0 else '?'

        return attributes


class ONNXParser(BaseParser):
    """Parser for ONNX models."""

    def parse(self, model_path: str) -> List[OperatorInfo]:
        """Parse an ONNX model."""
        try:
            import onnx
        except ImportError:
            raise ImportError("ONNX is required to parse .onnx files. Install with: pip install onnx")

        # Load the ONNX model
        model = onnx.load(model_path)

        # Get initializers and value info for shape information
        value_info = {vi.name: vi for vi in model.graph.value_info}
        inputs = {inp.name: inp for inp in model.graph.input}
        outputs = {out.name: out for out in model.graph.output}
        initializers = {init.name: init for init in model.graph.initializer}

        # Extract operators (nodes)
        operators = []
        for idx, node in enumerate(model.graph.node):
            op_type = node.op_type
            op_name = node.name if node.name else f"{op_type}_{idx}"

            # Get input shapes
            input_shapes = []
            for inp in node.input:
                shape = self._get_tensor_shape(inp, value_info, inputs, initializers)
                input_shapes.append(shape)

            # Get output shapes
            output_shapes = []
            for out in node.output:
                shape = self._get_tensor_shape(out, value_info, outputs, {})
                output_shapes.append(shape)

            # Extract operator attributes
            attributes = self._extract_attributes(node, input_shapes, output_shapes)

            operator = OperatorInfo(
                name=op_name,
                op_type=op_type,
                inputs=list(node.input),
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                attributes=attributes
            )
            operators.append(operator)

        return operators

    def _extract_attributes(self, node, input_shapes: List[List[int]],
                           output_shapes: List[List[int]]) -> Dict[str, Any]:
        """Extract key attributes from ONNX operator."""
        attributes = {}
        op_type = node.op_type

        if op_type in ['Conv']:
            # Conv2D attributes
            if len(input_shapes) >= 2:
                input_shape = input_shapes[0]
                weight_shape = input_shapes[1] if len(input_shapes) > 1 else []

                attributes['ci'] = input_shape[1] if len(input_shape) > 1 else '?'  # NCHW format
                attributes['co'] = weight_shape[0] if len(weight_shape) > 0 else '?'
                attributes['kw'] = weight_shape[3] if len(weight_shape) > 3 else '?'
                attributes['kh'] = weight_shape[2] if len(weight_shape) > 2 else '?'

            # Extract strides from attributes
            for attr in node.attribute:
                if attr.name == 'strides':
                    strides = list(attr.ints)
                    attributes['stride_h'] = strides[0] if len(strides) > 0 else '?'
                    attributes['stride_w'] = strides[1] if len(strides) > 1 else '?'
                elif attr.name == 'kernel_shape':
                    kernel = list(attr.ints)
                    attributes['kh'] = kernel[0] if len(kernel) > 0 else '?'
                    attributes['kw'] = kernel[1] if len(kernel) > 1 else '?'

        elif op_type in ['MaxPool', 'AveragePool']:
            # Pooling attributes
            if len(node.attribute) > 0:
                for attr in node.attribute:
                    if attr.name == 'kernel_shape':
                        kernel = list(attr.ints)
                        attributes['kw'] = kernel[1] if len(kernel) > 1 else '?'
                        attributes['kh'] = kernel[0] if len(kernel) > 0 else '?'
                    elif attr.name == 'strides':
                        strides = list(attr.ints)
                        attributes['stride_h'] = strides[0] if len(strides) > 0 else '?'
                        attributes['stride_w'] = strides[1] if len(strides) > 1 else '?'

        elif op_type in ['Gemm', 'MatMul']:
            # Dense/MatMul attributes
            if len(input_shapes) >= 1 and len(output_shapes) >= 1:
                input_shape = input_shapes[0]
                output_shape = output_shapes[0]
                attributes['input_features'] = input_shape[-1] if len(input_shape) > 0 else '?'
                attributes['output_features'] = output_shape[-1] if len(output_shape) > 0 else '?'

        elif op_type in ['Relu', 'Sigmoid', 'Tanh', 'LeakyRelu']:
            # Activation functions - no special attributes needed
            pass

        return attributes

    def _get_tensor_shape(self, tensor_name: str, value_info: Dict,
                         inputs_outputs: Dict, initializers: Dict) -> List[int]:
        """Get the shape of a tensor."""
        # Check value_info first
        if tensor_name in value_info:
            tensor_type = value_info[tensor_name].type
            if tensor_type.HasField('tensor_type'):
                shape = []
                for dim in tensor_type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value > 0 else None)
                return shape

        # Check inputs/outputs
        if tensor_name in inputs_outputs:
            tensor_type = inputs_outputs[tensor_name].type
            if tensor_type.HasField('tensor_type'):
                shape = []
                for dim in tensor_type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value > 0 else None)
                return shape

        # Check initializers
        if tensor_name in initializers:
            return list(initializers[tensor_name].dims)

        return []


class ModelParser:
    """Main model parser that dispatches to appropriate parser."""

    def __init__(self):
        self.parsers = {
            '.tflite': TFLiteParser(),
            '.onnx': ONNXParser(),
        }

    def parse(self, model_path: str) -> List[OperatorInfo]:
        """Parse a model file based on extension."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ext = os.path.splitext(model_path)[1].lower()
        if ext not in self.parsers:
            raise ValueError(f"Unsupported model format: {ext}. Supported formats: {list(self.parsers.keys())}")

        return self.parsers[ext].parse(model_path)

    def supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.parsers.keys())
