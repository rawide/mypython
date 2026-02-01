#!/usr/bin/env python3
"""Test script for new attribute-based operator classification."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_analyzer.model_parser import OperatorInfo


def test_conv2d_attributes():
    """Test Conv2D attribute extraction and signature generation."""
    print("Testing Conv2D operator...")

    # Create Conv2D operator with specific attributes
    op = OperatorInfo(
        name="conv1",
        op_type="Conv2D",
        inputs=[],
        input_shapes=[[1, 224, 224, 3], [3, 3, 3, 64]],
        output_shapes=[[1, 112, 112, 64]],
        attributes={
            'ci': 3,
            'co': 64,
            'kw': 3,
            'kh': 3,
            'stride_h': 2,
            'stride_w': 2
        }
    )

    signature = op.get_shape_signature()
    print(f"  Conv2D signature: {signature}")
    assert "ci=3" in signature
    assert "co=64" in signature
    assert "kw=3" in signature
    assert "kh=3" in signature
    assert "sh=2" in signature
    assert "sw=2" in signature
    print("  ✓ Conv2D test passed!\n")


def test_pooling_attributes():
    """Test pooling attribute extraction and signature generation."""
    print("Testing MaxPooling operator...")

    # MaxPooling (2x2)
    op1 = OperatorInfo(
        name="pool1",
        op_type="MaxPool",
        inputs=[],
        input_shapes=[[1, 112, 112, 64]],
        output_shapes=[[1, 56, 56, 64]],
        attributes={
            'type': 'max',
            'kw': 2,
            'kh': 2,
            'stride_h': 2,
            'stride_w': 2
        }
    )

    signature1 = op1.get_shape_signature()
    print(f"  MaxPool(2x2) signature: {signature1}")
    assert "type=max" in signature1
    assert "kw=2" in signature1
    assert "kh=2" in signature1

    # MaxPooling (4x4)
    op2 = OperatorInfo(
        name="pool2",
        op_type="MaxPool",
        inputs=[],
        input_shapes=[[1, 56, 56, 128]],
        output_shapes=[[1, 14, 14, 128]],
        attributes={
            'type': 'max',
            'kw': 4,
            'kh': 4,
            'stride_h': 4,
            'stride_w': 4
        }
    )

    signature2 = op2.get_shape_signature()
    print(f"  MaxPool(4x4) signature: {signature2}")
    assert "type=max" in signature2
    assert "kw=4" in signature2
    assert "kh=4" in signature2
    assert "sh=4" in signature2

    # Verify different signatures
    assert signature1 != signature2
    print("  ✓ MaxPooling test passed!\n")

    # AvgPooling (2x2)
    print("Testing AvgPooling operator...")
    op3 = OperatorInfo(
        name="pool3",
        op_type="AveragePool",
        inputs=[],
        input_shapes=[[1, 112, 112, 64]],
        output_shapes=[[1, 56, 56, 64]],
        attributes={
            'type': 'avg',
            'kw': 2,
            'kh': 2,
            'stride_h': 2,
            'stride_w': 2
        }
    )

    signature3 = op3.get_shape_signature()
    print(f"  AvgPool(2x2) signature: {signature3}")
    assert "type=avg" in signature3
    assert signature1 != signature3  # max vs avg should be different
    print("  ✓ AvgPooling test passed!\n")


def test_dense_attributes():
    """Test Dense layer attribute extraction."""
    print("Testing Dense operator...")

    op = OperatorInfo(
        name="dense1",
        op_type="Dense",
        inputs=[],
        input_shapes=[[1, 128]],
        output_shapes=[[1, 64]],
        attributes={
            'input_features': 128,
            'output_features': 64
        }
    )

    signature = op.get_shape_signature()
    print(f"  Dense signature: {signature}")
    assert "in=128" in signature
    assert "out=64" in signature
    print("  ✓ Dense test passed!\n")


def test_different_conv_same_attributes():
    """Test that Conv2D with same attributes have same signature."""
    print("Testing same attributes -> same signature...")

    op1 = OperatorInfo(
        name="conv1",
        op_type="Conv2D",
        inputs=[],
        input_shapes=[[1, 224, 224, 3], [3, 3, 3, 64]],
        output_shapes=[[1, 112, 112, 64]],
        attributes={
            'ci': 3, 'co': 64, 'kw': 3, 'kh': 3, 'stride_h': 2, 'stride_w': 2
        }
    )

    op2 = OperatorInfo(
        name="conv2",
        op_type="Conv2D",
        inputs=[],
        input_shapes=[[1, 112, 112, 64], [3, 3, 64, 128]],
        output_shapes=[[1, 56, 56, 128]],
        attributes={
            'ci': 64, 'co': 128, 'kw': 3, 'kh': 3, 'stride_h': 2, 'stride_w': 2
        }
    )

    op3 = OperatorInfo(
        name="conv3",
        op_type="Conv2D",
        inputs=[],
        input_shapes=[[1, 56, 56, 128], [3, 3, 128, 256]],
        output_shapes=[[1, 28, 28, 256]],
        attributes={
            'ci': 128, 'co': 256, 'kw': 3, 'kh': 3, 'stride_h': 2, 'stride_w': 2
        }
    )

    sig1 = op1.get_shape_signature()
    sig2 = op2.get_shape_signature()
    sig3 = op3.get_shape_signature()

    print(f"  Conv1: {sig1}")
    print(f"  Conv2: {sig2}")
    print(f"  Conv3: {sig3}")

    # All have same kw=3, kh=3, stride=2, so signatures should be similar pattern
    assert sig1 != sig2  # Different ci/co
    assert sig2 != sig3  # Different ci/co

    # Check that attributes are correctly represented
    assert "ci=3,co=64" in sig1
    assert "ci=64,co=128" in sig2
    assert "ci=128,co=256" in sig3
    assert "kw=3,kh=3" in sig1
    assert "kw=3,kh=3" in sig2
    assert "kw=3,kh=3" in sig3
    assert "sh=2,sw=2" in sig1
    assert "sh=2,sw=2" in sig2
    assert "sh=2,sw=2" in sig3

    print("  ✓ Different Conv2D with same attribute structure test passed!\n")


def test_attribute_formatting():
    """Test the _format_attributes method."""
    from model_analyzer.operator_stats import OperatorStats

    print("Testing attribute formatting...")

    stats = OperatorStats()

    # Mock operator with Conv2D attributes
    class MockOp:
        def __init__(self, attributes):
            self.attributes = attributes

    op = MockOp({
        'ci': 3, 'co': 64, 'kw': 3, 'kh': 3, 'stride_h': 2, 'stride_w': 2
    })

    formatted = stats._format_attributes(op.attributes)
    print(f"  Formatted attributes: {formatted}")

    assert "ci=3" in formatted
    assert "co=64" in formatted
    assert "kw=3" in formatted
    assert "kh=3" in formatted
    assert "sh=2" in formatted
    assert "sw=2" in formatted

    print("  ✓ Attribute formatting test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Attribute-Based Classification")
    print("=" * 60 + "\n")

    try:
        test_conv2d_attributes()
        test_pooling_attributes()
        test_dense_attributes()
        test_different_conv_same_attributes()
        test_attribute_formatting()

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
