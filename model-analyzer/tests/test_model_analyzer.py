"""Basic tests for Model Analyzer."""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from model_analyzer import ModelParser, OperatorStats, StatsExporter


class TestModelAnalyzer(unittest.TestCase):
    """Test cases for Model Analyzer functionality."""

    def test_tflite_parser_exists(self):
        """Test that TFLite parser is available."""
        parser = ModelParser()
        extensions = parser.supported_extensions()
        self.assertIn('.tflite', extensions)

    def test_onnx_parser_exists(self):
        """Test that ONNX parser is available."""
        parser = ModelParser()
        extensions = parser.supported_extensions()
        self.assertIn('.onnx', extensions)

    def test_get_operator_signature(self):
        """Test operator shape signature generation."""
        from model_analyzer.model_parser import OperatorInfo

        op = OperatorInfo(
            name="test_op",
            op_type="Conv2D",
            inputs=["input1", "input2"],
            input_shapes=[[1, 224, 224, 3], [3, 3, 3, 64]],
            output_shapes=[[1, 112, 112, 64]]
        )

        signature = op.get_shape_signature()
        self.assertIn("[1, 224, 224, 3]", signature)
        self.assertIn("[1, 112, 112, 64]", signature)

    def test_operator_stats_collection(self):
        """Test operator statistics collection."""
        from model_analyzer.model_parser import OperatorInfo

        # Create test operators
        ops = [
            OperatorInfo("op1", "Conv2D", [], [[1, 224, 224, 3]], [[1, 112, 112, 64]]),
            OperatorInfo("op2", "Conv2D", [], [[1, 112, 112, 64]], [[1, 56, 56, 128]]),
            OperatorInfo("op3", "ReLU", [], [[1, 112, 112, 64]], [[1, 112, 112, 64]]),
        ]

        stats = OperatorStats()
        stats.add_model("test_model", ops)

        # Check statistics
        self.assertEqual(stats.get_total_models(), 1)
        self.assertEqual(stats.get_total_operators(), 3)
        self.assertEqual(stats.get_unique_operator_types(), 2)  # Conv2D, ReLU

        # Check breakdown
        breakdown = stats.get_operator_breakdown()
        self.assertEqual(breakdown['Conv2D'], 2)
        self.assertEqual(breakdown['ReLU'], 1)

    def test_different_shapes_same_op_type(self):
        """Test that operators with same type but different shapes are tracked separately."""
        from model_analyzer.model_parser import OperatorInfo

        # Create operators with same type but different shapes
        ops = [
            OperatorInfo("op1", "Conv2D", [], [[1, 224, 224, 3]], [[1, 112, 112, 64]]),
            OperatorInfo("op2", "Conv2D", [], [[1, 112, 112, 64]], [[1, 56, 56, 128]]),
            OperatorInfo("op3", "Conv2D", [], [[1, 224, 224, 3]], [[1, 112, 112, 64]]),  # Same as op1
        ]

        stats = OperatorStats()
        stats.add_model("test_model", ops)

        # Should have 3 operators but only 1 unique type
        self.assertEqual(stats.get_total_operators(), 3)
        self.assertEqual(stats.get_unique_operator_types(), 1)

        # But 2 unique shape combinations
        self.assertEqual(stats.get_unique_shape_combinations(), 2)

        # Check global summary
        summary = stats.get_global_summary()

        # Find Conv2D entries
        conv_entries = [s for s in summary if s['Operator Type'] == 'Conv2D']
        self.assertEqual(len(conv_entries), 2)  # Two different shapes

        # Check counts in each entry
        for entry in conv_entries:
            if '[1, 224, 224, 3]' in entry['Example Input Shapes']:
                self.assertEqual(entry['Count'], 2)  # Appears twice
            else:
                self.assertEqual(entry['Count'], 1)  # Appears once

    def test_multiple_models(self):
        """Test statistics across multiple models."""
        from model_analyzer.model_parser import OperatorInfo

        stats = OperatorStats()

        # Model 1
        ops1 = [
            OperatorInfo("op1", "Conv2D", [], [[1, 224, 224, 3]], [[1, 112, 112, 64]]),
        ]
        stats.add_model("model1", ops1)

        # Model 2
        ops2 = [
            OperatorInfo("op1", "Conv2D", [], [[1, 224, 224, 3]], [[1, 112, 112, 64]]),
            OperatorInfo("op2", "ReLU", [], [[1, 112, 112, 64]], [[1, 112, 112, 64]]),
        ]
        stats.add_model("model2", ops2)

        # Check totals
        self.assertEqual(stats.get_total_models(), 2)
        self.assertEqual(stats.get_total_operators(), 3)

        # Check global stats
        summary = stats.get_global_summary()
        conv_entry = [s for s in summary if s['Operator Type'] == 'Conv2D'][0]

        # Conv2D appears in both models
        self.assertEqual(conv_entry['Models Using'], 2)
        self.assertEqual(conv_entry['Count'], 2)  # Total count across models


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running basic tests...")

    # Test model parser
    parser = ModelParser()
    print(f"Supported extensions: {parser.supported_extensions()}")

    # Test operator stats
    stats = OperatorStats()
    print(f"Initial stats - Models: {stats.get_total_models()}, Operators: {stats.get_total_operators()}")

    print("Basic tests passed!")


if __name__ == "__main__":
    # Run unittest
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

    # Run basic tests
    print("\n" + "=" * 60)
    run_basic_tests()
