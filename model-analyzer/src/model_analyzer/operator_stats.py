"""Operator statistics collection and analysis."""

from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
from .model_parser import OperatorInfo


class OperatorCategory:
    """Represents an operator with shape information."""

    def __init__(self, op_type: str, shape_signature: str):
        self.op_type = op_type
        self.shape_signature = shape_signature
        self.count = 0

    @property
    def full_name(self) -> str:
        """Get the full name including shape information."""
        return f"{self.op_type}_{self.shape_signature}"

    @property
    def base_name(self) -> str:
        """Get the base operator name without shape."""
        return self.op_type


class OperatorStats:
    """Collect and analyze operator statistics from models."""

    def __init__(self):
        self.model_stats = {}  # model_name -> List[OperatorInfo]
        self.global_stats = defaultdict(Counter)  # op_type -> Counter of shape_signatures
        self.model_categories = {}  # model_name -> Dict of categories

    def add_model(self, model_name: str, operators: List[OperatorInfo]):
        """Add operator information from a model."""
        self.model_stats[model_name] = operators

        # Analyze categories for this model
        categories = defaultdict(lambda: defaultdict(int))

        for op in operators:
            shape_signature = op.get_shape_signature()

            # Update global stats
            self.global_stats[op.op_type][shape_signature] += 1

            # Update model-specific categories
            categories[op.op_type][shape_signature] += 1

        self.model_categories[model_name] = categories

    def get_model_summary(self, model_name: str) -> List[Dict[str, Any]]:
        """Get summary statistics for a single model."""
        if model_name not in self.model_categories:
            return []

        summary = []
        categories = self.model_categories[model_name]

        for op_type, shape_dict in categories.items():
            for shape_signature, count in shape_dict.items():
                # Get example operator for shape details
                example_op = None
                for op in self.model_stats[model_name]:
                    if op.op_type == op_type and op.get_shape_signature() == shape_signature:
                        example_op = op
                        break

                # Format attributes for display
                attributes_str = self._format_attributes(example_op.attributes) if example_op else 'N/A'

                summary.append({
                    'Operator Type': op_type,
                    'Attributes': attributes_str,
                    'Count': count,
                    'Total Count (Type)': sum(shape_dict.values()),
                    'Category': shape_signature
                })

        return summary

    def get_global_summary(self) -> List[Dict[str, Any]]:
        """Get summary statistics across all models."""
        summary = []

        for op_type, shape_dict in self.global_stats.items():
            for shape_signature, count in shape_dict.items():
                # Count how many models use this operator+shape combination
                models_using = sum(
                    1 for model_name, categories in self.model_categories.items()
                    if op_type in categories and shape_signature in categories[op_type]
                )

                # Get total count for this op type across all models
                total_type_count = sum(
                    sum(categories[op_type].values())
                    for categories in self.model_categories.values()
                    if op_type in categories
                )

                # Find example operator from any model
                example_op = None
                for operators in self.model_stats.values():
                    for op in operators:
                        if op.op_type == op_type and op.get_shape_signature() == shape_signature:
                            example_op = op
                            break
                    if example_op:
                        break

                # Format attributes for display
                attributes_str = self._format_attributes(example_op.attributes) if example_op else 'N/A'

                summary.append({
                    'Operator Type': op_type,
                    'Attributes': attributes_str,
                    'Count': count,
                    'Total Count (Type)': total_type_count,
                    'Models Using': models_using,
                    'Category': shape_signature
                })

        # Sort by count descending
        summary.sort(key=lambda x: x['Count'], reverse=True)
        return summary

    def get_operator_breakdown(self) -> Dict[str, int]:
        """Get breakdown of operator types across all models."""
        breakdown = {}
        for op_type, shape_dict in self.global_stats.items():
            breakdown[op_type] = sum(shape_dict.values())
        return breakdown

    def get_model_list(self) -> List[str]:
        """Get list of model names."""
        return list(self.model_stats.keys())

    def get_total_models(self) -> int:
        """Get total number of models analyzed."""
        return len(self.model_stats)

    def get_total_operators(self) -> int:
        """Get total number of operators across all models."""
        return sum(
            sum(shape_dict.values())
            for shape_dict in self.global_stats.values()
        )

    def get_unique_operator_types(self) -> int:
        """Get number of unique operator types."""
        return len(self.global_stats)

    def get_unique_shape_combinations(self) -> int:
        """Get number of unique operator+shape combinations."""
        return sum(len(shape_dict) for shape_dict in self.global_stats.values())

    def _format_shape(self, shape: List[int]) -> str:
        """Format shape list to string."""
        if not shape:
            return "Unknown"
        return '[' + ', '.join([str(dim) if dim is not None else '?' for dim in shape]) + ']'

    def _format_attributes(self, attributes: Dict[str, Any]) -> str:
        """Format operator attributes for display."""
        if not attributes:
            return "N/A"

        # Format based on operator type
        attr_strs = []

        # For Conv2D: show ci, co, kw, kh, stride
        if 'ci' in attributes and 'co' in attributes:
            attr_strs.append(f"ci={attributes.get('ci', '?')}")
            attr_strs.append(f"co={attributes.get('co', '?')}")
            attr_strs.append(f"kw={attributes.get('kw', '?')}")
            attr_strs.append(f"kh={attributes.get('kh', '?')}")
            if 'stride_h' in attributes:
                attr_strs.append(f"sh={attributes.get('stride_h', '?')}")
                attr_strs.append(f"sw={attributes.get('stride_w', '?')}")

        # For Pooling
        elif 'kw' in attributes and 'kh' in attributes and 'stride_h' in attributes:
            pool_type = attributes.get('type', '?') if 'type' in attributes else None
            if pool_type:
                attr_strs.append(f"type={pool_type}")
            attr_strs.append(f"kw={attributes.get('kw', '?')}")
            attr_strs.append(f"kh={attributes.get('kh', '?')}")
            attr_strs.append(f"sh={attributes.get('stride_h', '?')}")
            attr_strs.append(f"sw={attributes.get('stride_w', '?')}")

        # For Dense
        elif 'input_features' in attributes and 'output_features' in attributes:
            attr_strs.append(f"in={attributes.get('input_features', '?')}")
            attr_strs.append(f"out={attributes.get('output_features', '?')}")

        # For other ops with attributes
        else:
            # Show all attributes
            for key, value in sorted(attributes.items()):
                attr_strs.append(f"{key}={value}")

        return " | ".join(attr_strs) if attr_strs else "N/A"