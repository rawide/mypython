"""Export operator statistics to Excel format."""

import os
from typing import List, Dict, Any
import pandas as pd


class StatsExporter:
    """Export operator statistics to Excel format with multiple sheets."""

    def __init__(self, stats):
        self.stats = stats

    def export_to_excel(self, output_path: str):
        """Export all statistics to a multi-sheet Excel file."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary across all models
            print("Generating global summary sheet...")
            self._write_global_summary(writer)

            # Sheets for each model
            model_list = self.stats.get_model_list()
            for idx, model_name in enumerate(model_list, 1):
                print(f"Generating sheet for model {idx}/{len(model_list)}: {model_name}")
                self._write_model_sheet(writer, model_name)

            # Last sheet: Detailed breakdown by operator type
            print("Generating operator breakdown sheet...")
            self._write_operator_breakdown(writer)

            # Sheet with metadata
            print("Generating metadata sheet...")
            self._write_metadata_sheet(writer)

        print(f"\nExported statistics to: {output_path}")

    def _write_global_summary(self, writer: pd.ExcelWriter):
        """Write global summary sheet."""
        global_summary = self.stats.get_global_summary()

        if not global_summary:
            df = pd.DataFrame([{"Message": "No operators found in any model"}])
            df.to_excel(writer, sheet_name="Global Summary", index=False)
            return

        df = pd.DataFrame(global_summary)

        # Reorder columns for better readability
        columns = [
            'Operator Type',
            'Attributes',
            'Count',
            'Total Count (Type)',
            'Models Using',
            'Category'
        ]
        df = df[columns]

        # Write to Excel
        df.to_excel(writer, sheet_name="Global Summary", index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets["Global Summary"]
        for idx, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)

    def _write_model_sheet(self, writer: pd.ExcelWriter, model_name: str):
        """Write a sheet for a specific model."""
        model_summary = self.stats.get_model_summary(model_name)

        if not model_summary:
            df = pd.DataFrame([{"Message": f"No operators found in {model_name}"}])
            sheet_name = self._sanitize_sheet_name(model_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            return

        df = pd.DataFrame(model_summary)

        # Reorder columns for better readability
        columns = [
            'Operator Type',
            'Attributes',
            'Count',
            'Total Count (Type)',
            'Category'
        ]
        df = df[columns]

        # Use sanitized model name as sheet name
        sheet_name = self._sanitize_sheet_name(model_name)

        # Write to Excel
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)

    def _write_operator_breakdown(self, writer: pd.ExcelWriter):
        """Write operator breakdown sheet."""
        breakdown = self.stats.get_operator_breakdown()

        if not breakdown:
            df = pd.DataFrame([{"Message": "No operators found"}])
            df.to_excel(writer, sheet_name="Operator Breakdown", index=False)
            return

        # Convert to DataFrame
        df = pd.DataFrame(list(breakdown.items()),
                         columns=['Operator Type', 'Total Count'])

        # Sort by count descending
        df = df.sort_values('Total Count', ascending=False).reset_index(drop=True)

        # Write to Excel
        df.to_excel(writer, sheet_name="Operator Breakdown", index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets["Operator Breakdown"]
        for idx, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_length

    def _write_metadata_sheet(self, writer: pd.ExcelWriter):
        """Write metadata sheet."""
        metadata = {
            'Metric': [
                'Total Models Analyzed',
                'Total Operators',
                'Unique Operator Types',
                'Unique Operator+Shape Combinations',
                'Supported Formats'
            ],
            'Value': [
                self.stats.get_total_models(),
                self.stats.get_total_operators(),
                self.stats.get_unique_operator_types(),
                self.stats.get_unique_shape_combinations(),
                '.tflite, .onnx'
            ]
        }

        df = pd.DataFrame(metadata)
        df.to_excel(writer, sheet_name="Metadata", index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets["Metadata"]
        for idx, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_length

    def _sanitize_sheet_name(self, name: str) -> str:
        """Sanitize sheet name to Excel constraints."""
        # Excel sheet names must be <= 31 characters
        # Cannot contain: : \ / ? * [ ]

        # Remove invalid characters
        invalid_chars = [':', '\\', '/', '?', '*', '[', ']']
        for char in invalid_chars:
            name = name.replace(char, '_')

        # Truncate to 31 characters
        if len(name) > 31:
            # Take last 28 chars + "..."
            name = "..." + name[-28:]

        # Ensure name is not empty
        if not name or name.isspace():
            name = "Model"

        return name

    def export_to_csv(self, output_dir: str):
        """Export statistics to separate CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Global summary CSV
        global_summary = self.stats.get_global_summary()
        if global_summary:
            df = pd.DataFrame(global_summary)
            df.to_csv(os.path.join(output_dir, "global_summary.csv"),
                     index=False, encoding='utf-8')

        # Per-model CSVs
        for model_name in self.stats.get_model_list():
            model_summary = self.stats.get_model_summary(model_name)
            if model_summary:
                df = pd.DataFrame(model_summary)
                safe_name = self._sanitize_filename(model_name)
                df.to_csv(os.path.join(output_dir, f"{safe_name}.csv"),
                         index=False, encoding='utf-8')

        # Operator breakdown CSV
        breakdown = self.stats.get_operator_breakdown()
        if breakdown:
            df = pd.DataFrame(list(breakdown.items()),
                            columns=['Operator Type', 'Total Count'])
            df = df.sort_values('Total Count', ascending=False).reset_index(drop=True)
            df.to_csv(os.path.join(output_dir, "operator_breakdown.csv"),
                     index=False, encoding='utf-8')

        print(f"\nExported CSV files to: {output_dir}")

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename for saving."""
        # Remove/replace invalid filename characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in invalid_chars:
            name = name.replace(char, '_')

        # Remove spaces
        name = name.replace(' ', '_')

        return name
