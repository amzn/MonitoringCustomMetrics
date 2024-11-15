# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
import unittest
from unittest import mock
from unittest.mock import Mock, patch, mock_open

import pandas as pd

from src.monitoring_custom_metrics.monitor_data_quality import (
    translate_data_type,
    execute_operation_for_data_quality,
    execute_for_data_quality,
    validate_environment_variables,
)
from src.model.data_type import DataType
from src.model.operation_type import OperationType

data = [["mike", 20], ["david", 37], ["sarah", 23], ["sam@gmail.com", 30]]
df = pd.DataFrame(data, columns=["Name", "Age"])
column = df["Age"]
statistic = "statistics"
constraint = {"lower_bound": 21, "upper_bound": 40}
constraint_output = {
    "version": 0.0,
    "features": [
        {
            "name": "Age",
            "inferred_type": "Integral",
            "num_constraints": {"sum": {"lower_bound": 21, "upper_bound": 40}},
        }
    ],
}
constraint_violation = {
    "constraint_check_type": "InBetweenBound",
    "description": "Metric sum 406 for CUSTOMER_ID value outside of range 5 - 10",
    "metric_name": "sum",
}

config = json.dumps({"custom_metric": {"threshold_overide": 10}})
config_json_path = "/opt/config.json"
constraints_json_path = "/opt/constraints.json"
expected_statistics = [
    {
        "name": "Age",
        "inferred_type": "Integral",
        "numerical_statistics": {
            "common": {"num_missing": 0, "num_present": 4},
            "sum": "statistics",
        },
    }
]
expected_statistics_file_content = {
    "version": 0.0,
    "dataset": {"item_count": 4},
}
expected_constraint_violations = [constraint_violation]
expected_constraints = [{"lower_bound": 21, "upper_bound": 40}]


class TestMonitorDataQuality(unittest.TestCase):
    def test_translate_data_type(self):
        self.assertEqual(DataType.Integral, translate_data_type("int64"))
        self.assertEqual(DataType.Fractional, translate_data_type("float64"))
        self.assertEqual(DataType.String, translate_data_type("bool"))
        self.assertEqual(DataType.String, translate_data_type("object"))

    @patch("src.monitoring_custom_metrics.monitor_data_quality.get_data_quality_class_path")
    @mock.patch("glob.glob", return_value=["sum.py"])
    def test_run_monitor_for_data_quality(self, mock_glob, mock_get_data_quality_class_path):
        sys.modules["sum"] = Mock()
        sys.modules["sum"].__name__ = "sum"

        instance = Mock()
        sys.modules["sum"].instance = instance
        instance.calculate_statistics.return_value = statistic
        instance.evaluate_constraints.return_value = constraint_violation
        instance.suggest_constraints.return_value = constraint

        output = execute_operation_for_data_quality(
            OperationType.run_monitor, DataType.Integral, column, constraint_output
        )

        self.assertTrue(len(output) == 3)
        assert expected_statistics == output[0]
        assert constraint_output["features"] == output[1]
        assert expected_constraint_violations == output[2]
        instance.evaluate_constraints.assert_called_once_with(
            statistic, column, expected_constraints[0]
        )
        instance.suggest_constraints.assert_called_once_with(
            statistic, column, expected_constraints[0]
        )

    @patch("src.monitoring_custom_metrics.monitor_data_quality.get_data_quality_class_path")
    @mock.patch("glob.glob", return_value=["sum.py"])
    def test_suggest_baseline_for_data_quality(self, mock_glob, mock_get_data_quality_class_path):
        sys.modules["sum"] = Mock()
        sys.modules["sum"].__name__ = "sum"

        instance = Mock()
        sys.modules["sum"].instance = instance
        instance.calculate_statistics.return_value = statistic
        instance.suggest_constraints.return_value = constraint

        output = execute_operation_for_data_quality(
            OperationType.suggest_baseline, DataType.Integral, column
        )

        self.assertTrue(len(output) == 3)
        assert expected_statistics == output[0]
        assert constraint_output["features"] == output[1]
        assert [] == output[2]
        instance.suggest_constraints.assert_called_once_with(statistic, column, None)

    @mock.patch.dict(
        os.environ,
        {"output_path": "/output"},
        clear=True,
    )
    @patch("src.monitoring_custom_metrics.monitor_data_quality.write_results_to_output_folder")
    @patch("src.monitoring_custom_metrics.monitor_data_quality.execute_operation_for_data_quality")
    def test_execute_for_data_quality_suggest_baseline(
        self, mock_execute_operation_for_data_quality, mock_write_results_to_output_folder
    ):
        operation_type = OperationType.suggest_baseline
        output = []
        with patch("builtins.open", mock_open(read_data="data")):
            output = execute_for_data_quality(operation_type, df)

        mock_call_1 = mock.call(operation_type, DataType.String, df["Name"], None)
        mock_call_2 = mock.call(operation_type, DataType.Integral, df["Age"], None)

        mock_execute_operation_for_data_quality.assert_has_calls(
            [mock_call_1, mock_call_2], any_order=True
        )
        mock_write_results_to_output_folder.assert_called_once()

        self.assertTrue(len(output) == 3)
        self.assertEqual(4, output[0]["dataset"]["item_count"])

    @mock.patch.dict(
        os.environ,
        {"baseline_constraints": constraints_json_path, "output_path": "/output"},
        clear=True,
    )
    @patch("src.monitoring_custom_metrics.monitor_data_quality.write_results_to_output_folder")
    @patch("src.monitoring_custom_metrics.monitor_data_quality.execute_operation_for_data_quality")
    @patch(
        "src.monitoring_custom_metrics.monitor_data_quality.retrieve_json_file_in_path",
        return_value="data",
    )
    def test_execute_for_data_quality_evaluate_constraints(
        self,
        mock_retrieve_json_file,
        mock_execute_operation_for_data_quality,
        write_results_to_output_folder,
    ):
        operation_type = OperationType.run_monitor

        with patch("builtins.open", mock_open(read_data="data")):
            execute_for_data_quality(operation_type, df)

        mock_retrieve_json_file.assert_called_with(constraints_json_path)

        write_results_to_output_folder.assert_called_once()

        mock_call_1 = mock.call(operation_type, DataType.String, df["Name"], "data")
        mock_call_2 = mock.call(operation_type, DataType.Integral, df["Age"], "data")

        mock_execute_operation_for_data_quality.assert_has_calls(
            [mock_call_1, mock_call_2], any_order=True
        )

    def test_validate_environment_variables_with_baseline_constraints_missing(
        self,
    ):
        with self.assertRaises(ValueError) as context:
            validate_environment_variables(OperationType.run_monitor)
        self.assertEqual(
            "Environment variable baseline_constraints is not set.",
            str(context.exception),
        )
