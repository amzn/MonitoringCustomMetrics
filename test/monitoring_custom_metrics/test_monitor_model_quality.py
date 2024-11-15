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
from unittest.mock import Mock, patch

import pandas as pd

from src.model.model_quality_attributes import ModelQualityAttributes
from src.model.problem_type import ProblemType
from src.monitoring_custom_metrics.monitor_model_quality import (
    execute_operation_for_model_quality,
    execute_for_model_quality,
    translate_problem_type,
    validate_environment_variables,
    get_model_quality_attributes,
)
from src.model.operation_type import OperationType

data = [["mike", 20], ["david", 37], ["sarah", 23], ["sam@gmail.com", 30]]
df = pd.DataFrame(data, columns=["Name", "Age"])
column = df["Age"]
constraint_label = "binary_classification_constraints"
constraint = {constraint_label: {"my_custom_metric": {"threshold": 50}}}
no_constraint = {constraint_label: {}}
config_that_includes_my_custom_metric = {"my_custom_metric": {"threshold_overide": 10}}
config_without_my_custom_metric = {}
config_json_path = "/opt/config.json"
constraints_json_path = "/opt/constraints.json"
statistics_result = 27.5
constraints_result = json.dumps({"threshold": 25.5})
expected_statistics = {"my_custom_metric": 27.5}
expected_constraints = {"my_custom_metric": json.dumps({"threshold": 25.5})}

expected_constraint_violations = json.dumps(
    {
        "constraint_check_type": "CustomMetric",
        "description": "Metric CustomMetric 27.5 is above threshold 25.5",
        "metric_name": "custom_metric",
    }
)
expected_statistics_file_content = {
    "version": 0.0,
    "dataset": {"item_count": 4},
    "binary_classification_metrics": {"my_custom_metric": 27.5},
}
expected_constraints_file_content = {
    "version": 0.0,
    "binary_classification_constraints": {"my_custom_metric": {"threshold": 25.5}},
}
expected_violations_file_content = {
    "violations": [
        {
            "constraint_check_type": "CustomMetric",
            "description": "Metric CustomMetric 27.5 is above threshold 25.5",
            "metric_name": "custom_metric",
        }
    ]
}
ground_truth_attribute = "ground_truth_attribute"
probability_attribute = "probability_attribute"
probability_threshold_attribute = "probability_threshold_attribute"
inference_attribute = "inference_attribute"
model_quality_attributes = ModelQualityAttributes(
    ground_truth_attribute,
    probability_attribute,
    probability_threshold_attribute,
    inference_attribute,
)


class TestMonitorModelQuality(unittest.TestCase):
    @patch("src.monitoring_custom_metrics.monitor_model_quality.get_model_quality_class_path")
    @mock.patch("glob.glob", return_value=["a.py"])
    def test_evaluate_constraints_for_model_quality(
        self, mock_glob, mock_get_model_quality_class_path
    ):
        sys.modules["a"] = Mock()
        instance = Mock()
        sys.modules["a"].instance = instance

        instance.calculate_statistics.return_value = statistics_result
        instance.suggest_constraints.return_value = constraints_result
        instance.evaluate_constraints.return_value = expected_constraint_violations
        sys.modules["a"].__name__ = "my_custom_metric"

        output = execute_operation_for_model_quality(
            OperationType.run_monitor,
            ProblemType.binary_classification,
            model_quality_attributes,
            df,
            config_that_includes_my_custom_metric,
            constraint_label,
            constraint,
        )
        self.assertTrue(len(output) == 3)

        self.assertEqual(expected_statistics, output[0])
        self.assertEqual(expected_constraints, output[1])
        self.assertEqual([expected_constraint_violations], output[2])

        instance.evaluate_constraints.assert_called_once_with(
            statistics_result,
            df,
            config_that_includes_my_custom_metric["my_custom_metric"],
            constraint[constraint_label]["my_custom_metric"],
            model_quality_attributes,
        )

    @patch("src.monitoring_custom_metrics.monitor_model_quality.get_model_quality_class_path")
    @mock.patch("glob.glob", return_value=["a.py"])
    def test_evaluate_constraints_for_model_quality_with_no_constraints_provided(
        self, mock_glob, mock_get_model_quality_class_path
    ):
        empty_dict = {}
        sys.modules["a"] = Mock()
        instance = Mock()
        sys.modules["a"].instance = instance

        instance.calculate_statistics.return_value = statistics_result
        instance.suggest_constraints.return_value = constraints_result
        instance.evaluate_constraints.return_value = expected_constraint_violations
        sys.modules["a"].__name__ = "my_custom_metric"

        output = execute_operation_for_model_quality(
            OperationType.run_monitor,
            ProblemType.binary_classification,
            model_quality_attributes,
            df,
            config_that_includes_my_custom_metric,
            constraint_label,
            no_constraint,
        )
        self.assertTrue(len(output) == 3)

        self.assertEqual(expected_statistics, output[0])
        self.assertEqual(expected_constraints, output[1])
        self.assertEqual([expected_constraint_violations], output[2])

        instance.evaluate_constraints.assert_called_once_with(
            statistics_result,
            df,
            config_that_includes_my_custom_metric["my_custom_metric"],
            empty_dict,
            model_quality_attributes,
        )

    @patch("src.monitoring_custom_metrics.monitor_model_quality.get_model_quality_class_path")
    @mock.patch("glob.glob", return_value=["a.py"])
    def test_evaluate_constraints_for_model_quality_without_module_config(
        self, mock_glob, mock_get_model_quality_class_path
    ):
        sys.modules["a"] = Mock()
        instance = Mock()
        sys.modules["a"].instance = instance

        instance.calculate_statistics.return_value = statistics_result
        instance.suggest_constraints.return_value = constraints_result
        instance.evaluate_constraints.return_value = expected_constraint_violations
        sys.modules["a"].__name__ = "my_custom_metric"

        output = execute_operation_for_model_quality(
            OperationType.run_monitor,
            ProblemType.binary_classification,
            model_quality_attributes,
            df,
            config_without_my_custom_metric,
            constraint_label,
            constraint,
        )

        self.assertTrue(len(output) == 3)

        self.assertEqual({}, output[0])
        self.assertEqual({}, output[1])
        self.assertEqual([], output[2])

        instance.evaluate_constraints.assert_not_called()

    @patch("src.monitoring_custom_metrics.monitor_model_quality.get_model_quality_class_path")
    @mock.patch("glob.glob", return_value=["a.py"])
    def test_suggest_baseline_for_model_quality(self, mock_glob, mock_get_model_quality_class_path):
        sys.modules["a"] = Mock()
        instance = Mock()
        sys.modules["a"].instance = instance
        instance.calculate_statistics.return_value = statistics_result
        instance.suggest_constraints.return_value = constraints_result
        sys.modules["a"].__name__ = "my_custom_metric"

        output = execute_operation_for_model_quality(
            OperationType.suggest_baseline,
            ProblemType.binary_classification,
            model_quality_attributes,
            df,
            config_that_includes_my_custom_metric,
            constraint_label,
        )

        mock_get_model_quality_class_path.assert_called_once()

        self.assertTrue(len(output) == 3)
        self.assertEqual(expected_statistics, output[0])
        self.assertEqual(expected_constraints, output[1])
        self.assertEqual([], output[2])

        instance.suggest_constraints.assert_called_once_with(
            statistics_result,
            df,
            config_that_includes_my_custom_metric["my_custom_metric"],
            model_quality_attributes,
        )

    @mock.patch.dict(
        os.environ,
        {"config_path": config_json_path, "problem_type": "BinaryClassification"},
        clear=True,
    )
    @patch("src.monitoring_custom_metrics.monitor_model_quality.validate_environment_variables")
    @patch("src.monitoring_custom_metrics.monitor_model_quality.get_model_quality_attributes")
    @patch("src.monitoring_custom_metrics.monitor_model_quality.write_results_to_output_folder")
    @patch(
        "src.monitoring_custom_metrics.monitor_model_quality.execute_operation_for_model_quality"
    )
    @patch(
        "src.monitoring_custom_metrics.monitor_model_quality.retrieve_json_file_in_path",
        return_value="data",
    )
    def test_execute_for_model_quality_suggest_baseline(
        self,
        mock_retrieve_json_file_in_path,
        mock_execute_operation_for_model_quality,
        mock_write_results_to_output_folder,
        mock_get_model_quality_attributes,
        mock_validate_environment_variables,
    ):
        operation_type = OperationType.suggest_baseline

        mock_execute_operation_for_model_quality.return_value = [
            {"my_custom_metric": 27.5},
            {"my_custom_metric": {"threshold": 25.5}},
            [],
        ]

        mock_get_model_quality_attributes.return_value = model_quality_attributes

        output = execute_for_model_quality(operation_type, df)

        mock_validate_environment_variables.assert_called_once()
        mock_retrieve_json_file_in_path.assert_called_with(config_json_path)
        mock_execute_operation_for_model_quality.assert_called_with(
            operation_type,
            ProblemType.binary_classification,
            model_quality_attributes,
            df,
            "data",
            constraint_label,
            None,
        )
        mock_write_results_to_output_folder.assert_called_with(output)

        self.assertTrue(len(output) == 3)

        self.assertEqual(expected_statistics_file_content, output[0])
        self.assertEqual(expected_constraints_file_content, output[1])
        self.assertEqual(None, output[2])

    @mock.patch.dict(
        os.environ,
        {
            "config_path": config_json_path,
            "baseline_constraints": constraints_json_path,
            "problem_type": "BinaryClassification",
        },
        clear=True,
    )
    @patch("src.monitoring_custom_metrics.monitor_model_quality.validate_environment_variables")
    @patch("src.monitoring_custom_metrics.monitor_model_quality.get_model_quality_attributes")
    @patch("src.monitoring_custom_metrics.monitor_model_quality.write_results_to_output_folder")
    @patch(
        "src.monitoring_custom_metrics.monitor_model_quality.execute_operation_for_model_quality"
    )
    @patch(
        "src.monitoring_custom_metrics.monitor_model_quality.retrieve_json_file_in_path",
        return_value="data",
    )
    def test_execute_for_model_quality_evaluate_constraints(
        self,
        mock_retrieve_json_file_in_path,
        mock_execute_operation_for_model_quality,
        mock_write_results_to_output_folder,
        mock_get_model_quality_attributes,
        mock_validate_environment_variables,
    ):
        operation_type = OperationType.run_monitor

        mock_execute_operation_for_model_quality.return_value = [
            {"my_custom_metric": 27.5},
            {"my_custom_metric": {"threshold": 25.5}},
            [
                '{"constraint_check_type": "CustomMetric", "description": "Metric CustomMetric 27.5 is above threshold 25.5", "metric_name": "custom_metric"}'
            ],
        ]

        mock_get_model_quality_attributes.return_value = model_quality_attributes

        output = execute_for_model_quality(operation_type, df)

        mock_retrieve_json_file_in_path.assert_has_calls(
            [mock.call(config_json_path), mock.call(constraints_json_path)]
        )
        mock_execute_operation_for_model_quality.assert_called_with(
            operation_type,
            ProblemType.binary_classification,
            model_quality_attributes,
            df,
            "data",
            constraint_label,
            "data",
        )
        mock_write_results_to_output_folder.assert_called_with(output)
        mock_validate_environment_variables.assert_called_once()

        self.assertTrue(len(output) == 3)

        self.assertEqual(expected_statistics_file_content, output[0])
        self.assertEqual(expected_constraints_file_content, output[1])
        self.assertTrue(expected_violations_file_content, output[2])

    @mock.patch.dict(
        os.environ,
        {
            "problem_type": "something",
        },
        clear=True,
    )
    @patch("src.monitoring_custom_metrics.monitor_model_quality.validate_environment_variables")
    def test_execute_for_model_quality_with_invalid_problem_type(
        self, mock_validate_environment_variables
    ):
        operation_type = OperationType.run_monitor

        with self.assertRaises(ValueError) as context:
            execute_for_model_quality(operation_type, df)
        self.assertEqual("Invalid problem type: something", str(context.exception))

        mock_validate_environment_variables.assert_called_once()

    def test_translate_problem_type(self):
        self.assertEqual(
            ProblemType.binary_classification, translate_problem_type("BinaryClassification")
        )
        self.assertEqual(
            ProblemType.multiclass_classification,
            translate_problem_type("MulticlassClassification"),
        )
        self.assertEqual(ProblemType.regression, translate_problem_type("Regression"))

        with self.assertRaises(ValueError) as context:
            translate_problem_type("something")
        self.assertEqual("Invalid problem type: something", str(context.exception))

    @mock.patch.dict(
        os.environ,
        {
            "config_path": config_json_path,
            "baseline_constraints": constraints_json_path,
            "problem_type": "BinaryClassification",
            "probability_attribute": "probability_attribute",
            "ground_truth_attribute": "ground_truth_attribute",
        },
        clear=True,
    )
    def test_validate_environment_variables_with_probability_attribute_missing_probability_threshold(
        self,
    ):
        with self.assertRaises(ValueError) as context:
            validate_environment_variables(OperationType.suggest_baseline)
        self.assertEqual(
            "Environment variable probability_threshold_attribute is not set.",
            str(context.exception),
        )

    @mock.patch.dict(
        os.environ,
        {
            "config_path": config_json_path,
            "baseline_constraints": constraints_json_path,
            "problem_type": "BinaryClassification",
            "ground_truth_attribute": "ground_truth_attribute",
        },
        clear=True,
    )
    def test_validate_environment_variables_without_probability_attribute_missing_inference_attribute(
        self,
    ):
        with self.assertRaises(ValueError) as context:
            validate_environment_variables(OperationType.suggest_baseline)
        self.assertEqual(
            "Environment variable inference_attribute is not set.", str(context.exception)
        )

    @mock.patch.dict(
        os.environ,
        {
            "config_path": config_json_path,
            "problem_type": "BinaryClassification",
            "ground_truth_attribute": "ground_truth_attribute",
            "inference_attribute": "inference_attribute",
        },
        clear=True,
    )
    def test_validate_environment_variables_without_probability_attribute_missing_baseline_constraints(
        self,
    ):
        with self.assertRaises(ValueError) as context:
            validate_environment_variables(OperationType.run_monitor)
        self.assertEqual(
            "Environment variable baseline_constraints is not set.", str(context.exception)
        )

    @mock.patch.dict(
        os.environ,
        {
            "ground_truth_attribute": "ground_truth_attribute",
            "inference_attribute": "inference_attribute",
        },
        clear=True,
    )
    def test_get_model_quality_attributes_without_probability_attribute(self):
        attributes = get_model_quality_attributes()
        self.assertEqual("ground_truth_attribute", attributes.ground_truth_attribute)
        self.assertEqual("inference_attribute", attributes.inference_attribute)
        self.assertIsNone(attributes.probability_attribute)
        self.assertIsNone(attributes.probability_threshold_attribute)

    @mock.patch.dict(
        os.environ,
        {
            "ground_truth_attribute": "ground_truth_attribute",
            "probability_attribute": "probability_attribute",
            "probability_threshold_attribute": "probability_threshold_attribute",
        },
        clear=True,
    )
    def test_get_model_quality_attributes_with_probability_attribute(self):
        attributes = get_model_quality_attributes()
        self.assertEqual("ground_truth_attribute", attributes.ground_truth_attribute)
        self.assertEqual("probability_attribute", attributes.probability_attribute)
        self.assertEqual(
            "probability_threshold_attribute", attributes.probability_threshold_attribute
        )
        self.assertIsNone(attributes.inference_attribute)
