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

import unittest
import pandas as pd
import os
from unittest import mock

from src.model.monitor_type import MonitorType
from src.model.operation_type import OperationType
from src.monitoring_custom_metrics.main import (
    monitoring,
    determine_operation_to_run,
    determine_monitor_type,
)
from unittest.mock import patch

data = [["mike", 20], ["david", 37], ["sarah", 23], ["sam@gmail.com", 30]]
df = pd.DataFrame(data, columns=["Name", "Age"])


class TestMonitoring(unittest.TestCase):
    @mock.patch.dict(
        os.environ,
        {"baseline_statistics": "statistics.json", "baseline_constraints": "constraints.json"},
        clear=True,
    )
    def test_determine_operation_to_run_should_return_evaluate_constraints(self):
        operation_type = determine_operation_to_run()
        self.assertTrue(operation_type == OperationType.run_monitor)

    def test_determine_operation_to_run_should_return_suggest_baseline(self):
        operation_type = determine_operation_to_run()
        self.assertTrue(operation_type == OperationType.suggest_baseline)

    @mock.patch.dict(os.environ, {"baseline_statistics": "statistics.json"}, clear=True)
    def test_determine_operation_to_run_should_throw_because_constraints_missing(self):
        with self.assertRaises(RuntimeError) as context:
            determine_operation_to_run()

        self.assertEqual(
            "For evaluate constraints operation, both 'baseline_statistics' and 'baseline_constraints' environment "
            "variables must be provided.",
            str(context.exception),
        )

    @mock.patch.dict(os.environ, {"baseline_constraints": "constraints.json"}, clear=True)
    def test_determine_operation_to_run_should_throw_because_statistics_missing(self):
        with self.assertRaises(RuntimeError) as context:
            determine_operation_to_run()

        self.assertEqual(
            "For evaluate constraints operation, both 'baseline_statistics' and 'baseline_constraints' environment "
            "variables must be provided.",
            str(context.exception),
        )

    @mock.patch.dict(
        os.environ,
        {"dataset_source": "/opt/ml/processing/data/input", "analysis_type": "MODEL_QUALITY"},
        clear=True,
    )
    @patch("src.monitoring_custom_metrics.main.get_dataframe_from_csv")
    @patch("src.monitoring_custom_metrics.main.execute_for_model_quality")
    def test_monitoring_for_model_quality(
        self, mock_execute_for_model_quality, mock_get_dataframe_from_csv
    ):
        mock_get_dataframe_from_csv.return_value = df

        monitoring()
        mock_execute_for_model_quality.assert_called_once_with(OperationType.suggest_baseline, df)

    @mock.patch.dict(
        os.environ,
        {"analysis_type": "MODEL_QUALITY"},
        clear=True,
    )
    def test_determine_monitor_type_with_analysis_type_model_quality_provided(self):
        monitor_type = determine_monitor_type()
        self.assertEqual(MonitorType.MODEL_QUALITY.value, monitor_type.value)

    @mock.patch.dict(
        os.environ,
        {"analysis_type": "DATA_QUALITY"},
        clear=True,
    )
    def test_determine_monitor_type_with_analysis_type_data_quality_provided(self):
        monitor_type = determine_monitor_type()
        self.assertEqual(MonitorType.DATA_QUALITY.value, monitor_type.value)

    @mock.patch.dict(
        os.environ,
        {"ground_truth_attribute": "ground_truth_attribute"},
        clear=True,
    )
    def test_determine_monitor_type_without_analysis_type_with_ground_truth_provided(self):
        monitor_type = determine_monitor_type()
        self.assertEqual(MonitorType.MODEL_QUALITY.value, monitor_type.value)

    def test_determine_monitor_type_without_analysis_without_ground_truth_provided(self):
        monitor_type = determine_monitor_type()
        self.assertEqual(MonitorType.DATA_QUALITY.value, monitor_type.value)
