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

import os
import unittest
from unittest import mock
from unittest.mock import patch, mock_open

from src.monitoring_custom_metrics.output_generator import (
    write_results_to_output_folder,
    write_exit_message,
)

NO_VIOLATIONS = {"violations": []}


VIOLATIONS = {
    "violations": [
        {
            "constraint_check_type": "LessThanThreshold",
            "description": "Metric ep with -0.1 was LessThanThreshold -0.05",
            "metric_name": "ep",
        }
    ]
}


class TestOutputGenerator(unittest.TestCase):
    @mock.patch.dict(
        os.environ,
        {"output_path": "/output"},
        clear=True,
    )
    @patch("src.monitoring_custom_metrics.output_generator.write_exit_message")
    def test_write_results_to_output_folder(self, mock_write_output_message):
        result = ["a", "b", "c"]
        with patch("builtins.open", mock_open(read_data="data")) as mock_file:
            write_results_to_output_folder(result)

        mock_file.assert_called()
        mock_write_output_message.assert_called()

    def test_write_exit_message_with_violations(self):
        result = ["a", "b", VIOLATIONS]
        output_path = "/output"

        with patch("builtins.open", mock_open(read_data="data")) as mock_file:
            write_exit_message(result, output_path)
        mock_file.assert_called_with("/opt/ml/output/message", "w", encoding="utf-8")
        mock_file.return_value.write.assert_called_once_with(
            "CompletedWithViolations: Job completed successfully with 1 violations."
        )

    def test_write_exit_message_with_no_violations(self):
        result = ["a", "b", NO_VIOLATIONS]
        output_path = "/output"

        with patch("builtins.open", mock_open(read_data="data")) as mock_file:
            write_exit_message(result, output_path)
        mock_file.assert_called_with("/opt/ml/output/message", "w", encoding="utf-8")
        mock_file.return_value.write.assert_called_once_with(
            "Completed: Job completed successfully with no violations."
        )
