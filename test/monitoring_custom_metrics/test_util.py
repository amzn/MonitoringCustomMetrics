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
from unittest.mock import patch, mock_open, call

import pandas as pd

from src.monitoring_custom_metrics.util import (
    retrieve_json_file,
    get_dataframe_from_csv,
    retrieve_first_json_file_in_path,
    retrieve_json_file_in_path,
    validate_environment_variable,
)


class TestUtil(unittest.TestCase):
    @patch("json.load")
    def test_retrieve_json_file(self, mock_json_load):
        file_path = "/abc/file.json"

        with patch("builtins.open", mock_open(read_data="data")) as mock_file:
            retrieve_json_file(file_path)
        mock_file.assert_called_with(file_path)
        mock_json_load.assert_called_once()

    @patch("json.load")
    @patch("src.monitoring_custom_metrics.util.get_first_file_from_directory")
    def test_retrieve_first_json_file_in_path(
        self, mock_get_first_file_from_directory, mock_json_load
    ):
        file_path = "/abc"
        mock_get_first_file_from_directory.return_value = "file.json"

        with patch("builtins.open", mock_open(read_data="data")):
            retrieve_first_json_file_in_path(file_path)

        mock_json_load.assert_called_once()

    @mock.patch.dict(os.environ, {"dataset_source": "/opt/ml/processing/input/data"}, clear=True)
    @patch("os.listdir")
    @patch("pandas.read_csv")
    @patch("pandas.concat")
    def test_get_dataframe_from_csv(self, concat_mock, read_csv_mock, list_dir_mock):
        read_csv_mock.return_value = pd.DataFrame({"data": [11, 4, 2020]})
        concat_mock.return_value = pd.DataFrame({"data": [11, 4, 2020]})
        list_dir_mock.return_value = ["someRandomFileName.csv", "anotherFileName.csv"]
        results = get_dataframe_from_csv()
        read_csv_mock.assert_has_calls(
            [
                call("/opt/ml/processing/input/data/someRandomFileName.csv"),
                call("/opt/ml/processing/input/data/anotherFileName.csv"),
            ]
        )

        self.assertEqual(1, len(results.columns))
        self.assertTrue(3, len(results.columns[0]))

    @patch("json.load")
    @patch("src.monitoring_custom_metrics.util.retrieve_json_file")
    def test_retrieve_json_file_in_path_for_file_path(
        self, mock_retrieve_json_file, mock_json_load
    ):
        file_path = "/abc/file.json"

        with patch("builtins.open", mock_open(read_data="data")):
            retrieve_json_file_in_path(file_path)

        mock_retrieve_json_file.assert_called_once_with(file_path)

    @patch("json.load")
    @patch("src.monitoring_custom_metrics.util.retrieve_first_json_file_in_path")
    def test_retrieve_json_file_in_path_for_directory_path(
        self, mock_retrieve_first_json_file_in_path, mock_json_load
    ):
        file_path = "/abc"

        with patch("builtins.open", mock_open(read_data="data")):
            retrieve_json_file_in_path(file_path)

        mock_retrieve_first_json_file_in_path.assert_called_once_with(file_path)

    def test_validate_environment_variable_missing(self):
        env_var_name = "my_variable"
        with self.assertRaises(ValueError) as context:
            validate_environment_variable(env_var_name)
        self.assertEqual(f"Environment variable {env_var_name} is not set.", str(context.exception))
