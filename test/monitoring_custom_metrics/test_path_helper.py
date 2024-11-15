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

import sys
import unittest
from unittest import mock
from unittest.mock import Mock

from src.monitoring_custom_metrics.path_helper import retrieve_modules, import_class_paths


class TestPathHelper(unittest.TestCase):
    @mock.patch("glob.glob", return_value=["a.py", "b.py"])
    def test_retrieve_modules(self, mock_glob):
        dummy_path = "dummy"
        sys.modules["a"] = Mock()
        sys.modules["b"] = Mock()
        modules = retrieve_modules(dummy_path)
        self.assertTrue(len(modules) == 2)

    def test_import_class_paths(self):
        import_class_paths()

        data_quality_numerical_path = "monitoring_custom_metrics/data_quality/numerical"
        data_quality_string_path = "monitoring_custom_metrics/data_quality/string"
        model_quality_binary_classification_path = (
            "monitoring_custom_metrics/model_quality/binary_classification"
        )
        model_quality_multiclass_classification_path = (
            "monitoring_custom_metrics/model_quality/multiclass_classification"
        )
        model_quality_regression_path = "monitoring_custom_metrics/model_quality/regression"

        contains_data_quality_numerical_path = False
        contains_data_quality_string_path = False
        contains_model_quality_binary_classification_path = False
        contains_model_quality_multiclass_classification_path = False
        contains_model_quality_regression_path = False

        for path in sys.path:
            if data_quality_numerical_path in path:
                contains_data_quality_numerical_path = True
            if data_quality_string_path in path:
                contains_data_quality_string_path = True
            if model_quality_binary_classification_path in path:
                contains_model_quality_binary_classification_path = True
            if model_quality_multiclass_classification_path in path:
                contains_model_quality_multiclass_classification_path = True
            if model_quality_regression_path in path:
                contains_model_quality_regression_path = True

        if not contains_data_quality_numerical_path:
            self.fail(data_quality_numerical_path + " not in sys.path")
        if not contains_data_quality_string_path:
            self.fail(data_quality_string_path + " not in sys.path")
        if not contains_model_quality_binary_classification_path:
            self.fail(model_quality_binary_classification_path + " not in sys.path")
        if not contains_model_quality_multiclass_classification_path:
            self.fail(model_quality_multiclass_classification_path + " not in sys.path")
        if not contains_model_quality_regression_path:
            self.fail(model_quality_regression_path + " not in sys.path")
