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

import pandas as pd
import pytest  # noqa
import unittest

from src.monitoring_custom_metrics.model_quality.binary_classification.custom_metric import (
    instance,
)
from src.model.model_quality_attributes import ModelQualityAttributes

DATA = [["mike", 20], ["david", 37], ["sarah", 23], ["sam@gmail.com", 30]]
CONSTRAINT_NO_VIOLATION = {"threshold": 50, "additional_properties": None}
CONSTRAINT_WITH_VIOLATION = {
    "threshold": 25.5,
    "comparison_operator": "GreaterThanThreshold",
    "additional_properties": None,
}
EMPTY_CONFIG = {}
CONFIG = {"threshold_overide": 10}
# Create the pandas DataFrame
DF = pd.DataFrame(DATA, columns=["Name", "Age"])

EXPECTED_STATISTIC = {"value": 27.5, "standard_deviation": 0}
EXPECTED_VIOLATION = {
    "constraint_check_type": "CustomMetric",
    "description": "Metric CustomMetric 27.5 is above threshold 25.5",
    "metric_name": "custom_metric",
}
EXPECTED_OVERRIDEN_VIOLATION = {
    "constraint_check_type": "CustomMetric",
    "description": "Metric CustomMetric 27.5 is above threshold 10",
    "metric_name": "custom_metric",
}


GROUND_TRUTH_ATTRIBUTE = "Age"
PROBABILITY_ATTRIBUTE = "probability_attribute"
PROBABILITY_THRESHOLD_ATTRIBUTE = "probability_threshold_attribute"
INFERENCE_ATTRIBUTE = "inference_attribute"
MODEL_QUALITY_ATTRIBUTES = ModelQualityAttributes(
    GROUND_TRUTH_ATTRIBUTE,
    PROBABILITY_ATTRIBUTE,
    PROBABILITY_THRESHOLD_ATTRIBUTE,
    INFERENCE_ATTRIBUTE,
)


class TestCustomMetric(unittest.TestCase):
    def test_calculate_statistics(self):
        statistic = instance.calculate_statistics(DF, None, MODEL_QUALITY_ATTRIBUTES)
        self.assertEqual(EXPECTED_STATISTIC, statistic)

    def test_evaluate_constraints(self):
        no_violation = instance.evaluate_constraints(
            EXPECTED_STATISTIC, DF, EMPTY_CONFIG, CONSTRAINT_NO_VIOLATION, MODEL_QUALITY_ATTRIBUTES
        )
        violation = instance.evaluate_constraints(
            EXPECTED_STATISTIC,
            DF,
            EMPTY_CONFIG,
            CONSTRAINT_WITH_VIOLATION,
            MODEL_QUALITY_ATTRIBUTES,
        )
        config_overide_violation = instance.evaluate_constraints(
            EXPECTED_STATISTIC, DF, CONFIG, CONSTRAINT_WITH_VIOLATION, MODEL_QUALITY_ATTRIBUTES
        )
        self.assertIsNone(no_violation)
        self.assertEqual(EXPECTED_VIOLATION, violation)
        self.assertEqual(EXPECTED_OVERRIDEN_VIOLATION, config_overide_violation)

    def test_suggest_constraints(self):
        suggested_baseline = instance.suggest_constraints(
            EXPECTED_STATISTIC, DF, EMPTY_CONFIG, MODEL_QUALITY_ATTRIBUTES
        )
        self.assertEqual(CONSTRAINT_WITH_VIOLATION, suggested_baseline)
