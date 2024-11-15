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

from src.model.model_quality_attributes import ModelQualityAttributes
from src.monitoring_custom_metrics.model_quality.binary_classification.gini import (
    instance,
)

predictions = [0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1]
actuals = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DF = pd.DataFrame()
DF["probability_attribute"] = predictions
DF["ground_truth_attribute"] = actuals

CONSTRAINT_NO_VIOLATION = {
    "threshold": 0.6111,
    "comparison_operator": "LessThanThreshold",
    "additional_properties": None,
}
CONSTRAINT_WITH_VIOLATION = {
    "threshold": 0.6611,
    "comparison_operator": "LessThanThreshold",
    "additional_properties": None,
}

CONFIG = {"metric_name": "gini"}

CONFIG_OVERRIDE = {"metric_name": "gini", "threshold_override": 0.05}

EXPECTED_STATISTIC = {"value": 0.6111, "standard_deviation": 0}
EXPECTED_VIOLATION = {
    "constraint_check_type": "LessThanThreshold",
    "description": "Metric gini with 0.6111 was LessThanThreshold 0.6611",
    "metric_name": "gini",
}

GROUND_TRUTH_ATTRIBUTE = "ground_truth_attribute"
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
        statistic = instance.calculate_statistics(DF, CONFIG, MODEL_QUALITY_ATTRIBUTES)
        self.assertEqual(EXPECTED_STATISTIC, statistic)

    def test_evaluate_constraints(self):
        no_violation = instance.evaluate_constraints(
            EXPECTED_STATISTIC, DF, CONFIG, CONSTRAINT_NO_VIOLATION, MODEL_QUALITY_ATTRIBUTES
        )
        violation = instance.evaluate_constraints(
            EXPECTED_STATISTIC, DF, CONFIG, CONSTRAINT_WITH_VIOLATION, MODEL_QUALITY_ATTRIBUTES
        )
        self.assertIsNone(no_violation)
        self.assertEqual(EXPECTED_VIOLATION, violation)

    def test_suggest_constraints(self):
        suggested_baseline = instance.suggest_constraints(
            EXPECTED_STATISTIC, DF, CONFIG, MODEL_QUALITY_ATTRIBUTES
        )
        suggested_baseline_w_override = instance.suggest_constraints(
            EXPECTED_STATISTIC, DF, CONFIG_OVERRIDE, MODEL_QUALITY_ATTRIBUTES
        )
        self.assertEqual(CONSTRAINT_NO_VIOLATION, suggested_baseline)
        self.assertEqual(CONSTRAINT_WITH_VIOLATION, suggested_baseline_w_override)
