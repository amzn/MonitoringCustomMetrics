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
from src.monitoring_custom_metrics.model_quality.binary_classification.score_diff import (
    instance,
)

predictions = [0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1]
actuals = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
DF = pd.DataFrame()
DF["probability_attribute"] = predictions
DF["ground_truth_attribute"] = actuals

# Test scenario 1: comparison_type = "absolute" and two_sided = True
CONSTRAINT_NO_VIOLATION = {
    "threshold": 0.072,
    "comparison_operator": "GreaterThanThreshold",
    "additional_properties": None,
}
CONSTRAINT_WITH_VIOLATION = {
    "threshold": 0.06,
    "comparison_operator": "GreaterThanThreshold",
    "additional_properties": None,
}

CONFIG = {"metric_name": "score_diff", "comparison_type": "absolute", "two_sided": True}

CONFIG_OVERRIDE = {
    "metric_name": "score_diff",
    "comparison_type": "absolute",
    "two_sided": True,
    "threshold_override": -0.012,
}

EXPECTED_STATISTIC = {"value": 0.072, "standard_deviation": 0}
EXPECTED_VIOLATION = {
    "constraint_check_type": "GreaterThanThreshold",
    "description": "Metric score_diff with 0.072 was GreaterThanThreshold 0.06",
    "metric_name": "score_diff",
}

# Test scenario 2: comparison_type = "absolute" and two_sided = False
CONSTRAINT_WITH_VIOLATION_ONE_SIDED = {
    "threshold": 0.09,
    "comparison_operator": "LessThanThreshold",
    "additional_properties": None,
}

CONFIG_ONE_SIDED = {
    "metric_name": "score_diff",
    "comparison_type": "absolute",
    "two_sided": False,
    "comparison_operator": "LessThanThreshold",
    "threshold_override": 0.018,
}

EXPECTED_VIOLATION_ONE_SIDED = {
    "constraint_check_type": "LessThanThreshold",
    "description": "Metric score_diff with 0.072 was LessThanThreshold 0.09",
    "metric_name": "score_diff",
}

# Test scenario 3: comparison_type = "relative" and two_sided = True
CONSTRAINT_WITH_VIOLATION_RELATIVE = {
    "threshold": 0.16,
    "comparison_operator": "GreaterThanThreshold",
    "additional_properties": None,
}

CONFIG_RELATIVE = {
    "metric_name": "score_diff",
    "comparison_type": "relative",
    "two_sided": True,
    "threshold_override": -0.02,
}

EXPECTED_STATISTIC_RELATIVE = {"value": 0.18, "standard_deviation": 0}
EXPECTED_VIOLATION_RELATIVE = {
    "constraint_check_type": "GreaterThanThreshold",
    "description": "Metric score_diff with 0.18 was GreaterThanThreshold 0.16",
    "metric_name": "score_diff",
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
        statistic_relative = instance.calculate_statistics(
            DF, CONFIG_RELATIVE, MODEL_QUALITY_ATTRIBUTES
        )
        self.assertEqual(EXPECTED_STATISTIC, statistic)
        self.assertEqual(EXPECTED_STATISTIC_RELATIVE, statistic_relative)

    def test_evaluate_constraints(self):
        no_violation = instance.evaluate_constraints(
            EXPECTED_STATISTIC, DF, CONFIG, CONSTRAINT_NO_VIOLATION, MODEL_QUALITY_ATTRIBUTES
        )
        violation = instance.evaluate_constraints(
            EXPECTED_STATISTIC, DF, CONFIG, CONSTRAINT_WITH_VIOLATION, MODEL_QUALITY_ATTRIBUTES
        )
        violation_one_sided = instance.evaluate_constraints(
            EXPECTED_STATISTIC,
            DF,
            CONFIG_ONE_SIDED,
            CONSTRAINT_WITH_VIOLATION_ONE_SIDED,
            MODEL_QUALITY_ATTRIBUTES,
        )
        violation_relative = instance.evaluate_constraints(
            EXPECTED_STATISTIC_RELATIVE,
            DF,
            CONFIG_RELATIVE,
            CONSTRAINT_WITH_VIOLATION_RELATIVE,
            MODEL_QUALITY_ATTRIBUTES,
        )
        self.assertIsNone(no_violation)
        self.assertEqual(EXPECTED_VIOLATION, violation)
        self.assertEqual(EXPECTED_VIOLATION_ONE_SIDED, violation_one_sided)
        self.assertEqual(EXPECTED_VIOLATION_RELATIVE, violation_relative)

    def test_suggest_constraints(self):
        suggested_baseline = instance.suggest_constraints(
            EXPECTED_STATISTIC, DF, CONFIG, MODEL_QUALITY_ATTRIBUTES
        )
        suggested_baseline_w_override = instance.suggest_constraints(
            EXPECTED_STATISTIC, DF, CONFIG_OVERRIDE, MODEL_QUALITY_ATTRIBUTES
        )
        suggested_baseline_one_sided = instance.suggest_constraints(
            EXPECTED_STATISTIC, DF, CONFIG_ONE_SIDED, MODEL_QUALITY_ATTRIBUTES
        )
        suggested_baseline_relative = instance.suggest_constraints(
            EXPECTED_STATISTIC_RELATIVE, DF, CONFIG_RELATIVE, MODEL_QUALITY_ATTRIBUTES
        )
        self.assertEqual(CONSTRAINT_NO_VIOLATION, suggested_baseline)
        self.assertEqual(CONSTRAINT_WITH_VIOLATION, suggested_baseline_w_override)
        self.assertEqual(CONSTRAINT_WITH_VIOLATION_ONE_SIDED, suggested_baseline_one_sided)
        self.assertEqual(CONSTRAINT_WITH_VIOLATION_RELATIVE, suggested_baseline_relative)
