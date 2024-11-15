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

from src.monitoring_custom_metrics.data_quality.numerical.sum import instance

DATA = [["mike", 20], ["david", 37], ["sarah", 23], ["sam@gmail.com", 30]]
CONSTRAINT_NO_VIOLATION_NO_BASELINE_PROVIDED = {
    "lower_bound": 99.0,
    "upper_bound": 121.0,
    "additional_properties": None,
}
CONSTRAINT_NO_VIOLATION = {
    "lower_bound": 99.0,
    "upper_bound": 121.0,
    "additional_properties": {"baseline_lower_bound": 10.0, "baseline_upper_bound": 1000.0},
}
CONSTRAINT_WITH_VIOLATION = {"lower_bound": 21, "upper_bound": 40, "additional_properties": None}
BASELINE_CONSTRAINTS = {"lower_bound": 10.0, "upper_bound": 1000.0, "additional_properties": None}

# Create the pandas DataFrame
DF = pd.DataFrame(DATA, columns=["Name", "Age"])

EXPECTED_SUM = 110
EXPECTED_VIOLATION = {
    "feature_name": "Age",
    "constraint_check_type": "InBetweenBound",
    "description": "Metric sum 110 for Age value outside of range 21 - 40",
    "metric_name": "sum",
}


class TestSum(unittest.TestCase):
    def test_calculate_statistics(self):
        sum_statistic = instance.calculate_statistics(DF["Age"])
        self.assertEqual(EXPECTED_SUM, sum_statistic)

    def test_evaluate_constraints(self):
        statistics = EXPECTED_SUM
        no_violation = instance.evaluate_constraints(statistics, DF["Age"], CONSTRAINT_NO_VIOLATION)
        violation = instance.evaluate_constraints(statistics, DF["Age"], CONSTRAINT_WITH_VIOLATION)
        self.assertIsNone(no_violation)
        self.assertEqual(EXPECTED_VIOLATION, violation)

    def test_suggest_constraints(self):
        statistics = EXPECTED_SUM
        suggested_baseline = instance.suggest_constraints(
            statistics, DF["Age"], BASELINE_CONSTRAINTS
        )
        self.assertEqual(CONSTRAINT_NO_VIOLATION, suggested_baseline)

    def test_suggest_constraints_with_no_baseline(self):
        statistics = EXPECTED_SUM
        suggested_baseline = instance.suggest_constraints(statistics, DF["Age"], None)
        self.assertEqual(CONSTRAINT_NO_VIOLATION_NO_BASELINE_PROVIDED, suggested_baseline)
