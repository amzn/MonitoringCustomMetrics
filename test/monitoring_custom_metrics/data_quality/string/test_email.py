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

from src.monitoring_custom_metrics.data_quality.string.email import instance

DATA = [["mike", 20], ["david", 37], ["sarah", 23], ["sam@gmail.com", 30]]
CONSTRAINT_NO_VIOLATION = {
    "lower_bound": None,
    "upper_bound": None,
    "additional_properties": {"allowed": True},
}
CONSTRAINT_WITH_VIOLATION = {
    "lower_bound": None,
    "upper_bound": None,
    "additional_properties": {"allowed": False},
}

# Create the pandas DataFrame
DF = pd.DataFrame(DATA, columns=["Name", "Age"])

EXPECTED_STATISTIC = True
EXPECTED_VIOLATION = {
    "feature_name": "Name",
    "constraint_check_type": "EmailNotAllowed",
    "description": "Metric hasEmail True for Name with email allowed set to False",
    "metric_name": "email",
}


class TestEmail(unittest.TestCase):
    def test_calculate_statistics(self):
        statistic = instance.calculate_statistics(DF["Name"])
        self.assertEqual(EXPECTED_STATISTIC, statistic)

    def test_evaluate_constraints(self):
        statistics = True
        no_violation = instance.evaluate_constraints(
            statistics, DF["Name"], CONSTRAINT_NO_VIOLATION
        )
        violation = instance.evaluate_constraints(statistics, DF["Name"], CONSTRAINT_WITH_VIOLATION)
        self.assertIsNone(no_violation)
        self.assertEqual(EXPECTED_VIOLATION, violation)

    def test_suggest_constraints(self):
        statistics = True
        suggested_baseline = instance.suggest_constraints(statistics, DF["Name"], None)
        self.assertEqual(CONSTRAINT_NO_VIOLATION, suggested_baseline)
