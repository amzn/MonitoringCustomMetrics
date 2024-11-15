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

from typing import Union

import pandas

from src.monitoring_custom_metrics.data_quality.data_quality_metric import DataQualityMetric
from src.model.data_quality_constraint import DataQualityConstraint
from src.model.violation import Violation

# from https://emailregex.com/
email_regex = "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\\])"


class Email(DataQualityMetric):
    def calculate_statistics(
        self, column: Union[pandas.Series, pandas.DataFrame]
    ) -> Union[int, str, bool, float]:
        series = column.str.contains(email_regex)
        for single_series in series:
            if single_series:
                return True
        return False

    def evaluate_constraints(
        self,
        statistics: Union[int, str, bool, float],
        column: Union[pandas.Series, pandas.DataFrame],
        constraint: DataQualityConstraint,
    ) -> Union[Violation, None]:
        has_email = statistics
        allowed = True

        if "additional_properties" in constraint:
            if "allowed" in constraint["additional_properties"]:
                allowed = constraint["additional_properties"]["allowed"]

        in_violation = has_email and not allowed
        if in_violation:
            violation = {
                "feature_name": column.name,
                "constraint_check_type": "EmailNotAllowed",
                "description": f"Metric hasEmail {has_email} for {column.name} with email allowed set to {allowed}",
                "metric_name": "email",
            }
            return violation
        return

    def suggest_constraints(
        self,
        statistics: Union[int, str, bool, float],
        column: Union[pandas.Series, pandas.DataFrame],
        baseline_constraints: Union[DataQualityConstraint, None],
    ) -> DataQualityConstraint:
        constraint = DataQualityConstraint(
            additional_properties={"allowed": statistics}, upper_bound=None, lower_bound=None
        )

        return constraint


instance = Email()
