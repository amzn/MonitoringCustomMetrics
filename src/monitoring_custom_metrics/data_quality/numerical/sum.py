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


class Sum(DataQualityMetric):
    ten_units = 10

    def calculate_statistics(
        self, column: Union[pandas.Series, pandas.DataFrame]
    ) -> Union[int, str, bool, float]:
        return column.sum()

    def evaluate_constraints(
        self,
        statistics: Union[int, str, bool, float],
        column: Union[pandas.Series, pandas.DataFrame],
        constraint: DataQualityConstraint,
    ) -> Union[Violation, None]:
        lower_bound = constraint["lower_bound"]
        upper_bound = constraint["upper_bound"]
        in_violation = statistics > upper_bound or statistics < lower_bound
        if in_violation:
            violation = {
                "feature_name": column.name,
                "constraint_check_type": "InBetweenBound",
                "description": f"Metric sum {statistics} for {column.name} value outside of range {lower_bound} - {upper_bound}",
                "metric_name": "sum",
            }
            return violation
        return None

    def suggest_constraints(
        self,
        statistics: Union[int, str, bool, float],
        column: Union[pandas.Series, pandas.DataFrame],
        baseline_constraints: Union[DataQualityConstraint, None],
    ) -> DataQualityConstraint:
        lower_bound = statistics - (statistics / self.ten_units)
        upper_bound = statistics + (statistics / self.ten_units)

        additional_properties = None

        if baseline_constraints is not None:
            if "lower_bound" in baseline_constraints and "upper_bound" in baseline_constraints:
                additional_properties = {
                    "baseline_lower_bound": baseline_constraints["lower_bound"],
                    "baseline_upper_bound": baseline_constraints["upper_bound"],
                }

        constraint = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "additional_properties": additional_properties,
        }
        return constraint


instance = Sum()
