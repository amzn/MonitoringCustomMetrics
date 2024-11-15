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

from typing import Dict, Union

import pandas as pd

from src.monitoring_custom_metrics.model_quality.model_quality_metric import ModelQualityMetric
from src.model.model_quality_attributes import ModelQualityAttributes
from src.model.model_quality_constraint import ModelQualityConstraint
from src.model.model_quality_statistic import ModelQualityStatistic
from src.model.violation import Violation

"""
Score difference measures the absolute/relative difference between predicted probability and the actual outcome.
It supports two types of comparison: absolute difference and relative difference. Absolute difference is the default option.
It supports two types of constraints:
- two_sided = True will set the constraint and violation policy by the absolute value of the score difference.
- two_sided = False will set the constraint and violation policy by the original value of the score difference.
To detect over-prediction only, set comparison_operator = "GreaterThanThreshold"
To detect under-prediction only, set comparison_operator = "LessThanThreshold".
"""


class ScoreDiff(ModelQualityMetric):
    def calculate_statistics(
        self, df: pd.DataFrame, config: Dict, model_quality_attributes: ModelQualityAttributes
    ) -> ModelQualityStatistic:
        """
        Score difference calculation requires following parameters:
        comparison_type: String, absolute/relative, default = "absolute"
        """
        actual = model_quality_attributes.ground_truth_attribute
        pred = model_quality_attributes.probability_attribute
        if df[actual].isnull().values.any():
            raise ValueError("Missing value in {} column".format(actual))
        if df[pred].isnull().values.any():
            raise ValueError("Missing value in {} column".format(pred))

        comparison_type = config.get("comparison_type", "absolute")
        if comparison_type == "absolute":
            statistics = df[pred].mean() - df[actual].mean()
        elif comparison_type == "relative":
            if df[actual].mean() == 0:
                raise ZeroDivisionError("Denominator cannot be zero")
            statistics = (df[pred].mean() - df[actual].mean()) / df[actual].mean()
        else:
            raise ValueError("invalid comparison type")

        return {"value": statistics.round(decimals=4), "standard_deviation": 0}

    def evaluate_constraints(
        self,
        statistics: ModelQualityStatistic,
        df: pd.DataFrame,
        config: Dict,
        constraint: ModelQualityConstraint,
        model_quality_attributes: ModelQualityAttributes,
    ) -> Union[Violation, None]:
        custom_metric = statistics["value"]
        metric_name = "score_diff"

        threshold = 0.0
        if "threshold" in constraint and constraint["threshold"] is not None:
            threshold = constraint["threshold"]
        comparison_operator = constraint["comparison_operator"]

        in_violation = False
        if comparison_operator == "GreaterThanThreshold":
            in_violation = custom_metric > threshold
        elif comparison_operator == "LessThanThreshold":
            in_violation = custom_metric < threshold

        if in_violation:
            return Violation(
                constraint_check_type="{}".format(comparison_operator),
                description="Metric {} with {} was {} {}".format(
                    metric_name, custom_metric, comparison_operator, threshold
                ),
                metric_name="{}".format(metric_name),
            )
        return None

    def suggest_constraints(
        self,
        statistics: ModelQualityStatistic,
        df: pd.DataFrame,
        config: Dict,
        model_quality_attributes: ModelQualityAttributes,
    ) -> ModelQualityConstraint:
        """
        two_sided: Boolean, indicating whether to return the absolute value of the score difference, default = False
        comparison_operator: String, GreaterThanThreshold or LessThanThreshold
        """
        statistics_value = statistics["value"]
        two_sided = config.get("two_sided", False)
        custom_metric = abs(statistics_value) if two_sided else statistics_value
        # threshold_override > 0 means the threshold is set above the baseline. In this case, we will accept some deterioration in the metrics.
        threshold_override = config["threshold_override"] if "threshold_override" in config else 0
        threshold = custom_metric + threshold_override
        comparison_operator = (
            "GreaterThanThreshold"
            if two_sided
            else config.get("comparison_operator", "LessThanThreshold")
        )
        return ModelQualityConstraint(
            threshold=threshold, comparison_operator=comparison_operator, additional_properties=None
        )


instance = ScoreDiff()
