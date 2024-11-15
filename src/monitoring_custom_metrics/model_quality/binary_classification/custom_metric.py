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

import pandas

from src.monitoring_custom_metrics.model_quality.model_quality_metric import ModelQualityMetric
from src.model.model_quality_attributes import ModelQualityAttributes
from src.model.model_quality_constraint import ModelQualityConstraint
from src.model.model_quality_statistic import ModelQualityStatistic
from src.model.violation import Violation


class CustomMetric(ModelQualityMetric):
    standard_deviation = 0
    threshold_discount = 2

    def calculate_statistics(
        self, df: pandas.DataFrame, config: Dict, model_quality_attributes: ModelQualityAttributes
    ) -> ModelQualityStatistic:
        return {
            "value": df[model_quality_attributes.ground_truth_attribute].mean(),
            "standard_deviation": self.standard_deviation,
        }

    def evaluate_constraints(
        self,
        statistics: ModelQualityStatistic,
        df: pandas.DataFrame,
        config: Dict,
        constraint: ModelQualityConstraint,
        model_quality_attributes: ModelQualityAttributes,
    ) -> Union[Violation, None]:
        custom_metric = float(statistics["value"])

        # When users want to monitor a metric but don't want to set alerts on, they could leave threshold and comparison_operator as missing
        # This is to make sure no threshold or comparison_operator is a valid input, and no violation will be triggered
        if (
            "threshold" not in constraint
            or constraint["threshold"] is None
            or "comparison_operator" not in constraint
            or constraint["comparison_operator"] is None
        ):
            return None

        threshold = float(constraint["threshold"])

        threshold_override = None
        if config and "threshold_overide" in config and config["threshold_overide"] is not None:
            threshold_override = config["threshold_overide"]

        if threshold_override:
            in_violation = custom_metric > threshold_override
            threshold = threshold_override
        else:
            in_violation = custom_metric > threshold

        if in_violation:
            return Violation(
                constraint_check_type="CustomMetric",
                description=f"Metric CustomMetric {custom_metric} is above threshold {threshold}",
                metric_name="custom_metric",
            )
        return None

    def suggest_constraints(
        self,
        statistics: ModelQualityStatistic,
        df: pandas.DataFrame,
        config: Dict,
        model_quality_attributes: ModelQualityAttributes,
    ) -> ModelQualityConstraint:
        statistic = statistics["value"]

        return ModelQualityConstraint(
            threshold=statistic - self.threshold_discount,
            comparison_operator="GreaterThanThreshold",
            additional_properties=None,
        )


instance = CustomMetric()
