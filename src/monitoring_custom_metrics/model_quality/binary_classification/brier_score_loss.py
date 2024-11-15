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
from sklearn.metrics import brier_score_loss

from src.monitoring_custom_metrics.model_quality.model_quality_metric import ModelQualityMetric
from src.model.model_quality_attributes import ModelQualityAttributes
from src.model.model_quality_constraint import ModelQualityConstraint
from src.model.model_quality_statistic import ModelQualityStatistic
from src.model.violation import Violation

"""
The Brier score measures the mean squared difference between the predicted probability and the actual outcome.
Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
"""


class BrierScoreLoss(ModelQualityMetric):
    def calculate_statistics(
        self, df: pd.DataFrame, config: Dict, model_quality_attributes: ModelQualityAttributes
    ) -> ModelQualityStatistic:
        actual = model_quality_attributes.ground_truth_attribute
        pred = model_quality_attributes.probability_attribute
        if df[actual].isnull().values.any():
            raise ValueError("Missing value in {} column".format(actual))
        if df[pred].isnull().values.any():
            raise ValueError("Missing value in {} column".format(pred))
        return {
            "value": brier_score_loss(df[actual], df[pred]).round(decimals=4),
            "standard_deviation": 0,
        }

    def evaluate_constraints(
        self,
        statistics: ModelQualityStatistic,
        df: pd.DataFrame,
        config: Dict,
        constraint: ModelQualityConstraint,
        model_quality_attributes: ModelQualityAttributes,
    ) -> Union[Violation, None]:
        custom_metric = statistics["value"]
        metric_name = "brier_score_loss"

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
        custom_metric = statistics["value"]
        # threshold_override > 0 means the threshold is set above the baseline. In this case, we will accept some deterioration in the metrics.
        threshold_override = config["threshold_override"] if "threshold_override" in config else 0
        return ModelQualityConstraint(
            threshold=float(custom_metric + threshold_override),
            comparison_operator="GreaterThanThreshold",
            additional_properties=None,
        )


instance = BrierScoreLoss()
