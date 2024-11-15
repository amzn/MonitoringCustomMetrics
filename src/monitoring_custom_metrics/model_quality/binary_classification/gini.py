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
import numpy as np

from src.monitoring_custom_metrics.model_quality.model_quality_metric import ModelQualityMetric
from src.model.model_quality_attributes import ModelQualityAttributes
from src.model.model_quality_constraint import ModelQualityConstraint
from src.model.model_quality_statistic import ModelQualityStatistic
from src.model.violation import Violation

"""
GINI is a model performance metric that measures the ranking power of a model. It ranges from 0 to 1 - 0 means no ranking power
while 1 means perfect ranking power.
"""


class Gini(ModelQualityMetric):
    def calculate_statistics(
        self, df: pd.DataFrame, config: Dict, model_quality_attributes: ModelQualityAttributes
    ) -> ModelQualityStatistic:
        actual = model_quality_attributes.ground_truth_attribute
        pred = model_quality_attributes.probability_attribute
        if df[actual].isnull().values.any():
            raise ValueError("Missing value in {} column".format(actual))
        if df[pred].isnull().values.any():
            raise ValueError("Missing value in {} column".format(pred))

        # To ensure enough samples in every bin, use 5 bins with <100 samples and 10 bins with >100 samples
        if len(df) < 100:
            n_bins = 5
        else:
            n_bins = 10

        df["bins"] = pd.qcut(df[pred], n_bins, labels=False, duplicates="drop")

        def agg_func(x, actual, pred):
            agg_metrics = dict()
            agg_metrics["total_cnt"] = x[actual].count()
            agg_metrics["pos_cnt"] = x[actual].sum()
            agg_metrics["neg_cnt"] = agg_metrics["total_cnt"] - agg_metrics["pos_cnt"]
            return pd.Series(agg_metrics, index=["total_cnt", "pos_cnt", "neg_cnt"])

        df_grouped = df.groupby("bins").apply(agg_func, actual=actual, pred=pred).reset_index()
        df_grouped.sort_values("bins", ascending=False, inplace=True)

        df_grouped["cum_pos_pct"] = df_grouped["pos_cnt"].cumsum() / df_grouped["pos_cnt"].sum()
        df_grouped["cum_pos_pct_lag"] = df_grouped["cum_pos_pct"].shift(1, fill_value=0)
        df_grouped["cum_neg_pct"] = df_grouped["neg_cnt"].cumsum() / df_grouped["neg_cnt"].sum()
        df_grouped["cum_neg_pct_lag"] = df_grouped["cum_neg_pct"].shift(1, fill_value=0)
        df_grouped["incr_pos_pct"] = df_grouped["cum_pos_pct"] - df_grouped["cum_pos_pct_lag"]
        df_grouped["sum_neg_pct"] = df_grouped["cum_neg_pct"] + df_grouped["cum_neg_pct_lag"]
        gini = 1 - np.dot(df_grouped.sum_neg_pct, df_grouped.incr_pos_pct)

        return {"value": gini.round(4), "standard_deviation": 0}

    def evaluate_constraints(
        self,
        statistics: ModelQualityStatistic,
        df: pd.DataFrame,
        config: Dict,
        constraint: ModelQualityConstraint,
        model_quality_attributes: ModelQualityAttributes,
    ) -> Union[Violation, None]:
        custom_metric = statistics["value"]
        metric_name = "gini"

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
            threshold=custom_metric + threshold_override,
            comparison_operator="LessThanThreshold",
            additional_properties=None,
        )


instance = Gini()
