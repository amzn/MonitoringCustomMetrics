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

from abc import abstractmethod, ABC
from typing import Union, Dict

import pandas

from src.model.model_quality_attributes import ModelQualityAttributes
from src.model.model_quality_constraint import ModelQualityConstraint
from src.model.model_quality_statistic import ModelQualityStatistic
from src.model.violation import Violation


class ModelQualityMetric(ABC):
    @abstractmethod
    def calculate_statistics(
        self, df: pandas.DataFrame, config: Dict, model_quality_attributes: ModelQualityAttributes
    ) -> ModelQualityStatistic:
        pass

    @abstractmethod
    def evaluate_constraints(
        self,
        statistics: ModelQualityStatistic,
        df: pandas.DataFrame,
        config: Dict,
        constraint: ModelQualityConstraint,
        model_quality_attributes: ModelQualityAttributes,
    ) -> Union[Violation, None]:
        pass

    @abstractmethod
    def suggest_constraints(
        self,
        statistics: ModelQualityStatistic,
        df: pandas.DataFrame,
        config: Dict,
        model_quality_attributes: ModelQualityAttributes,
    ) -> ModelQualityConstraint:
        pass
