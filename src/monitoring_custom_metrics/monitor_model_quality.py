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

import os
from typing import Any, Dict, List

import pandas

from src.monitoring_custom_metrics.output_generator import write_results_to_output_folder
from src.model.model_quality_attributes import ModelQualityAttributes
from src.model.problem_type import ProblemType
from src.monitoring_custom_metrics.constant import (
    CONFIG_PATH_ENV_VAR,
    BASELINE_CONSTRAINTS_ENV_VAR,
    PROBLEM_TYPE_ENV_VAR,
    GROUND_TRUTH_ATTRIBUTE_ENV_VAR,
    PROBABILITY_THRESHOLD_ATTRIBUTE_ENV_VAR,
    PROBABILITY_ATTRIBUTE_ENV_VAR,
    INFERENCE_ATTRIBUTE_ENV_VAR,
)
from src.monitoring_custom_metrics.path_helper import (
    retrieve_modules,
    get_model_quality_class_path,
)
from src.monitoring_custom_metrics.util import (
    retrieve_json_file_in_path,
    validate_environment_variable,
)
from src.model.operation_type import OperationType


def execute_operation_for_model_quality(
    operation_type: OperationType,
    problem_type: ProblemType,
    model_quality_attributes: ModelQualityAttributes,
    df: pandas.DataFrame,
    config: Any,
    constraints_label: str,
    constraint: Any = None,
) -> List:
    output_statistics_dict: Dict = {}
    output_constraints_dict: Dict = {}
    output_constraint_violations: List = []
    class_path = os.path.join(get_model_quality_class_path(problem_type))

    print(f"Retrieving modules from {class_path}")
    modules = retrieve_modules(class_path)

    print("Traversing modules for MODEL QUALITY:")
    for module in modules:
        instance = module.instance
        if module.__name__ in config:
            print(f" - {module.__name__} found in the provided config. Executing metric logic.")
            module_config = config[module.__name__]
            statistics = instance.calculate_statistics(df, module_config, model_quality_attributes)
            output_statistics_dict[module.__name__] = statistics
            output_constraints_dict[module.__name__] = instance.suggest_constraints(
                statistics, df, module_config, model_quality_attributes
            )

            if operation_type == OperationType.run_monitor:
                constraints: Dict = {}

                if (
                    constraints_label in constraint
                    and module.__name__ in constraint[constraints_label]
                ):
                    constraints = constraint[constraints_label][module.__name__]

                violations = instance.evaluate_constraints(
                    statistics,
                    df,
                    module_config,
                    constraints,
                    model_quality_attributes,
                )
                if violations is not None:
                    output_constraint_violations.append(violations)
        else:
            print(f" - {module.__name__} not found in the provided config. Skipping metric logic.")
    print("Finished traversing modules for MODEL QUALITY.")

    return [output_statistics_dict, output_constraints_dict, output_constraint_violations]


def get_model_quality_attributes() -> ModelQualityAttributes:
    ground_truth_attribute = os.environ[GROUND_TRUTH_ATTRIBUTE_ENV_VAR]
    probability_attribute = None
    probability_threshold_attribute = None
    inference_attribute = None

    if os.environ.get(PROBABILITY_ATTRIBUTE_ENV_VAR) is not None:
        probability_attribute = os.environ[PROBABILITY_ATTRIBUTE_ENV_VAR]
        probability_threshold_attribute = os.environ[PROBABILITY_THRESHOLD_ATTRIBUTE_ENV_VAR]
    else:
        inference_attribute = os.environ[INFERENCE_ATTRIBUTE_ENV_VAR]

    return ModelQualityAttributes(
        ground_truth_attribute,
        probability_attribute,
        probability_threshold_attribute,
        inference_attribute,
    )


def validate_environment_variables(operation_type):
    validate_environment_variable(PROBLEM_TYPE_ENV_VAR)
    validate_environment_variable(CONFIG_PATH_ENV_VAR)
    validate_environment_variable(GROUND_TRUTH_ATTRIBUTE_ENV_VAR)

    if operation_type == OperationType.run_monitor:
        validate_environment_variable(BASELINE_CONSTRAINTS_ENV_VAR)

    if os.environ.get(PROBABILITY_ATTRIBUTE_ENV_VAR) is not None:
        validate_environment_variable(PROBABILITY_THRESHOLD_ATTRIBUTE_ENV_VAR)
    else:
        validate_environment_variable(INFERENCE_ATTRIBUTE_ENV_VAR)


def execute_for_model_quality(operation_type: OperationType, df: pandas.DataFrame) -> List:
    validate_environment_variables(operation_type)
    problem_type: ProblemType = translate_problem_type(os.environ[PROBLEM_TYPE_ENV_VAR])
    config: Any = retrieve_json_file_in_path(os.environ[CONFIG_PATH_ENV_VAR])

    model_quality_attributes: ModelQualityAttributes = get_model_quality_attributes()

    constraint: Any = None

    if operation_type == OperationType.run_monitor:
        constraint = retrieve_json_file_in_path(os.environ[BASELINE_CONSTRAINTS_ENV_VAR])

    statistics_label: str = problem_type.name + "_metrics"
    constraints_label: str = problem_type.name + "_constraints"

    result = execute_operation_for_model_quality(
        operation_type,
        problem_type,
        model_quality_attributes,
        df,
        config,
        constraints_label,
        constraint,
    )
    item_count = len(df.index)

    output_statistic: Dict = {
        "version": 0.0,
        "dataset": {"item_count": item_count},
        statistics_label: result[0],
    }
    output_constraint: Dict = {"version": 0.0, constraints_label: result[1]}

    output_violation = None

    if operation_type == OperationType.run_monitor:
        output_violation = {"violations": result[2]}

    output_result: List = [output_statistic, output_constraint, output_violation]

    write_results_to_output_folder(output_result)

    return output_result


def translate_problem_type(problem_type) -> ProblemType:
    if problem_type == "Regression":
        return ProblemType.regression
    elif problem_type == "BinaryClassification":
        return ProblemType.binary_classification
    elif problem_type == "MulticlassClassification":
        return ProblemType.multiclass_classification
    else:
        raise ValueError(f"Invalid problem type: {problem_type}")
