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
from typing import Any, Dict, List, Union

import pandas

from src.monitoring_custom_metrics.constant import BASELINE_CONSTRAINTS_ENV_VAR
from src.monitoring_custom_metrics.output_generator import write_results_to_output_folder
from src.monitoring_custom_metrics.path_helper import (
    get_data_quality_class_path,
    retrieve_modules,
)
from src.monitoring_custom_metrics.util import (
    retrieve_json_file_in_path,
    validate_environment_variable,
)
from src.model.data_type import DataType
from src.model.operation_type import OperationType


def translate_data_type(column_data_type):
    if column_data_type == "object":
        return DataType.String
    elif column_data_type == "bool":
        return DataType.String
    elif column_data_type == "int64":
        return DataType.Integral
    elif column_data_type == "float64":
        return DataType.Fractional


def get_data_types_for_columns(df) -> Dict:
    df.info()
    return df.dtypes.to_dict()


def execute_operation_for_data_quality(
    operation_type: OperationType,
    data_type: DataType,
    column: Union[pandas.Series, pandas.DataFrame],
    constraint_from_file: Any = None,
) -> List:
    output_statistics_features: List = []
    output_constraints: List = []
    output_constraint_violations: List = []
    feature = None
    constraint = None
    violation = None  # noqa
    class_path = get_data_quality_class_path(data_type)

    if data_type is DataType.String:
        feature = {
            "name": column.name,
            "inferred_type": data_type.name,
            "string_statistics": {},
        }
        constraint = {
            "name": column.name,
            "inferred_type": data_type.name,
            "string_constraints": {},
        }
    else:
        feature = {
            "name": column.name,
            "inferred_type": data_type.name,
            "numerical_statistics": {},
        }
        constraint = {
            "name": column.name,
            "inferred_type": data_type.name,
            "num_constraints": {},
        }

    feature_dict = {}

    if constraint_from_file is not None:
        feature_constraints = (
            constraint_from_file["features"] if constraint_from_file is not None else None
        )
        for feature_constraint in feature_constraints:
            if "num_constraints" in feature_constraint:
                feature_dict[feature_constraint["name"]] = {
                    "inferred_type": feature_constraint["inferred_type"],
                    "num_constraints": feature_constraint["num_constraints"],
                }
            else:
                feature_dict[feature_constraint["name"]] = {
                    "inferred_type": feature_constraint["inferred_type"],
                    "string_constraints": feature_constraint["string_constraints"],
                }

    modules = retrieve_modules(class_path)
    constraint_type = "string_constraints" if DataType.String == data_type else "num_constraints"

    for module in modules:
        instance = module.instance
        module_statistics = instance.calculate_statistics(column)
        statistic_type = (
            "string_statistics" if DataType.String == data_type else "numerical_statistics"
        )

        feature[statistic_type]["common"] = calculate_common_statistics(column)
        feature[statistic_type][module.__name__] = module_statistics

        if operation_type == OperationType.run_monitor:
            result = instance.evaluate_constraints(
                module_statistics,
                column,
                feature_dict[column.name][constraint_type][module.__name__],
            )
            if result is not None:
                output_constraint_violations.append(result)

        original_constraints = None

        if (
            column.name in feature_dict
            and module.__name__ in feature_dict[column.name][constraint_type]
        ):
            original_constraints = feature_dict[column.name][constraint_type][module.__name__]

        constraint[constraint_type][module.__name__] = instance.suggest_constraints(
            module_statistics, column, original_constraints
        )

    output_statistics_features.append(feature)
    output_constraints.append(constraint)

    return [output_statistics_features, output_constraints, output_constraint_violations]


def validate_environment_variables(operation_type):
    if operation_type == OperationType.run_monitor:
        validate_environment_variable(BASELINE_CONSTRAINTS_ENV_VAR)


def calculate_common_statistics(column: Union[pandas.Series, pandas.DataFrame]) -> Any:
    num_present = column.count()
    num_missing = column.isnull().sum()
    output = {"num_present": num_present, "num_missing": num_missing}

    return output


def execute_for_data_quality(operation_type: OperationType, df: pandas.DataFrame) -> List:
    validate_environment_variables(operation_type)
    constraint: Any = None

    if operation_type == OperationType.run_monitor:
        constraint = retrieve_json_file_in_path(os.environ[BASELINE_CONSTRAINTS_ENV_VAR])

    dtypes: Dict = get_data_types_for_columns(df)
    output_statistic_features: List[str] = []
    output_constraints: List[str] = []
    output_constraint_violations: List[str] = []

    for column_name, column_data_type in dtypes.items():
        data_type: DataType = translate_data_type(column_data_type)
        result = execute_operation_for_data_quality(
            operation_type, data_type, df[column_name], constraint
        )
        output_statistic_features = output_statistic_features + result[0]
        output_constraints = output_constraints + result[1]
        output_constraint_violations = output_constraint_violations + result[2]

    item_count = len(df.index)

    output_statistic: Any = {
        "version": 0.0,
        "dataset": {
            "item_count": item_count,
        },
        "features": output_statistic_features,
    }

    output_constraint: Any = {"version": 0.0, "features": output_constraints}

    output_violation: Any = None

    if operation_type == OperationType.run_monitor:
        output_violation = {"violations": output_constraint_violations}

    result = [output_statistic, output_constraint, output_violation]
    write_results_to_output_folder(result)
    return result
