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

import glob
import os
import sys

from src.model.problem_type import ProblemType
from src.model.data_type import DataType


def retrieve_modules(folder):
    modules = []
    folder_with_filter = os.path.join(folder, "*.py")

    for file in glob.glob(folder_with_filter):
        filename = os.path.splitext(os.path.basename(file))[0]

        module = __import__(filename)

        if filename != "__init__":
            modules.append(module)
    return modules


def import_class_paths():
    data_quality_numerical_path = get_data_quality_class_path(DataType.Integral)
    data_quality_string_path = get_data_quality_class_path(DataType.String)

    model_quality_binary_classification_path = get_model_quality_class_path(
        ProblemType.binary_classification
    )
    model_quality_multiclass_classification_path = get_model_quality_class_path(
        ProblemType.multiclass_classification
    )
    model_quality_regression_path = get_model_quality_class_path(ProblemType.regression)
    sys.path.insert(1, data_quality_numerical_path)
    sys.path.insert(1, data_quality_string_path)
    sys.path.insert(1, model_quality_binary_classification_path)
    sys.path.insert(1, model_quality_multiclass_classification_path)
    sys.path.insert(1, model_quality_regression_path)


def get_data_quality_class_path(data_type):
    if data_type == DataType.Integral or data_type == DataType.Fractional:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_quality", "numerical")
    elif data_type == DataType.String:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_quality", "string")
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")


def get_model_quality_class_path(problem_type):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_quality", problem_type.name
    )
