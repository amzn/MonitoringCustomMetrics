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

import json
import os
from typing import Any, List

import pandas
import pandas as pd

from src.monitoring_custom_metrics.constant import (
    DATASET_SOURCE_ENV_VAR,
    DEFAULT_DATA_PATH,
)


def retrieve_json_file_in_path(path) -> Any:
    if path.lower().endswith(".json"):
        return retrieve_json_file(path)
    else:
        return retrieve_first_json_file_in_path(path)


def retrieve_first_json_file_in_path(path) -> Any:
    filename = get_first_file_from_directory(path)
    return retrieve_json_file(os.path.join(path, filename))


def retrieve_json_file(file_path) -> Any:
    result = None
    with open(file_path) as file:
        result = json.load(file)
    return result


def get_dataframe_from_csv(path=None) -> pandas.DataFrame:
    folder_path: str = ""
    if os.environ.get(DATASET_SOURCE_ENV_VAR) is not None:
        folder_path = os.environ[DATASET_SOURCE_ENV_VAR]
    elif path is not None:
        folder_path = path
    else:
        folder_path = DEFAULT_DATA_PATH

    print(f"Retrieving data from path: {folder_path}")

    filenames: List[str] = get_files_in_directory(folder_path)
    data_frames = []

    for filename in filenames:
        full_path = os.path.join(folder_path, filename)
        print(f"  Reading data from file: {folder_path}")
        data_frames.append(pd.read_csv(full_path))

    print(f"Finished retrieving data from path: {folder_path}")
    return pd.concat(data_frames)


def get_first_file_from_directory(path) -> Any:
    return os.listdir(path)[0]


def get_files_in_directory(path) -> List[str]:
    return os.listdir(path)


def validate_environment_variable(env_var_name):
    if os.environ.get(env_var_name) is None:
        raise ValueError(f"Environment variable {env_var_name} is not set.")
