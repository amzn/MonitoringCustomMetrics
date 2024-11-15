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

import pandas

from src.monitoring_custom_metrics.constant import (
    BASELINE_STATISTICS_ENV_VAR,
    BASELINE_CONSTRAINTS_ENV_VAR,
    ANALYSIS_TYPE_ENV_VAR,
    GROUND_TRUTH_ATTRIBUTE_ENV_VAR,
)
from src.monitoring_custom_metrics.monitor_data_quality import execute_for_data_quality
from src.monitoring_custom_metrics.monitor_model_quality import execute_for_model_quality
from src.monitoring_custom_metrics.path_helper import import_class_paths
from src.monitoring_custom_metrics.util import (
    get_dataframe_from_csv,
)
from src.model.monitor_type import MonitorType
from src.model.operation_type import OperationType


def determine_operation_to_run() -> OperationType:
    print("Determining operation to run based on provided parameters ...")
    if (
        os.environ.get(BASELINE_STATISTICS_ENV_VAR) is not None
        and os.environ.get(BASELINE_CONSTRAINTS_ENV_VAR) is not None
    ):
        return OperationType.run_monitor
    elif (
        os.environ.get(BASELINE_STATISTICS_ENV_VAR) is None
        and os.environ.get(BASELINE_CONSTRAINTS_ENV_VAR) is None
    ):
        return OperationType.suggest_baseline
    else:
        raise RuntimeError(
            f"For evaluate constraints operation, both '{BASELINE_STATISTICS_ENV_VAR}' and '{BASELINE_CONSTRAINTS_ENV_VAR}'"
            f" environment variables must be provided."
        )


def determine_monitor_type() -> MonitorType:
    print("Determining monitor type ...")
    if os.environ.get(ANALYSIS_TYPE_ENV_VAR) is not None:
        print("Monitor type detected based on 'analysis_type' environment variable")
        return MonitorType(os.environ[ANALYSIS_TYPE_ENV_VAR])
    else:
        if os.environ.get(GROUND_TRUTH_ATTRIBUTE_ENV_VAR) is not None:
            print(
                "Monitor type detected based on the existence of 'ground_truth_attribute' environment variable"
            )
            return MonitorType.MODEL_QUALITY
        else:
            print(
                "Monitor type detected based on the absense of 'ground_truth_attribute' environment variable"
            )
            return MonitorType.DATA_QUALITY


def monitoring():
    print("Starting Monitoring Custom Metrics")
    import_class_paths()
    df: pandas.DataFrame = get_dataframe_from_csv()
    operation_type: OperationType = determine_operation_to_run()
    monitor_type: MonitorType = determine_monitor_type()

    print(f"Operation type: {operation_type}")
    print(f"Monitor type: {monitor_type}")

    if monitor_type == MonitorType.MODEL_QUALITY:
        execute_for_model_quality(operation_type, df)
    elif monitor_type == MonitorType.DATA_QUALITY:
        execute_for_data_quality(operation_type, df)
    else:
        raise ValueError(f"Monitor type {monitor_type} not valid")


if __name__ == "__main__":
    monitoring()
