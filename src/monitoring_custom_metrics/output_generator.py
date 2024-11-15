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
import json
from typing import List

from src.monitoring_custom_metrics.constant import (
    OUTPUT_PATH_ENV_VAR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_MESSAGE_PATH,
    OUTPUT_MESSAGE_FILE_NAME,
)


def write_results_to_output_folder(result):
    local_output_path = DEFAULT_OUTPUT_PATH

    if os.environ.get(OUTPUT_PATH_ENV_VAR) is not None:
        local_output_path = str(os.environ.get(OUTPUT_PATH_ENV_VAR))

    if result[0] is not None:
        output_path = os.path.join(local_output_path, "community_statistics.json")
        write_output_file(result[0], output_path)
    if result[1] is not None:
        output_path = os.path.join(local_output_path, "community_constraints.json")
        write_output_file(result[1], output_path)
    if result[2] is not None:
        output_path = os.path.join(local_output_path, "community_constraint_violations.json")
        write_output_file(result[2], output_path)
        write_exit_message(result, local_output_path)


def write_exit_message(result: List, local_output_path):
    print("Writing output message ...")
    output_path = os.path.join(DEFAULT_MESSAGE_PATH, OUTPUT_MESSAGE_FILE_NAME)

    violations = result[2]["violations"]

    message = ""
    if violations is not None and violations:
        message = f"CompletedWithViolations: Job completed successfully with {len(violations)} violations."
    else:
        message = "Completed: Job completed successfully with no violations."

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(message)

    print(f"### OUTPUT MESSAGE CONTENT WRITTEN TO {output_path}:")
    print(message)
    print("### END OF OUTPUT MESSAGE CONTENT")


def write_output_file(data, output_file_path):
    formatted_json = json.dumps(data, default=int, indent=4)
    with open(output_file_path, "w") as file:
        file.write(formatted_json)

    with open(output_file_path, "r") as file:
        print(f"### OUTPUT FILE CONTENT FOR {output_file_path}:")
        print(formatted_json)
        print("### END OF OUTPUT FILE CONTENT")
