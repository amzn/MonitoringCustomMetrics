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


class ModelQualityAttributes:
    ground_truth_attribute = None
    probability_attribute = None
    probability_threshold_attribute = None
    inference_attribute = None

    def __init__(
        self,
        ground_truth_attribute,
        probability_attribute,
        probability_threshold_attribute,
        inference_attribute,
    ):
        self.ground_truth_attribute = ground_truth_attribute
        self.probability_attribute = probability_attribute
        self.probability_threshold_attribute = probability_threshold_attribute
        self.inference_attribute = inference_attribute
