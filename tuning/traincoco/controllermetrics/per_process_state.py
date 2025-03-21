
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from typing import Any

# Third Party
from transformers import TrainerState
import torch

# Local
from tuning.traincoco.controllermetrics.metricshandler import MetricHandler


class PerProcessState(MetricHandler):
    """Implements the controller metric which exposes the per process state"""

    def __init__(self, **kwargs):
        """Initializes the metric handler, by registering the event \
            list and arguments with base handler.

        Args:
            kwargs: List of arguments (key, value)-pairs
        """
        super().__init__(
            events=[
                "on_init_end",
                "on_step_end",
                "on_epoch_begin",
                "on_epoch_end",
                "on_prediction_step",
                "on_predict",
                "on_log",
                "on_train_end",
                "on_train_begin",
                "on_evaluate",
                "on_save",
            ],
            **kwargs,
        )

    def validate(self) -> bool:
        """Validate the training arguments (e.g logging_steps) are \
            compatible with the computation of this metric.

        Returns:
            bool
        """
        return True

    def compute(self, _: TrainerState = None, **kwargs) -> Any:
        """Exposes the trainer state.

        Args:
            state: TrainerState object
            kwargs: Remaining event arguments

        Returns:
            dict. Trainer state as a dictionary
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return {"rank": torch.distributed.get_rank()}
        return {"rank": None}
