
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
from typing import Type

# Local
from .eval_metrics import EvalMetrics
from .history_based_metrics import HistoryBasedMetric
from .loss import Loss
from .trainingstate import TrainingState
from tuning.traincoco.controllermetrics.per_process_state import PerProcessState

# List of metric handlers
handlers = []


def register(cl: Type):
    """Registers the list of metric handlers by adding to the handler list.

    Args:
        cl: Class type of the handler
    """
    handlers.append(cl)


# Register the default metric handlers in this package here
register(TrainingState)
register(PerProcessState)
register(EvalMetrics)
register(Loss)
register(HistoryBasedMetric)
