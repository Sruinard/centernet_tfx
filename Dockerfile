# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FROM tensorflow/tfx:latest
FROM valerianomanassero/tfx-nvidia-gpu:1.0.4
# Change to following base image if docker hub is preferred.
# FROM tensorflow/tfx_base:latest

LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

# TODO(zhitaoli): Remove pinned version of tensorflow and related packages here
# once we switch default tensorflow version in released image to TF 2.x.
# TODO(b/151392812): Remove `google-api-python-client` and `google-apitools`
#                    when patching is not needed any more.

# docker build command should be run under root directory of github checkout.
RUN pip install --no-cache-dir tensorflow==2.3.1

ENV TFX_SRC_DIR=/tfx-src
ADD . ${TFX_SRC_DIR}
WORKDIR ${TFX_SRC_DIR}

# TODO(b/166202742): Consolidate container entrypoint with Kubeflow runner.

COPY core ${TFX_SRC_DIR}/core
COPY trainer ${TFX_SRC_DIR}/trainer/


# ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["python", "-m", "tfx.scripts.run_executor"]