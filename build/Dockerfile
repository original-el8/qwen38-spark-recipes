ARG BASE_IMAGE=spark-vllm:ds41-flash-spark-2ac48a52-b12x135c9715-sm121-r2
FROM ${BASE_IMAGE}
ARG PROFILE_ID
ARG VLLM_COMMIT
ARG VLLM_TREE
ARG VLLM_VERSION
ARG B12X_COMMIT
ARG B12X_TREE
ARG BASE_IMAGE
ARG BASE_IMAGE_ID
WORKDIR /opt/spark-vllm
RUN mv /opt/spark-vllm/vllm /opt/spark-vllm/base-vllm \
    && mv /opt/spark-vllm/b12x /opt/spark-vllm/base-b12x \
    && cp /opt/spark-vllm/image-manifest.json /opt/spark-vllm/parent-image-manifest.json \
    && uv venv --clear --system-site-packages --python /usr/bin/python3.12 /opt/spark-vllm/.venv
ADD vllm.tar /opt/spark-vllm/vllm/
ADD b12x.tar /opt/spark-vllm/b12x/
COPY image-manifest.json install-candidate.py /opt/spark-vllm/
COPY test-wheels/ /opt/spark-vllm/test-wheels/
RUN uv pip install --python /opt/spark-vllm/.venv/bin/python --offline --no-deps /opt/spark-vllm/test-wheels/*.whl
RUN /opt/spark-vllm/.venv/bin/python /opt/spark-vllm/install-candidate.py
ENV PATH=/opt/spark-vllm/.venv/bin:$PATH
ENV VLLM_BUILD_COMMIT=${VLLM_COMMIT} \
    VLLM_IMAGE_TAG=spark-vllm:${PROFILE_ID}
LABEL local-inference.profile.id="${PROFILE_ID}" \
      local-inference.status="candidate" \
      local-inference.base.image-id="${BASE_IMAGE_ID}" \
      local-inference.target="linux-arm64-sm121" \
      local-inference.vllm.commit="${VLLM_COMMIT}" \
      local-inference.vllm.tree="${VLLM_TREE}" \
      local-inference.vllm.branch="codex/qwen-moe-padding-20260917" \
      local-inference.vllm.version="${VLLM_VERSION}" \
      local-inference.image.tag="spark-vllm:${PROFILE_ID}" \
      local-inference.b12x.commit="${B12X_COMMIT}" \
      local-inference.b12x.tree="${B12X_TREE}" \
      local-inference.qualification="unqualified-source-update-candidate"
WORKDIR /opt/spark-vllm/vllm
