#!/usr/bin/env bash
# Download the French streaming zipformer-transducer model for sherpa-onnx.
# Override MODEL_NAME / MODEL_URL to point at a different release.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-sherpa-onnx-streaming-zipformer-fr-2023-04-14}"
MODEL_URL="${MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2}"
DEST_DIR="${DEST_DIR:-models}"

mkdir -p "${DEST_DIR}"

if [ -d "${DEST_DIR}/${MODEL_NAME}" ]; then
  echo "Model already present at ${DEST_DIR}/${MODEL_NAME}"
  exit 0
fi

echo "Downloading ${MODEL_URL}"
cd "${DEST_DIR}"
curl -L --fail -o "${MODEL_NAME}.tar.bz2" "${MODEL_URL}"
tar xjf "${MODEL_NAME}.tar.bz2"
rm "${MODEL_NAME}.tar.bz2"
echo "Done. Model at ${DEST_DIR}/${MODEL_NAME}"
