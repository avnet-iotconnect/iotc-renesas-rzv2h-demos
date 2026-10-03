#!/bin/bash
# Build the DRP-AI TVM compiler Docker image used by drpai/compile.sh.
#
# Run on an x86-64 PC with Docker (Linux, or Windows via WSL2 / Git Bash with
# Docker Desktop). One-time, ~30-60 min and ~25 GB of disk.
#
# Renesas' tools are licensed downloads (free "My Renesas" login) and are NOT
# in this repo. Put both files in drpai/downloads/ first:
#
#   1. DRP-AI Translator i8 installer   DRP-AI_Translator_i8-v*-Linux-x86_64-Install
#        https://www.renesas.com/software-tool/drp-ai-translator-i8
#   2. RZ/V2H AI SDK                    RTK0EF0180F*SJ.zip
#        https://www.renesas.com/software-tool/rzv2h-ai-software-development-kit
#        (use the AI SDK version your board's SD card was made from)
#
# The image is pinned to the DRP-AI TVM release that matches the runtime on
# the board. Check the board with:   ls /usr/lib/libtvm_runtime.so.*
# and override if yours differs:      TVM_TAG=v2.6.1 drpai/build_image.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DL="$HERE/downloads"
TVM_TAG="${TVM_TAG:-v2.5.1}"
IMAGE="${IMAGE:-drp-ai_tvm_v2h:$TVM_TAG}"

shopt -s nullglob
translator=("$DL"/DRP-AI_Translator_i8-*-Linux-x86_64-Install)
sdk_zip=("$DL"/RTK0EF0180F*SJ.zip)
toolchain=("$DL"/*toolchain*.sh)
shopt -u nullglob

if [ ${#translator[@]} -ne 1 ]; then
    echo "ERROR: expected exactly one DRP-AI_Translator_i8-*-Linux-x86_64-Install in $DL" >&2
    exit 1
fi
if [ ${#toolchain[@]} -eq 0 ]; then
    if [ ${#sdk_zip[@]} -ne 1 ]; then
        echo "ERROR: expected exactly one RTK0EF0180F*SJ.zip (RZ/V2H AI SDK) in $DL" >&2
        exit 1
    fi
    echo "=== Extracting the SDK toolchain installer from $(basename "${sdk_zip[0]}") ==="
    tmp="$DL/.sdk_unzip"
    rm -rf "$tmp" && mkdir -p "$tmp"
    unzip -q "${sdk_zip[0]}" -d "$tmp"
    found="$(find "$tmp" -name "*toolchain*.sh" | head -n 1)"
    [ -n "$found" ] || { echo "ERROR: no *toolchain*.sh inside the SDK zip" >&2; exit 1; }
    mv "$found" "$DL/"
    rm -rf "$tmp"
fi

# Docker build context: just the two installers + the Renesas Dockerfile.
CTX="$HERE/build/context"
rm -rf "$CTX" && mkdir -p "$CTX"
cp "$DL"/*toolchain*.sh "${translator[0]}" "$CTX/"

echo "=== Fetching DockerfileV2H for DRP-AI TVM $TVM_TAG ==="
curl -fsSL "https://raw.githubusercontent.com/renesas-rz/rzv_drp-ai_tvm/$TVM_TAG/DockerfileV2H" \
    -o "$CTX/Dockerfile"
# The Renesas Dockerfile clones the default branch of the repo, which moves on
# to newer releases. A model compiled with a newer DRP-AI TVM does not load on
# an older board runtime, so pin the clone to the same tag.
CLONE="git clone --recursive https://github.com/renesas-rz/rzv_drp-ai_tvm.git"
grep -q "$CLONE" "$CTX/Dockerfile" \
    || { echo "ERROR: unexpected Dockerfile layout — cannot pin the TVM tag" >&2; exit 1; }
sed -i "s#$CLONE#git clone --recursive -b $TVM_TAG https://github.com/renesas-rz/rzv_drp-ai_tvm.git#" \
    "$CTX/Dockerfile"

echo "=== Building $IMAGE (this takes a while) ==="
docker build -t "$IMAGE" --build-arg PRODUCT=V2H "$CTX"
rm -rf "$CTX"
echo ""
echo "Built $IMAGE. Next:  drpai/compile.sh model/ball_best.onnx"
