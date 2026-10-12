#!/bin/bash
# Build the DRP-AI TVM compiler Docker image used by drpai/compile.sh.
#
# Run on an x86-64 PC with Docker: Linux, or Windows with Docker Desktop
# (from Git Bash or WSL) — a separate Linux machine is not needed. One-time,
# roughly an hour and ~30 GB of disk.
#
# The only input is the RZ/V2H AI SDK zip, a licensed download (free
# "My Renesas" login) that is NOT in this repo. Put it in drpai/downloads/:
#
#     RTK0EF0180F*SJ.zip
#     https://www.renesas.com/software-tool/rzv2h-ai-software-development-kit
#
# Use the AI SDK version your board's SD card was made from. The zip carries
# everything the image needs in its ai_sdk_setup/ folder: Renesas' Dockerfile,
# the cross toolchain, and the DRP-AI Translator i8 installer. That Dockerfile
# also pins the DRP-AI TVM release that belongs to the SDK.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DL="$HERE/downloads"
IMAGE="${IMAGE:-drp-ai_tvm_v2h:latest}"

shopt -s nullglob
sdk_zip=("$DL"/RTK0EF0180F*SJ.zip)
shopt -u nullglob
if [ ${#sdk_zip[@]} -ne 1 ]; then
    echo "ERROR: expected exactly one RTK0EF0180F*SJ.zip (RZ/V2H AI SDK) in $DL" >&2
    exit 1
fi

CTX="$HERE/build/context"
rm -rf "$CTX" && mkdir -p "$CTX"
echo "=== Extracting ai_sdk_setup/ from $(basename "${sdk_zip[0]}") ==="
unzip -j -o -q "${sdk_zip[0]}" "ai_sdk_setup/*" -d "$CTX"
for need in Dockerfile; do
    [ -f "$CTX/$need" ] || {
        echo "ERROR: this SDK zip has no ai_sdk_setup/$need — it predates the" >&2
        echo "       bundled DRP-AI TVM setup; see Renesas' rzv_drp-ai_tvm setup guide." >&2
        exit 1; }
done
ls "$CTX"/*toolchain*.sh "$CTX"/DRP-AI_Translator*Install >/dev/null

echo "=== Building $IMAGE (the SDK install step alone can take half an hour) ==="
docker build -t "$IMAGE" --build-arg PRODUCT=V2H "$CTX"
rm -rf "$CTX"
echo ""
echo "Built $IMAGE. Next (from a demo folder):  ../common/drpai/compile.sh model/ball_best.onnx"
