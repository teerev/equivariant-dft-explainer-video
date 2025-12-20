#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Usage:
#   ./get_trev2.sh input.mov
#   ./get_trev2.sh --keep-workdir input.mov
#
# Output:
#   Writes <input>_trev.mov in the SAME directory as input
#
# Guarantees:
# - Handles VFR -> CFR
# - Uses MPS on Apple Silicon
# - Corrects RVM time offset (skipped initial frames)
# - Outputs ProRes 4444 with alpha
# ============================================================

KEEP_WORKDIR=0
if [ "${1:-}" = "--keep-workdir" ]; then
  KEEP_WORKDIR=1
  shift
fi

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [--keep-workdir] input.mov"
  exit 1
fi

IN="$1"
IN_DIR="$(cd "$(dirname "$IN")" && pwd)"
BASENAME="$(basename "$IN")"
STEM="${BASENAME%.*}"
OUT="${IN_DIR}/${STEM}_trev.mov"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
RVM_DIR="RobustVideoMatting"
INFERENCE="${RVM_DIR}/inference.py"
MODEL="${RVM_DIR}/checkpoints/rvm_mobilenetv3.pth"
DEVICE="${DEVICE:-mps}"

WORKDIR="$(mktemp -d)"
ALPHA_DIR="${WORKDIR}/alpha"
mkdir -p "${ALPHA_DIR}"

cleanup() {
  if [ "$KEEP_WORKDIR" -eq 1 ]; then
    echo
    echo "Keeping workdir: ${WORKDIR}"
  else
    rm -rf "${WORKDIR}"
  fi
}

trap 'rc=$?; if [ $rc -eq 0 ]; then cleanup; else echo; echo "ERROR (rc=$rc). Keeping workdir: ${WORKDIR}"; fi; exit $rc' EXIT

hr() { printf "\n============================================================\n"; }

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
get_duration_seconds() {
  ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$1"
}

get_avg_fps_frac() {
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=avg_frame_rate \
    -of default=nw=1:nk=1 "$1"
}

calc_fps_float() {
  python3 - <<PY
from fractions import Fraction
f = Fraction("${1}")
print(float(f) if f.numerator and f.denominator else 30.0)
PY
}

# ------------------------------------------------------------
# ffmpeg progress helper
# ------------------------------------------------------------
ffmpeg_with_progress() {
  local label="$1"
  local duration="$2"
  shift 2

  echo "=== ${label} ==="

  ffmpeg -hide_banner -nostats -loglevel error \
    -progress pipe:1 \
    "$@" 2>/dev/null | awk -v D="$duration" -v LABEL="$label" '
      function fmt(sec){
        h=int(sec/3600); m=int((sec%3600)/60); s=int(sec%60)
        return (h>0)?sprintf("%02d:%02d:%02d",h,m,s):sprintf("%02d:%02d",m,s)
      }
      /^out_time_ms=/{
        t=$0; sub(/.*=/,"",t); t/=1000000
        p=int(100*t/D+0.5)
        printf("\r%s: %3d%% | %s / %s", LABEL, p, fmt(t), fmt(D))
        fflush()
      }
      /^progress=end/{
        printf("\r%s: 100%% | %s / %s\n", LABEL, fmt(D), fmt(D))
      }
    '
}

# ------------------------------------------------------------
# 0) CFR transcode
# ------------------------------------------------------------
hr
FPS_FRAC="$(get_avg_fps_frac "$IN")"
FPS="$(calc_fps_float "$FPS_FRAC")"
DUR="$(get_duration_seconds "$IN")"
CFR="${WORKDIR}/cfr.mp4"

echo "Input FPS: ${FPS_FRAC}  =>  ${FPS}"
echo "Duration : ${DUR} s"
echo "Workdir  : ${WORKDIR}"

ffmpeg_with_progress "CFR transcode (videotoolbox)" "$DUR" \
  -y -i "$IN" \
  -vsync cfr -r "$FPS" \
  -c:v h264_videotoolbox -b:v 20M -pix_fmt yuv420p \
  -c:a aac -b:a 192k \
  "$CFR"

# ------------------------------------------------------------
# 1) RVM inference
# ------------------------------------------------------------
hr
echo "=== RVM alpha inference (device: ${DEVICE}) ==="

PYTORCH_ENABLE_MPS_FALLBACK=1 \
python3 -u "${INFERENCE}" \
  --variant mobilenetv3 \
  --checkpoint "${MODEL}" \
  --device "${DEVICE}" \
  --input-source "${CFR}" \
  --output-type png_sequence \
  --output-alpha "${ALPHA_DIR}"


# ------------------------------------------------------------
# 2) Validate + compute offset
# ------------------------------------------------------------
hr

ALPHA_COUNT="$(find "${ALPHA_DIR}" -type f -name '*.png' | wc -l | tr -d ' ')"
if [ "$ALPHA_COUNT" -eq 0 ]; then
  echo "ERROR: RVM produced zero alpha frames"
  exit 2
fi

VID_COUNT="$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames \
  -of default=nw=1:nk=1 "$CFR")"

SKIP=$((VID_COUNT - ALPHA_COUNT))
if [ "$SKIP" -lt 0 ]; then SKIP=0; fi

echo "Video frames : $VID_COUNT"
echo "Alpha frames : $ALPHA_COUNT"
echo "Offset (SKIP): $SKIP"


# ------------------------------------------------------------
# 3) Merge with CORRECT alignment
# ------------------------------------------------------------
hr
ffmpeg_with_progress "Merge RGB + alpha (aligned)" "$DUR" \
  -y \
  -i "$CFR" \
  -framerate "$FPS" -pattern_type glob -i "${ALPHA_DIR}/*.png" \
  -filter_complex \
    "[0:v]select='gte(n,${SKIP})',setpts=PTS-STARTPTS,format=rgba[rgb]; \
     [1:v]setpts=PTS-STARTPTS,format=gray[alpha]; \
     [rgb][alpha]alphamerge[v]" \
  -map "[v]" -map '0:a?' \
  -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le \
  -c:a pcm_s16le \
  -shortest \
  "$OUT"

# ------------------------------------------------------------
hr
echo "Done."
echo "Wrote: ${OUT}"
