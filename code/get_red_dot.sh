#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 input.mov"
  exit 1
fi

INPUT="$1"

# Intermediate files
LASER_RGB="laser_rgb.mp4"
LASER_ALPHA="laser_alpha.mp4"

# Final output
OUTPUT="laser_pure_red.mov"

echo "=== Running laser detection ==="

python3 - <<EOF
import cv2
import numpy as np

INPUT = "$INPUT"

OUT_RGB   = "$LASER_RGB"
OUT_ALPHA = "$LASER_ALPHA"

R_MIN = 90
DELTA = 10
MIN_AREA = 2
MAX_AREA = 400
BLUR_K = 5
MIN_SCORE = 120.0

EXCLUDE_RIGHT_FRAC = 0.25
MAX_DIST = 80
last_good_centroid = None

cap = cv2.VideoCapture(INPUT)
assert cap.isOpened(), f"Cannot open input video: {INPUT}"

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

FORBIDDEN_X = (1.0 - EXCLUDE_RIGHT_FRAC) * w

fourcc = cv2.VideoWriter_fourcc(*"avc1")
rgb_writer   = cv2.VideoWriter(OUT_RGB,   fourcc, fps, (w, h))
alpha_writer = cv2.VideoWriter(OUT_ALPHA, fourcc, fps, (w, h))

def score_blob(frame, stats):
    x, y, bw, bh, area = stats
    roi = frame[y:y+bh, x:x+bw].astype(np.int16)
    b, g, r = cv2.split(roi)
    return 2.5 * np.mean(r - (g + b) / 2) + 0.7 * np.mean(r) - area * 0.04

def centroid_of(stats):
    x, y, bw, bh, _ = stats
    return (x + bw / 2.0, y + bh / 2.0)

def close_enough(c):
    global last_good_centroid
    if last_good_centroid is None:
        return True
    dx = c[0] - last_good_centroid[0]
    dy = c[1] - last_good_centroid[1]
    return dx*dx + dy*dy <= MAX_DIST * MAX_DIST

while True:
    ok, frame = cap.read()
    if not ok:
        break

    blur = cv2.GaussianBlur(frame, (BLUR_K, BLUR_K), 0)
    b, g, r = cv2.split(blur.astype(np.int16))

    mask = ((r > R_MIN) & (r > g + DELTA) & (r > b + DELTA)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_label, best_stats, best_score = None, None, -1e18

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA or area > MAX_AREA:
            continue
        c = centroid_of(stats[i])
        if c[0] > FORBIDDEN_X:
            continue
        s = score_blob(frame, stats[i])
        if s > best_score:
            best_label, best_stats, best_score = i, stats[i], s

    rgb   = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)

    if best_label is not None and best_score > MIN_SCORE:
        c = centroid_of(best_stats)
        if close_enough(c):
            alpha = (labels == best_label).astype(np.uint8) * 255
            alpha = cv2.dilate(alpha, np.ones((3,3), np.uint8), 1)
            rgb[..., 2] = 255  # pure red
            last_good_centroid = c
        else:
            last_good_centroid = None
    else:
        last_good_centroid = None

    rgb_writer.write(rgb)
    alpha_writer.write(cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR))

cap.release()
rgb_writer.release()
alpha_writer.release()

print("Python step complete.")
print("Wrote:", OUT_RGB, OUT_ALPHA)
EOF

echo "=== Assembling ProRes 4444 with alpha ==="

ffmpeg -y \
  -i "$LASER_RGB" \
  -i "$LASER_ALPHA" \
  -filter_complex "[1:v]format=gray[alpha];[0:v][alpha]alphamerge" \
  -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le \
  "$OUTPUT"

echo "=== Done ==="
echo "Final output: $OUTPUT"
