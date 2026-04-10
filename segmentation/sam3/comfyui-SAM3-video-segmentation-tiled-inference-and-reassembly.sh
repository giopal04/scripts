#!/bin/bash

# process-urban-videos.sh
# Runs SAM3 segmentation (two tiles/GPUs) for each prompt/color combination
# on every .mp4 in the dataset directory, then reassembles before the next file.

#set -euo pipefail
set -uo pipefail

BASE_DIR="/mnt/raid1/dataset/urban/elaborati-rilievo-fotografico-video"

# ── Prompt / color / color_step combos ──────────────────────────────────────
PROMPTS=(
    "window"
    "zebra_crossing"
    "door double_door front_door gate"
    "sidewalk"
    "tree"
    "car van SUV bus vehicle"
)
#PROMPTS=(					# test with just two prompts
#    "door double_door front_door gate"
#    "sidewalk"
#)
COLORS=(   blue   red   green   cyan   magenta   yellow )
COLOR_STEPS=(  30     5     5       5      10        30  )

# ── Iterate over every .mp4 in the dataset directory ────────────────────────
for mp4_path in "$BASE_DIR"/*.mp4; do
#for mp4_path in "$BASE_DIR"/Track_X-Sphere.mp4 "$BASE_DIR"/Track_X-Sphere.mp4 ; do		# test with the two smaller files
    [[ -f "$mp4_path" ]] || { echo "No .mp4 files found in $BASE_DIR"; exit 1; }

    filename="$(basename "$mp4_path")"          # e.g. Track_F-Sphere.mp4
    name="${filename%.mp4}"                     # e.g. Track_F-Sphere
    wip_dir="$BASE_DIR/$name/wip"
    base_outdir="$BASE_DIR/$name"

    echo "========================================================"
    echo "Processing: $filename"
    echo "  wip_dir   : $wip_dir"
    echo "  base_outdir: $base_outdir"
    echo "========================================================"

    mkdir -p "$wip_dir"

    # ── For each prompt+color+color_step combo ───────────────────────────────
    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        color="${COLORS[$i]}"
        color_step="${COLOR_STEPS[$i]}"

        echo ""
        echo "  ── Prompt: '$prompt'  color: $color  color_step: $color_step"

        # --- Left tile – GPU 0 ------------------------------------------------
        echo "    [GPU 0] tile 0,0,3518,2440"
        ./comfyui-SAM3-video-segmentation.py \
            --input-video  "$mp4_path" \
            --prompt       "$prompt" \
            --chunk-size   50 \
            --num-frames   1000 \
            --tile         0,0,3518,2440 \
            --tile-resize  1008x1008 \
            --gpu          0 \
            --overlay-color  "$color" \
            --overlay-alpha  0.9 \
            --color-step   "$color_step" \
            --output-dir   "$wip_dir"

        # --- Right tile – GPU 1 -----------------------------------------------
        echo "    [GPU 1] tile 3518,0,3518,2440"
        ./comfyui-SAM3-video-segmentation.py \
            --input-video  "$mp4_path" \
            --prompt       "$prompt" \
            --chunk-size   50 \
            --num-frames   1000 \
            --tile         3518,0,3518,2440 \
            --tile-resize  1008x1008 \
            --gpu          1 \
            --overlay-color  "$color" \
            --overlay-alpha  0.9 \
            --color-step   "$color_step" \
            --output-dir   "$wip_dir"

    done  # end prompt loop

    # ── Reassemble before moving to the next .mp4 ───────────────────────────
    echo ""
    echo "  ── Reassembling tiles for $name …"

    # Glob-expand the master file at runtime (the * covers any timestamp/suffix
    # the segmentation script may insert after y0)
    #master_glob="$wip_dir/${name}-master-tile-x0-y0"*".mp4"
    master_glob="$BASE_DIR/${name}.mp4"
    master_file=""
    for f in $master_glob; do
        [[ -f "$f" ]] && { master_file="$f"; break; }
    done

    if [[ -z "$master_file" ]]; then
        echo "  WARNING: no master file matched '$master_glob' – skipping reassemble for $name"
    else
        echo "master file    : $master_file"
	new_master_file="$wip_dir"/`basename "$master_file"`
        echo "new master file: $new_master_file"
	ln "$master_file" "$wip_dir"
        ./reassemble-SAM3-mask-video-tiles.py \
            --write-images \
            --output-dir "$base_outdir" \
            "$new_master_file"
    fi

    echo "  Done: $filename"
    echo ""

done  # end mp4 loop

echo "All files processed."
