import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from utils.alignment_utils import AlignmentUtils

utils = AlignmentUtils()
sr = 1000 # Use low SR for easier testing

# 1s segment
seg1 = np.ones(1000) * 1
# 2s segment
seg2 = np.ones(2000) * 2

# Test case 1: No overlap
# Segment 1: 1.0s to 2.0s
# Segment 2: 3.0s to 5.0s
segments = [
    (1.0, 2.0, seg1),
    (3.0, 5.0, seg2)
]
track = utils.reconstruct_character_track(segments, sr)
print(f"Track length: {len(track)/sr}s (Expected: 5.0s)")
# Check if seg2 is at 3.0s
print(f"Value at 3.1s: {track[3100]} (Expected: 2.0)")

# Test case 2: Overlap with pushing
# Segment 1: 1.0s to 2.0s
# Segment 2: 1.5s to 3.5s (Target 1.5s should be pushed to 2.0s)
segments = [
    (1.0, 2.0, seg1),
    (1.5, 3.5, seg2)
]
track = utils.reconstruct_character_track(segments, sr)
# Seg1: 1.0-2.0
# Seg2 starts at 2.0, lasts 2.0s -> ends at 4.0s
print(f"Pushed track length: {len(track)/sr}s (Expected: 4.0s)")
print(f"Value at 1.9s: {track[1900]} (Expected: 1.0)")
print(f"Value at 2.1s: {track[2100]} (Expected: 2.0)")

# Test case 3: min_gap = 1.0s
# Segment 1 ends at 2.0s. Segment 2 target 1.5s should be pushed to 2.0 + 1.0 = 3.0s
track = utils.reconstruct_character_track(segments, sr, min_gap=1.0)
# Seg1: 1.0-2.0
# Gap: 2.0-3.0
# Seg2: 3.0-5.0
print(f"Track with gap length: {len(track)/sr}s (Expected: 5.0s)")
print(f"Value at 2.5s: {track[2500]} (Expected: 0.0)")
print(f"Value at 3.5s: {track[3500]} (Expected: 2.0)")
