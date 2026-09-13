"""Body-part layout of the Zenodo walking dataset, INFERRED from the data.

The dataset ships no column names. This layout was recovered on 2026-08-05 by
reading per-part mean position and per-part motion out of a 300-sequence sample
(see inspect_zenodo.py). It is a hypothesis supported by four independent
regularities, all of which held:

1. Midline sits at x ~= 68. Parts 0-11 are all right of it, 12-23 all left.
2. Within each block of four, mean frame-to-frame motion rises monotonically
   (~0.42 -> ~1.7). That is proximal-to-distal along a limb.
3. The three blocks per side point forward (y falls), outward (x runs away
   from midline), and backward (y rises). Front, middle, hind leg.
4. The last eight parts form four symmetric pairs plus two midline points,
   matching the stated "head features, thorax, abdomen, wings".

Frame convention, read off the means: y is the body axis with LOW y anterior
(head parts at y~30) and HIGH y posterior (abdomen at y~96). x is left-right.
Units are cropped-image pixels, NOT millimetres.

If any downstream result looks strange, re-check this file first. It is the
only guessed thing in the pipeline.
"""

MIDLINE_X = 68.0

# Six legs, each proximal -> distal (4 points).
LEGS = {
    'R1': [0, 1, 2, 3],      # right front
    'R2': [4, 5, 6, 7],      # right middle
    'R3': [8, 9, 10, 11],    # right hind
    'L1': [12, 13, 14, 15],  # left front
    'L2': [16, 17, 18, 19],  # left middle
    'L3': [20, 21, 22, 23],  # left hind
}

LEG_TIPS = {name: idx[-1] for name, idx in LEGS.items()}

# Non-leg landmarks. Pair assignments are by symmetry about the midline;
# which pair is antenna vs eye is not resolvable from geometry alone, so they
# are named by position only.
HEAD_PAIR_ANTERIOR = (24, 25)    # y ~= 30
HEAD_PAIR_POSTERIOR = (26, 27)   # y ~= 38
WING_TIPS = (28, 29)             # y ~= 117, most posterior
THORAX = 30                      # midline, y ~= 60
ABDOMEN = 31                     # midline, y ~= 96

# The two tripod groups. In tetrapod/tripod walking, these two sets alternate.
TRIPOD_A = ['R1', 'L2', 'R3']
TRIPOD_B = ['L1', 'R2', 'L3']

FPS = 80.0
N_FRAMES = 234
N_PARTS = 32
