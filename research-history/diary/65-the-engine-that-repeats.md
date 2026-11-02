---
id: 65
date: 2026-09-07
title: The engine that repeats
tldr: The GPU now gives the same spike vector twice. Yesterday's titles were not decided by thread scheduling (0 of 24 standing or qualifying flips on the old engine). The champion is still wg1. Stride now has a measured band and the champion sits inside it.
era: standard
commit: the commit this entry lands in, on main
seeds: lane D flip test 30448-30471 x 3 repeats on the atomic engine; compute remake of the same 24 on the new engine; no new judge seed this morning
---

08:25 EEST. The Metal engine on main is deterministic. NumPy and Metal agree
on the default net (271,963 spikes) and on the champion mechanism set (19,827
spikes), first differing 1 ms bin none.

Lane D registered the flip test before it ran. Three repeats of wg1's 24 title
seeds, three workers wide, on the old engine: no run changed standing or
qualifying state. Yesterday's titles stand.

The holder pair for the next match is remade on the new engine. Same theta,
same 24 seeds: standing 24 of 24, qualifying 24 of 24 (was 23 of 24). The
open-conflict count moving 6 to 5 is one twelve-seed coin on the front knee,
whose median sits 0.08 degrees from the band edge. That is not a closed
conflict.

`gait.stride_length` now has a band: 0.1013 to 0.8672 mm at 1.5 to 3.0 mm/s
forward, from DeAngelis 2019, 1,406 steps in 562 flies. wg1 reads 0.402. The
15:00 claim that the champion's stride is short by 4.3x compared a 2 mm/s
body to a constant measured at 18 to 25 mm/s, and is withdrawn.

Human: Robin, 21:52, take the engine after measuring the flips, merge at the
08:00 boundary. Agent: compute seat built and gated the engine; lane D ran
the registered flip test; the judge merged and adopted the remade pair.
