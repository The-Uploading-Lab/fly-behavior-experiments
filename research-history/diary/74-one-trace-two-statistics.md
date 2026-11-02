---
id: 74
date: 2026-09-08
title: One trace, two response statistics
tldr: A native 0.1 ms replay recorded conductance, current and voltage at four DNa01/DNa02 cells. The first read compared a side-normalized spike response with a raw current difference and placed an inversion inside DNa02. Applying either statistic to both quantities gives matching signs. The inversion is withdrawn; the dynamic trace and exact replay parity remain.
era: standard
commit: the commit this entry lands in, on main
seeds: spent seed 50614; no fresh seed
budget_exception: Fourth entry because C-N153's committed answer mixed two response statistics and placed a sign inversion inside DNa02; the correction must preserve that sequence rather than rewrite an earlier diary entry.
media_note: No figure or clip; this is a deterministic trace-statistic correction with no model or visible-body change.
---

04:50 EEST.

## Question

Does the wrong-sign visual response already exist in the synaptic current into
DNa01/DNa02, or does the target cell invert it through threshold, reset or
refractory dynamics?

## Method

The two C-N151 visual arms were replayed once on spent seed 50614 with the whole
188,508-neuron female CNS. A default-off recorder retained the four target
cells at every 0.1 ms tick from 500 through 5,500 ms: target voltage, decayed
excitatory and inhibitory conductance, spike and graded increments, signed
engine current, and separate direct-visual and other-network routes. All nine
named 5 ms population traces reproduce the earlier run exactly.

The raw artifact is 24,929,325 bytes with SHA-256
`cd8969869ae4a376895c21d761b314a11a724d0d2cc83674489320cc4f4ebdd8`.
The Metal source stamp is `48ebcded0df39751`; the dylib SHA-256 is
`c40c5e1ab0f16eb13ebd1bfa6a6e69363542c313d353dfafa26e42cc0fc55ce9`.

## Result

The first read mixed two calculations. C-N151 normalized the arm difference
separately for each target side before subtracting right from left. C-N153 used
that spike sign but classified current from a raw left-minus-right double
difference.

With one calculation applied to both quantities:

- DNa01 raw spikes/current are -41 and -17,380.423; side-normalized values are
  -0.270457 and -0.428943.
- DNa02 raw spikes/current are +8 and +10,454.842; side-normalized values are
  -0.022909 and -0.101953.

The signs agree under both matched calculations. DNa01's raw response is
negative in direct visual input (-1,580.749) and all other network sources
(-15,799.678). DNa02 receives opposing raw components: +38,416.709 from the
direct visual route and -27,961.872 from all other network sources.

## Interpretation

The data do not place a sign inversion in DNa02's threshold, reset or refractory
dynamics. That inference is withdrawn. The native arrays, conductance
reconstruction, exact 5 ms parity and DNa01 upstream-negative result remain.

The result covers four cells in one female CNS on one spent seed. It does not
select a neuron model or establish either cell type's population behavior. A
later temporal test must state one response statistic before inspecting current
or spikes.

## Source closure in the same pass

The compiled-body audit now discovers fifteen consumers on integrated main.
The shared selector guards every live whole-position reader, including CD2-GZ
and Lane D R54. Corrected historical body-position counts remain 80,784,
79,764 and 80,580, and the independent qpos and foot-tip evidence remains.

## Attribution

Lane C built the native recorder, ran the exact replay and caught the statistic
mismatch before the judge verdict. Lane A repaired the CD2-GZ pose reader. Lane
E found the fifteenth consumer and closed the integrated audit. The judge
repeated the focused tests and retained the narrower result.
