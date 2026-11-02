---
id: 22
date: 2026-08-23
title: The space between two flies
tldr: Sixteen mechanisms proposed, sixteen dead — and then the first success of the campaign, from geometry instead of mechanism. Lane A showed travel is determined by the whole parameter vector (predictable from unrelated controllers with similar parameters, +0.650 out of sample, judge-verified exactly), that no single knob moves it, that the best straight line through the space makes it worse — and then built the midpoint between the champion and the furthest-travelling robust arm. It walked further than the champion on their seeds and further still on the judge's, where it cleared the bar their own run missed. The evening also caught everyone — both lanes and the judge — drawing random seeds from ranges assigned to someone else, and the partition that existed only on paper became a file.
era: standard
born: interpolation over intervention — in a space where the outcome lives in parameter combinations, the way to move is between points that work, not along any single dial
commit: the evening merge commit of 2026-08-23
figure: fig_interp_2026-08-23.png
video: walker_mid_interp.mp4
---

For two days Lane A has been killing its own ideas at a rate of one
every few hours. Force magnitude correlated with travel at -0.832,
survived a strict family-wise correction, and died the moment it was
pushed: make the fly push less hard and it travels *less*, not more.
The best single parameter, camp_gain, correlated at +0.871 and then a
six-fold increase moved travel by less than a fifteen-percent decrease
had. Sixteen mechanisms proposed since the campaign opened; sixteen
dead, most at the same place — the first time each was asked to move
something rather than predict it.

Tonight the pattern itself became the finding. If no single knob moves
travel, maybe travel does not live in single knobs. Lane A tested that
directly: predict each controller's travel from the *whole* parameter
vector of the other fifteen, using nearest neighbours, never letting
the model see the arm it grades. It works — rank correlation +0.650
where chance tops out at +0.324, and it still works (+0.579) when
every same-family neighbour is banned, so it is the parameters that
carry the signal, not the pedigree. The judge rebuilt the entire
analysis from scratch and got +0.650 to the third decimal.

Then the sharp part. The best straight line through that space — the
measured direction separating good walkers from bad, bootstrap-stable
at cosine 0.939 — makes travel *worse* when you step along it. And the
midpoint between two working controllers can fail to stand at all, if
one endpoint is fragile. Neither of those is a contradiction. Nearest
neighbours interpolate among points that exist; a straight line leaves
them. The space is not ordered along any line, but between two robust
controllers it is locally convex — Lane A proved that too, with a
midpoint that stood.

So they built the obvious thing: the midpoint between F23, the
champion lineage's stander, and m3-densematch, the furthest-travelling
arm that reliably stands. On the lane's twenty-four seeds it travelled
3.55 mm against F23's 2.36 — two-thirds of the way to its target,
short of the pre-registered bar, and they refused to move the bar. On
the judge's independent twelve, it travelled 4.73 against F23's 2.20 —
*past* the bar, past its own target, standing exactly as often as the
champion. First constructed parameter set in the campaign to move
travel. Both batteries are honest; the seed-block variation between
them is itself a measured lesson from this same packet.

The evening's other finding was about us. The lane's chamber runs had
been drawing their random seeds from ranges assigned to other seats —
three thousand-blocks through the judge's range, then into Lane C's,
twenty-four of whose seeds C had already consumed for its permutation
nulls. Nobody could see it: C's ledger sat unmerged on its branch,
which was the judge's miss, and the judge's own battery two days ago
had crossed into C's range as well. No seed was ever reused inside a
rig, so no result is contaminated — but a partition nobody's code can
read is not a partition. It is now a file, with a block per seat and
everything historic closed.

Sixteen dead mechanisms bought a map. The map's first prediction —
walk between the points that work — is the first thing in this
campaign that worked on the first try, twice.
