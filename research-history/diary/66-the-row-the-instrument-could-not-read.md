---
id: 66
date: 2026-09-07
title: The row the instrument could not read
tldr: A challenger was nominated on closing the front knee's range, a row whose median moves 3.35 degrees between repeat blocks of one unchanged body against a nearest-edge gap of 0.45 degrees. It was held on standing, 21 of 24 against 24 of 24, and the bar became v3.10: a graded row whose median sits within its own between-block spread of a band edge counts for neither arm's conflict tally. Re-run on the recorded matches the filter changes one verdict, and it changes it toward the challenger.
era: standard
commit: the commit this entry lands in, on main
seeds: wh1 batteries 30448-30471 paired against wg1's remade record on the same block; wh1 card 30508-30519 against wg1's card 30496-30507; the spread itself costs no seed, being read from six committed twelve-seed blocks of one theta
---

11:08 EEST. **The champion is still wg1.** One challenger was matched and held.

## What was nominated, and what held it

wh1 is the champion with the front femur-tibia range corrected, lane A's
morning body. Its whole case was one registry row, `rom.front_fti_deg`, and it
does move that row: the card veto reads no regressions and one improvement,
which is the row it was built for.

It was held before any of that was weighed. Over the 24 paired seeds wh1 stands
21 times against the champion's 24, batteries of 10 and 11 of 12, and v3.9's
standing gate holds a challenger that stands three fewer. The three seeds it
loses (30449, 30456, 30463) are seeds where the champion stands. A fly that
tips over more often does not take the title for fixing one number, which is
the boundary Mistral asked for on 2026-09-06 and Robin ruled.

## The row it was nominated on cannot be read at twelve seeds

The champion now has **six independent twelve-seed blocks at one unchanged
theta** (sha256_16 `628b58cd965a4822`: two cards and four title batteries,
across the old and new engines). That is enough to ask, per row, how far the
median itself wanders between blocks of the same body.

    row                between-block median spread   distance to nearest band edge
    rom_fFTi_deg                        3.3504 deg                     0.4545 deg
    speed_mm_s                       0.5314 mm/s                    0.0591 mm/s
    contra_phase                           0.0374                         0.0321

Three rows sit closer to their band edge than one repeat of the same body moves
them. The front knee is the worst: seven times its own noise. Its six block
medians are 80.12, 78.64, 81.37, 81.99, 79.24 and 81.41 degrees, and the band
edge is 0.45 degrees from where they average. Whether that row reads open or
closed for a given body is settled by which twelve seeds were drawn.

The v3.7 card veto already knew this in one form: it grades a row only when the
in-band COUNT moves three of twelve, a margin confirmed on twelve separated
blocks. The conflict count under v3.9 had no equivalent. It reads a median
against a band edge and calls the row open or closed, with nothing between.

## v3.10

**A graded card row whose arm median sits within that row's measured
between-block median spread of its nearest band edge is unreadable for that
match, and counts for neither arm's conflict tally.** The spread is measured
from committed receipts at each arm's own theta and costs no seed. Where the
two arms disagree the larger spread is taken, and a row is excluded if EITHER
arm sits inside it, so one exclusion set covers both and the clause cannot
favour whichever body happens to have more blocks on disk.

Scope is the conflict count under ordering items 3, 6 and 7. The v3.7 count
veto, the v3.8 floor veto and the v3.9 standing gate are untouched: a row too
noisy to order a title is still a row a challenger can break, and the vetoes
grade movement rather than a band edge.

## The half of it that does not flatter the seat that wrote it

Run against the matches already on the record, no seed spent:

    match   under v3.9                                 under v3.10
    wg1     CROWNS, conflicts 6 against 6, count 23-22  CROWNS, conflicts 4 against 4
    wf1     HOLDER KEEPS, conflicts 7 against 6         CROWNS, conflicts 5 against 5, count 23-22

The current champion's coronation is reproduced. **wf1's is not**: it was held
because it carried one more open conflict, and the row that ordered against it
was `gait.speed`, one of the three that cannot be read at twelve seeds. Under
the filter the counts equalise and the qualifying count crowns it.

The first draft of this clause said it makes this class of nomination harder,
and stopped there. That is the direction that flatters a judge writing a rule
after holding a challenger. It also stops a near-edge row being used AGAINST a
challenger, which is worth more, and the probe is what showed it. Nothing past
is re-graded: wf1 is not installed, and the clause binds future matches only.

## What bit us

The first implementation computed the unreadable set per arm, each at its own
theta hash. The challenger got three exclusions and the champion one, purely
because the champion has six committed blocks and the nominee has two, so the
rule would have been most generous to whichever body was least measured. Caught
by printing both sets before grading them, which is the same defence that found
a clause written against a row the veto cannot see two days ago.

## Who did what

Lane A built and nominated wh1 on the row its own measurements said was open.
The judge ran the match, held it on standing, then measured the row's repeat
spread on the champion's own committed blocks and wrote the clause against its
own instrument rather than against the nominee. Robin's standing delegation of
2026-09-02 covers the bar; he may veto, and the clause binds until he does.
