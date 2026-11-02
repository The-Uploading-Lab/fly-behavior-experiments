---
id: 84
date: 2026-09-11
title: One cord cell at half output stands more and walks slower
tldr: Lane B's nominee halves the output of one ventral cord cell type, INXXX466, and stands on 22 of the 24 title seeds where the champion stands on 19. The card holds it: its speed falls to 1.70 mm/s, under the animal's 2.0 mm/s floor, while the champion walks 2.26. No conflict closes. wi1 keeps the title, and the pattern from yesterday's candidate repeats: standing is bought with slower walking.
era: standard
video_receipt: publication-receipts/walker_b-e2h_30533_body.sha256-1ed76341b3bd4fea380fa27c1d39fb18f4d8a11caa2bc5cb35b125475f3c6478.publication.json,publication-receipts/b-e2h-cmd110-81547.sha256-f314b0167556fe932f89ae321cf6d49108b6a9ae419917c8d5e76306ca940393.publication.json,lanes/B/2026-09-11-e2-owned-pair/publication-receipts/b-e2-half-output-vs-wi1-81548.sha256-ce55007c28245a8d3508579e4b2d8355d9d9aa398f9a8da3550bf92031cc6edb.publication.json
commit: the commit this entry lands in, on main; nomination registered at 6b7f073ef
theta: lanes/judge/2026-09-11-b-e2h-theta.json (sha256 144754a8a7d2291666b3abc6ad0051c48849eac4dc27452699c30522a8289b68), wi1 plus type_out_gain INXXX466 0.5; holder results-wi1-theta.json (sha256 ba18676b65b3db0747532d344cd2a2847ac8e38335bb77d3c34b29e9b7802fe7)
seeds: paired batteries 30532-30555, card 30556-30567; no fresh seed
n: 24 paired battery seeds, 12 card seeds per arm
wall: batteries and card about 11 min with three workers on the deterministic Metal engine; holder re-card about 1 min; static T4 read 18 s
reproduce: PYTHONPATH=. .venv/bin/python3 lanes/judge/title_match_v311.py --dry-run --id b-e2h --theta lanes/judge/2026-09-11-b-e2h-theta.json --holder-receipts lanes/judge/2026-09-07-title-v36-wi1r-battery1.json lanes/judge/2026-09-07-title-v36-wi1r-battery2.json --holder-card lanes/judge/2026-09-11-card-wi1r-metal.json --freedom-manifest lanes/judge/2026-09-11-b-e2h-freedom-manifest.json --static-receipts lanes/judge/2026-09-11-static-b-e2h-wi1r-t4-sign.json --battery-seeds 30532-30543 30544-30555 --card-seeds 30556-30567 --date 2026-09-11
---

**Question.** Lane B's morning work on the walking pacemaker cells found
that silencing one cord cell type, INXXX466, cuts the headless body's travel
while its legs keep lifting, and that with the brain attached the same cell
matters. B then halved the cell type's output instead of silencing it and
saw the intact champion stand on all 24 of its lane seeds where wi1 stands on
18. Does that fly beat the champion on the title block?

**Predicted.** B claimed no gain and predicted nothing; B's own owned pair
came in before the match at 1 of 2 standing against wi1's 2 of 2, with travel
halved on the seed where both stood. The judge's expectation, recorded in the
0853 CURRENT before the nominee arrived, was that a gain arm is a candidate
only on the bar's fields and the card rows, and that travel orders nothing.

**Method.** The v3.11 title match as for every nominee: the nominee's theta
is wi1's 56 settings plus one, the outgoing gain of INXXX466 at 0.5, run on
the deterministic Metal engine with exact joint integration, three workers,
on the judge's paired block 30532-30555 against the holder's receipts of
record, and on the card seeds 30556-30567 against a fresh holder card taken
today with the current instrument (bit-equal on every field it shares with
the 2026-09-07 card). Conflict rows, card veto, floor veto, the v3.10
unreadable-row exclusion, the static T4 sign receipt and the successor
freedom gate all ran; the verdict file names each.

**Result.** Both batteries pass at 11 of 12. Standing 22 against 19; the
nominee stands on five seeds where wi1 falls (30533, 30547, 30550, 30552,
30554) and falls on four where wi1 stands (30540, 30548, 30549, 30555).
Qualifying 19 against 17. The card holds it: speed in band on 3 of 11 runs
against 9 of 12, median 1.70 mm/s under the 2.0 floor against 2.26; no row
improves. Open conflicts 8 against 8 after the exclusions; the three the
champion carries (height 1.18 vs 0.80-0.89 mm, straightness 0.25 vs 0.50-0.98,
vertical bob 0.64 vs 0-0.10) stay open on both. HELD BY THE CARD; wi1 keeps.

**Reading.** A quieter INXXX466 gives a fly that falls less and goes slower;
yesterday's candidate, the longer adaptation constant, gave a fly that fell
less and stepped slower and shorter. Two one-key changes in two days buy
standing with speed. The animal does both at once, so the routes that spend
speed are not the routes to the title; a body that stands more at the same
speed has not been found yet. Lane B's cadence question stays open: the
nominee's cadence reads 5.41 Hz, in band, on 9 of 11 card runs, as wi1's
does on 10 of 12.

**Second step, 2026-09-11 14:15 EEST.** Lane B put the walking command up by a
tenth on top of the half-output cord cell (packet 1350; matched theta
`lanes/judge/2026-09-11-b-e2h-cmd110-theta.json`, sha256 87c541f9; B's own file
4e542f1f). Same instrument, same seeds, three Metal workers. Both batteries
pass; standing 20 against 19, qualifying 20 against 17, the best count any
challenger has reached. It stands where wi1 falls on 30539, 30542, 30547, 30550,
30552 and 30554, and falls where wi1 stands on 30540, 30553 and 30555. The
card holds it: speed in band on 3 of 9 runs against 9 of 12, median 1.99 mm/s
against the 2.0 floor (wi1 2.26); front FTi range in band on 1 of 9 against 5
of 12, median 78.3 against the 81.2 degree floor (wi1 80.7); and it stands on
9 of the 12 card seeds where wi1 stands on all 12. No row improves; open
conflicts 8 against 8; the champion's three stay open on both arms. HELD BY
THE CARD; wi1 keeps. Reading: the command dial gives back most of the speed
the cord cut cost, 1.70 to 1.99 mm/s, and keeps the standing gain on the
block, at the price of three card falls and a shorter front-leg swing; the
dial sits one hundredth of a millimetre per second from the band. Verdict of
record `lanes/judge/2026-09-11-title-v311-b-e2h-cmd110-verdict-DRYRUN.json`.
Clip: lane B's seed 81547 on this fly, upright six seconds, 11.8 mm, receipt
`publication-receipts/b-e2h-cmd110-81547.sha256-f314b0167556fe932f89ae321cf6d49108b6a9ae419917c8d5e76306ca940393.publication.json`
(Telegram 6423).
Lane B's paired clip from the morning packet (0902) shows the other side of the
same fly: wi1 upright on the left while the half-output fly overturns on the
right, seed 81548; receipt
`lanes/B/2026-09-11-e2-owned-pair/publication-receipts/b-e2-half-output-vs-wi1-81548.sha256-ce55007c28245a8d3508579e4b2d8355d9d9aa398f9a8da3550bf92031cc6edb.publication.json`.
