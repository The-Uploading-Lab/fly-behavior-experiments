"""The missing layer: joint drive -> joint angles -> proprioceptor firing.

This is what closes the sensorimotor loop. Until now the proprioceptors have
been driven by unstructured Poisson noise, because nothing converted motor
output back into sensory input. In a walking fly the leg's own movement is what
drives the proprioceptors, and three regimes have now been searched for a motor
rhythm without one -- so the open loop is the leading explanation.

⚠️ READ MUSCLE.md. It exists because this layer is the only component that is
both fitted AND in the causal path, so freedom here does not explore a
hypothesis, it launders a bad result into a good score.

THE DISCIPLINE APPLIED HERE: NOTHING IS FITTED.

Every constant below is a measured anchor, a published encoding, or an
explicitly-swept unknown. There is no optimiser anywhere in this file. That is
deliberate -- if closing the loop with zero fitted parameters produces a
rhythm, nothing was laundered to get it, and MUSCLE.md's alarm (does the
real-vs-shuffled gap shrink once muscles are added?) cannot fire because there
is no fit to shrink it.

    quantity                what it is                           source
    tau_act = 12.3 ms       muscle activation, first order       Azevedo 2020:
                            (8.5 ms to half max => tau/ln2)      eLife 56754
    joint working ranges    per-DOF min/max over a step cycle    flygym's own
                                                                 CPG walker
    FeCO encoding           claw = angle, hook = signed          Mamiya, Gurung
                            velocity, club = |velocity|          & Tuthill 2018
    per-MN gain             ALL EQUAL                            not fitted; the
                                                                 391-parameter
                                                                 block of
                                                                 MUSCLE.md set
                                                                 to a constant
    tau_joint, drive_scale  SWEPT, never fitted                  see below

⚠️ THE JOINT IS FIRST-ORDER BY CONSTRUCTION, AND THAT IS A CONTROL, NOT A
SHORTCUT. A second-order joint (mass-spring-damper) can RESONATE, and a
resonant limb would produce a rhythm with no connectome involvement at all --
we would be measuring the body and calling it a brain. Overdamped first-order
dynamics cannot oscillate on their own, so any rhythm that appears has to come
from the loop. Physically this is the overdamped limit of

    b dtheta/dt = torque_muscle(a) - k (theta - theta_0)

rewritten as a relaxation toward an activation-dependent equilibrium with time
constant tau_joint = b/k.

⚠️ THIS IS NOT NeuroMechFly. It is 42 independent joints with no inertia, no
contact, no gravity and no body. It is enough to ask whether proprioceptive
feedback organises a rhythm, because that question is about loop TIMING. It is
NOT enough for a behavioural score -- that has to come from the real body
model, and any number from here must never be fed to flyscore/.

TWO PARAMETERS ARE SWEPT RATHER THAN SET, and for a reason. `tau_joint` is the
loop's dominant lag, and delay_law.py already established that the lag is what
sets a feedback loop's frequency -- so fixing it would be assuming the answer.
`drive_scale` is the loop gain, which sets whether the loop oscillates at all.
Sweeping them and reporting the relationship is the experiment; fitting them
to produce 14 Hz would be circular.
"""
import json

import numpy as np

import motormap as MM

TAU_ACT_MS = 8.5 / np.log(2.0)      # 12.26 ms, from Azevedo's 8.5 ms half-max

# Which DOF each leg's femoral chordotonal organ reports. NOT a guess: the
# FeCO sits in the femur and monitors the femur-tibia joint. Anatomy, not fit.
FECO_DOF = "FTi_pitch"
LEG_CODES = ["lf", "lm", "lh", "rf", "rm", "rh"]
# BANC's body_part_sensory / nerve -> our leg code
NERVE_LEG = {("left", "prothoracic"): "lf", ("left", "mesothoracic"): "lm",
             ("left", "metathoracic"): "lh", ("right", "prothoracic"): "rf",
             ("right", "mesothoracic"): "rm", ("right", "metathoracic"): "rh"}


def load_ranges(path="data/dof_ranges.json"):
    d = json.load(open(path))
    lo = np.array([d[n][0] for n in MM.DOF_NAMES])
    hi = np.array([d[n][1] for n in MM.DOF_NAMES])
    return lo, hi


class Body:
    """42 independent overdamped joints driven by first-order muscle activation.

    step() advances by dt_ms and returns (theta, vel) in radians and rad/s.
    Normalised position is 0 at the low end of the DOF's working range and 1 at
    the high end, which is what the proprioceptor encodings consume.
    """

    def __init__(self, tau_joint_ms=15.0, drive_scale=200.0,
                 tau_act_ms=TAU_ACT_MS, ranges=None):
        self.lo, self.hi = ranges if ranges is not None else load_ranges()
        self.mid = 0.5 * (self.lo + self.hi)
        self.half = 0.5 * (self.hi - self.lo)
        self.tau_act = float(tau_act_ms)
        self.tau_joint = float(tau_joint_ms)
        self.drive_scale = float(drive_scale)
        self.n = len(self.lo)
        self.reset()

    def reset(self):
        self.a = np.zeros(self.n)          # muscle activation, signed
        self.theta = self.mid.copy()       # joint angle, rad
        self.vel = np.zeros(self.n)        # rad/s
        # two-channel state (only advanced by step2; zero otherwise)
        self.a_flex = np.zeros(self.n)     # flexor-side activation, >= 0
        self.a_ext = np.zeros(self.n)      # extensor-side activation, >= 0
        self.cocon = np.zeros(self.n)      # a_flex + a_ext, the stiffness drive

    def step(self, drive, dt_ms):
        """drive: signed 42-vector of motor-neuron activity (spikes/bin)."""
        self.a += (dt_ms / self.tau_act) * (np.asarray(drive, float) - self.a)
        # bounded by construction: tanh keeps the joint inside its working
        # range whatever the drive does, which is what a joint stop does.
        theta_eq = self.mid + self.half * np.tanh(self.a / self.drive_scale)
        d_theta = (dt_ms / self.tau_joint) * (theta_eq - self.theta)
        self.theta += d_theta
        self.vel = d_theta / (dt_ms / 1000.0)
        return self.theta, self.vel

    def step2(self, drive_flex, drive_ext, dt_ms):
        """Two-channel step (see MUSCLE-SPLIT.md): antagonist activations kept
        separately so their SUM (co-contraction) exists as a signal. The theta
        path is the same physics as step() -- the activation low-pass is
        linear, so lowpass(ext) - lowpass(flex) equals lowpass(ext - flex) up
        to float association -- the split's only NEW quantity is `cocon`.
        Nothing here maps cocon to stiffness; that lives in muscle_split.py.
        """
        drive_flex = np.asarray(drive_flex, float)
        drive_ext = np.asarray(drive_ext, float)
        self.a_flex += (dt_ms / self.tau_act) * (drive_flex - self.a_flex)
        self.a_ext += (dt_ms / self.tau_act) * (drive_ext - self.a_ext)
        self.a = self.a_ext - self.a_flex   # downstream consumers (prop.rates)
        self.cocon = self.a_ext + self.a_flex
        theta_eq = self.mid + self.half * np.tanh(self.a / self.drive_scale)
        d_theta = (dt_ms / self.tau_joint) * (theta_eq - self.theta)
        self.theta += d_theta
        self.vel = d_theta / (dt_ms / 1000.0)
        return self.theta, self.vel

    def normalised(self):
        return np.clip((self.theta - self.lo) / (2 * self.half + 1e-12), 0, 1)


class Proprioceptors:
    """Joint state -> firing rate, one rate per BANC proprioceptor.

    ENCODING, from Mamiya, Gurung & Tuthill 2018 (Neuron 100:636) which
    characterised the three femoral chordotonal organ subtypes that BANC's
    `cell_sub_class` column names directly:

      claw  tonic, POSITION tuned. Two populations exist, flexion-tuned and
            extension-tuned, so half are given each sign.
      hook  phasic and DIRECTIONALLY SELECTIVE -- separate flexion and
            extension populations, each firing only for movement one way.
      club  phasic, bidirectional, non-directional: responds to movement and
            vibration regardless of sign. Encodes |velocity|.

    Plus two classes outside the FeCO:

      hair plate           fires near the joint's extremes
      campaniform sensilla cuticular strain, i.e. LOAD. Encoded from muscle
                           activation magnitude, which is the only force-like
                           quantity this reduced body has.

    ⚠️ WHAT IS ASSUMED RATHER THAN KNOWN. BANC names the subtype but not which
    member of a subtype is flexion- vs extension-tuned, so that split is made
    by index parity -- a hypothesised assignment, flagged here, and one a
    search could later replace. Hair plates sit at several joints and BANC does
    not say which, so they are attached to the same leg's ThC. Neither is a
    measurement and both are recorded in the returned `assumptions` dict.
    """

    R_MAX = 120.0        # Hz at full drive; fly leg afferents reach ~100-200
    R_REST = 5.0         # Hz baseline
    V_SCALE = 8.0        # rad/s giving a full phasic response

    def __init__(self, meta, net, rng=None, split_seed=None,
                 claw_polarity=None, hook_polarity=None):
        # POLARITY OVERRIDES. claw/hook_polarity = +1 assigns that whole
        # population flexion-tuned, -1 extension-tuned; None keeps the split.
        #
        # These exist because sign_test.py measured that the split decides the
        # reflex sign entirely -- 0 of 6 joints kept their sign across four
        # different random splits -- so the assignment is an unidentified
        # parameter and the parameter rule says search it rather than assume it.
        #
        # And control theory says what to look for. Claw encodes POSITION and
        # hook encodes VELOCITY, and in a delayed feedback loop those do
        # opposite things: negative position feedback is what makes a loop
        # oscillate, while negative velocity feedback DAMPS it and positive
        # velocity feedback anti-damps. So the interesting corner is negative
        # on claw and positive on hook, and that is a prediction rather than a
        # sweep.
        # split_seed selects WHICH members of claw/hook are flexion- vs
        # extension-tuned. None reproduces the original parity split exactly.
        # This is a parameter and not a constant because it is an ASSUMPTION:
        # BANC names the subtype but not the tuning, and if the reflex signs
        # depend on it then they are a property of this guess rather than of
        # the fly. sign_test.py measures exactly that.
        rng = rng or np.random.default_rng(0)
        self.split_seed = split_seed
        srng = np.random.default_rng(split_seed) if split_seed is not None \
            else None
        self.dof_index = {n: i for i, n in enumerate(MM.DOF_NAMES)}
        sub = meta["cell_sub_class"].fillna("").to_numpy()
        cls = meta["cell_class"].fillna("").to_numpy()
        nerve = meta["nerve"].fillna("").to_numpy()
        side = meta["side"].fillna("").to_numpy()

        groups = {k: [] for k in ("claw_flex", "claw_ext", "hook_flex",
                                  "hook_ext", "club", "hair", "campaniform")}
        self.dof_of = {k: [] for k in groups}
        n_unassigned = 0
        # LEG proprioceptors only. The first version selected on cell_class
        # alone and picked up chordotonal, hair-plate and campaniform afferents
        # from the wing, haltere, neck and abdomen -- 583 "club" neurons where
        # the legs have 342, and 571 campaniform where the legs have 61. Those
        # would have been driven by a leg joint's angle, which is simply wrong.
        # body_part_sensory is the selector the rest of the repo uses.
        part = meta["body_part_sensory"].fillna("").to_numpy()
        is_leg = np.isin(part, ["front_leg", "middle_leg", "hind_leg"])
        for row in np.flatnonzero(is_leg & np.isin(cls, [
                "chordotonal_organ_neuron", "hair_plate_neuron",
                "campaniform_sensillum_neuron"])):
            nv = nerve[row]
            leg = None
            for (sd, seg), code in NERVE_LEG.items():
                if nv.startswith(sd) and seg in nv:
                    leg = code
                    break
            if leg is None:
                n_unassigned += 1
                continue
            s = sub[row]
            if "claw" in s:
                if claw_polarity is not None:
                    flex = claw_polarity > 0
                else:
                    flex = (row % 2 == 0) if srng is None \
                        else bool(srng.random() < .5)
                g = "claw_flex" if flex else "claw_ext"
                dof = f"{leg}_{FECO_DOF}"
            elif "hook" in s:
                if hook_polarity is not None:
                    flex = hook_polarity > 0
                else:
                    flex = (row % 2 == 0) if srng is None \
                        else bool(srng.random() < .5)
                g = "hook_flex" if flex else "hook_ext"
                dof = f"{leg}_{FECO_DOF}"
            elif "club" in s or cls[row] == "chordotonal_organ_neuron":
                g = "club"
                dof = f"{leg}_{FECO_DOF}"
            elif cls[row] == "hair_plate_neuron":
                g = "hair"
                dof = f"{leg}_ThC_pitch"
            else:
                g = "campaniform"
                dof = f"{leg}_{FECO_DOF}"
            groups[g].append(row)
            self.dof_of[g].append(self.dof_index[dof])

        self.groups = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}
        self.dof_of = {k: np.array(v, dtype=np.int64)
                       for k, v in self.dof_of.items()}
        self.n_unassigned = n_unassigned
        self.all_rows = np.concatenate([v for v in self.groups.values()
                                        if len(v)])
        self.assumptions = {
            "flex_ext_split": "by row parity — BANC names the subtype but not "
                              "which members are flexion- vs extension-tuned",
            "hair_plate_dof": "attached to the same leg's ThC_pitch — BANC "
                              "does not say which joint each hair plate sits "
                              "at",
            "campaniform_from_activation": "load encoded from |muscle "
                                           "activation|; this reduced body "
                                           "has no contact forces",
            "unassigned_by_nerve": n_unassigned,
        }

    def counts(self):
        return {k: int(len(v)) for k, v in self.groups.items()}

    def rates(self, body):
        """-> (row_indices, rates_hz) for every proprioceptor."""
        pos = body.normalised()
        vel = body.vel
        act = np.abs(body.a)
        out_idx, out_rate = [], []

        def emit(key, r):
            if len(self.groups[key]):
                out_idx.append(self.groups[key])
                out_rate.append(np.clip(r, 0.0, self.R_MAX))

        d = self.dof_of
        emit("claw_flex", self.R_REST + self.R_MAX * pos[d["claw_flex"]])
        emit("claw_ext", self.R_REST + self.R_MAX * (1 - pos[d["claw_ext"]]))
        emit("hook_flex", self.R_REST + self.R_MAX
             * np.maximum(0.0, vel[d["hook_flex"]]) / self.V_SCALE)
        emit("hook_ext", self.R_REST + self.R_MAX
             * np.maximum(0.0, -vel[d["hook_ext"]]) / self.V_SCALE)
        emit("club", self.R_REST + self.R_MAX
             * np.abs(vel[d["club"]]) / self.V_SCALE)
        # hair plates: silent in mid-range, firing as the joint nears a limit
        p = pos[d["hair"]]
        emit("hair", self.R_REST + self.R_MAX
             * np.maximum(0.0, np.abs(2 * p - 1) - 0.7) / 0.3)
        emit("campaniform", self.R_REST + self.R_MAX
             * np.minimum(1.0, act[d["campaniform"]] / (2 * body.drive_scale)))
        return np.concatenate(out_idx), np.concatenate(out_rate)
