"""Timed body interventions and read-only capture for study b309bcde.

Windows use capture labels, T = callback time + tick. No object is constructed
for an absent or empty window. Each write records both its old and new value.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np

DRIVE_LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
CAPTURE_LEGS = ("lf", "lh", "lm", "rf", "rh", "rm")


def leg_name(name):
    name = name.lower()
    return next((leg for leg in DRIVE_LEGS
                 if re.search(r"(?:^|[/_])" + leg + r"(?:_|/|$)", name)), None)


class WindowHook:
    def __init__(self, spec, *, tick, theta, body, simulation, pool_map,
                 split, playback):
        import body as B
        import motormap as MM

        self.spec = dict(spec)
        self.kind = spec["kind"]
        if self.kind not in {"adhesion_hold", "leg_drive", "pair_friction",
                             "abdomen_hold", "joint_range", "nudge"}:
            raise ValueError(f"unknown window kind: {self.kind}")
        if not split or playback is not None:
            raise ValueError("window replay requires the live split-muscle path")
        self.tick = float(tick)
        self.t0, self.t1 = float(spec["t0_ms"]), float(spec["t1_ms"])
        if not (self.tick <= self.t0 < self.t1):
            raise ValueError("window bounds must name positive capture labels")
        if any(not np.isclose(t / self.tick, round(t / self.tick), atol=1e-9,
                              rtol=0) for t in (self.t0, self.t1)):
            raise ValueError("window bounds must lie on the control grid")
        self.first = int(round(self.t0 / self.tick)) - 1
        self.stop = int(round(self.t1 / self.tick)) - 1
        self.index = -1
        self.active = False
        self.events = []
        self.restoration = None
        self.sim = simulation
        self.leg = spec.get("leg")
        self.drive_leg = DRIVE_LEGS.index(self.leg) if self.leg else None
        self.capture_leg = CAPTURE_LEGS.index(self.leg) if self.leg else None
        self.reference = None
        if spec.get("writeback"):
            ref = spec["writeback"]
            path = Path(ref["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                raise ValueError("write-back reference hash differs")
            with np.load(path, allow_pickle=False) as z:
                self.reference = {k: z[k] for k in z.files}
        self.table = spec.get("dof_table", [])
        self.pool_map_equal = None
        if self.kind == "leg_drive":
            if theta.get("adh") != "flex" or any(
                    theta.get(k) is not None for k in
                    ("adh_knee_legs", "adh_ctr_flex", "adh_motor_balance")):
                raise ValueError("arm B requires the unchanged flex-only gate")
            if spec.get("mode") not in {"activation", "drive"}:
                raise ValueError("leg_drive mode must be activation or drive")
            expected = [i for i, name in enumerate(MM.DOF_NAMES)
                        if name.startswith(self.leg + "_")]
            if [row["index"] for row in self.table] != expected:
                raise ValueError("leg_drive table must cover the named leg in drive order")
            if spec.get("channel_pool_map") != pool_map:
                raise ValueError("current channel_pool_map differs from the frozen table")
            for row in self.table:
                name, channel = row["dof"], row["channel"]
                if (MM.DOF_NAMES[row["index"]] != name or channel != "a_ext"
                        or pool_map.get(name, {}).get(channel) != row["pools"]
                        or not np.isfinite(row["value"])):
                    raise ValueError(f"invalid frozen channel: {row}")
            self.pool_map_equal = True
        if self.kind == "joint_range":
            self.original_ranges = {k: getattr(body, k).copy()
                                    for k in ("lo", "hi", "mid", "half")}
            lo, hi = B.measured_ranges(rom_override=spec.get("rom_override"))
            self.ranges = {"lo": lo, "hi": hi, "mid": (lo + hi) * .5,
                           "half": (hi - lo) * .5}
        if self.kind == "pair_friction":
            model = simulation.mj_model
            self.pairs = []
            self.pair_names = []
            for p in range(model.npair):
                names = [model.geom(int(ids[p])).name or ""
                         for ids in (model.pair_geom1, model.pair_geom2)]
                if any(leg_name(n) == self.leg and "tibia" not in n.lower()
                       and any(s in n.lower() for s in ("tarsus", "claw"))
                       for n in names):
                    self.pairs.append(p)
                    self.pair_names.append(names)
            if not self.pairs:
                raise ValueError("friction window matched no tarsal pairs")
            self.original_friction = model.pair_friction[self.pairs, :2].copy()
        if self.kind == "nudge":
            self.dof = MM.DOF_INDEX[spec["dof"]]
            if self.stop - self.first != 1:
                raise ValueError("a nudge must treat exactly one callback")
        if self.kind == "abdomen_hold":
            if spec["axis"] not in ("roll", "pitch"):
                raise ValueError("abdomen axis must be roll or pitch")
            self.abd_indices = np.arange(1 if spec["axis"] == "roll" else 0, 10, 2)

    def event(self, **fields):
        e = {"callback_ms": self.index * self.tick,
             "row_ms": (self.index + 1) * self.tick, **fields}
        self.events.append(e)
        return e

    def begin(self, t, body):
        self.index = int(round(float(t) / self.tick))
        self.active = self.first <= self.index < self.stop
        if self.kind == "joint_range":
            if self.index == self.first:
                for k, value in self.ranges.items():
                    setattr(body, k, value.copy())
            if self.active:
                if not all(np.array_equal(getattr(body, k), v)
                           for k, v in self.ranges.items()):
                    raise ValueError("joint range hold was not applied")
                self.event(ranges_held=True,
                           changed=any(not np.array_equal(self.original_ranges[k], v)
                                       for k, v in self.ranges.items()))
            elif self.index == self.stop:
                for k, value in self.original_ranges.items():
                    setattr(body, k, value.copy())
                self.restoration = {"row_ms": (self.index + 1) * self.tick,
                                    "exact": all(np.array_equal(getattr(body, k), v)
                                                 for k, v in self.original_ranges.items())}
        if self.kind == "pair_friction":
            if self.active:
                before = self.sim.mj_model.pair_friction[self.pairs, :2].copy()
                self.sim.mj_model.pair_friction[self.pairs, :2] = float(self.spec["mu"])
                self.event(before=before.tolist(), after=self.sim.mj_model.pair_friction[
                    self.pairs, :2].tolist())
            elif self.index == self.stop:
                self.sim.mj_model.pair_friction[self.pairs, :2] = self.original_friction
                self.restoration = {"row_ms": (self.index + 1) * self.tick,
                                    "exact": bool(np.array_equal(self.sim.mj_model.pair_friction[
                                        self.pairs, :2], self.original_friction))}

    def before_body(self, body, df, de):
        if not self.active or self.kind != "leg_drive":
            return
        writes = []
        for row in self.table:
            j = row["index"]
            channel = row["channel"]
            activation = getattr(body, channel)
            drive = df if channel == "a_flex" else de
            k = 0 if channel == "a_flex" else 1
            old_a, old_d = float(activation[j]), float(drive[j])
            drive[j] = (self.reference["drive"][self.index, k, j]
                        if self.reference is not None else row["value"])
            if self.spec["mode"] == "activation":
                activation[j] = (self.reference[channel][self.index - 1, j]
                                 if self.reference is not None else row["value"])
            writes.append({"dof": row["dof"], "channel": channel,
                           "activation_before": old_a, "activation_after": float(activation[j]),
                           "drive_before": old_d, "drive_after": float(drive[j])})
        self.event(writes=writes)

    def after_body(self, body):
        if not self.active:
            return
        if self.kind == "leg_drive":
            ix = [row["index"] for row in self.table]
            eq = body.mid + body.half * np.tanh(body.a / body.drive_scale_vec)
            if body.joint_bias is not None:
                eq = np.clip(eq + body.joint_bias, body.lo, body.hi)
            self.events[-1].update(theta_eq_rad=eq[ix].tolist(),
                                   a_flex=body.a_flex[ix].tolist(),
                                   a_ext=body.a_ext[ix].tolist())
        if self.kind == "nudge":
            before = float(body.theta[self.dof])
            body.theta[self.dof] += np.radians(float(self.spec["deg"]))
            after = float(body.theta[self.dof])
            clipped = float(np.clip(after, body.lo[self.dof], body.hi[self.dof]))
            self.event(dof=self.spec["dof"], index=self.dof, before_rad=before,
                       after_rad=after, lo_rad=float(body.lo[self.dof]),
                       hi_rad=float(body.hi[self.dof]), clip_erased_write=False,
                       would_immediate_clip_erase=(clipped == before),
                       note="The registered write follows step2 clipping; no second clip is imposed.")

    def after_abdomen(self, adapter):
        if not self.active or self.kind != "abdomen_hold":
            return
        if adapter is None:
            raise ValueError("abdomen hold requires an abdomen adapter")
        ix = self.abd_indices
        before = adapter.bod.theta[ix].copy()
        values = (self.reference["abdomen_joint_rad"][self.index, ix]
                  if self.reference is not None else np.asarray(self.spec["values_rad"]))
        if np.shape(values) != (5,):
            raise ValueError("abdomen hold requires five angles")
        adapter.bod.theta[ix] = values
        adapter.bod.vel[ix] = 0.0
        self.event(indices=ix.tolist(), before_rad=before.tolist(),
                   after_rad=adapter.bod.theta[ix].tolist(), vel_after=[0.0] * 5)

    def adhesion(self, onoff):
        if not self.active or self.kind != "adhesion_hold":
            return
        before = bool(onoff[self.drive_leg])
        value = (self.reference["adhesion_onoff"][self.index, self.drive_leg]
                 if self.reference is not None else self.spec["state"])
        onoff[self.drive_leg] = bool(value)
        self.event(leg=self.leg, before=before, after=bool(onoff[self.drive_leg]))

    def receipt(self):
        expected = list(range(self.first, self.stop))
        observed = [int(round(e["callback_ms"] / self.tick)) for e in self.events]
        out = {"spec": self.spec, "drive_leg_index": self.drive_leg,
               "capture_leg_index": self.capture_leg, "events": self.events,
               "scheduled_writes_complete": observed == expected,
               "channel_pool_map_equal": self.pool_map_equal,
               "restoration": self.restoration}
        if self.kind == "joint_range":
            out.update(ranges_before={k: v.tolist() for k, v in self.original_ranges.items()},
                       ranges_after={k: v.tolist() for k, v in self.ranges.items()})
        if self.kind == "pair_friction":
            out.update(pair_indices=self.pairs, pair_names=self.pair_names,
                       friction_before=self.original_friction.tolist())
        return out


class WindowRecorder:
    """Append columns 20-23 without writing simulator state."""
    def __init__(self, sim, fly, abdomen_adapter):
        import abdomen as AB
        from flygym.compose import ActuatorType

        self.sim = sim
        model = sim.mj_model
        self.tarsal = {}
        self.geom_names = {}
        for i in range(model.ngeom):
            name = model.geom(i).name or ""
            leg = leg_name(name)
            if leg and "tibia" not in name.lower() and any(
                    s in name.lower() for s in ("tarsus", "claw")):
                self.tarsal[i] = CAPTURE_LEGS.index(leg)
                self.geom_names[str(i)] = name
        self.adhesion_ids = []
        for leg in CAPTURE_LEGS:
            ids = [i for i in range(model.nu) if "adhesion" in
                   (model.actuator(i).name or "").lower()
                   and leg_name(model.actuator(i).name or "") == leg]
            if len(ids) != 1:
                raise ValueError(f"expected one {leg} adhesion actuator, got {ids}")
            self.adhesion_ids.append(ids[0])
        self.abd_qadr = []
        self.abd_body_ids = []
        self.abd_body_names = []
        if abdomen_adapter is not None:
            order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
            logical = [f"{o.parent.link}-{o.child.link}-{o.axis.value}" for o in order[42:]]
            if logical[:len(AB.DOF_NAMES)] != AB.DOF_NAMES or logical[len(AB.DOF_NAMES):] not in (
                    [], AB.PROBOSCIS_DOF_NAMES):
                raise ValueError("abdomen actuator order differs from abdomen.DOF_NAMES")
            ids = [i for i in range(model.nu) if "abdomen" in
                   (model.actuator(i).name or "").lower()
                   and "adhesion" not in (model.actuator(i).name or "").lower()]
            if len(ids) != len(AB.DOF_NAMES):
                raise ValueError("missing achieved abdomen joint positions")
            self.abd_qadr = [int(model.jnt_qposadr[int(model.actuator_trnid[i, 0])]) for i in ids]
            for segment in AB.ABD_LINKS:
                ids = [i for i in range(model.nbody)
                       if (model.body(i).name or "").split("/")[-1] == "c_" + segment]
                if len(ids) != 1:
                    raise ValueError(f"missing abdomen body {segment}: {ids}")
                self.abd_body_ids.append(ids[0])
                self.abd_body_names.append(model.body(ids[0]).name)
        self.masses = model.body_mass[self.abd_body_ids].copy()
        self.metadata = {
            "adhesion_onoff_order": list(DRIVE_LEGS),
            "per_leg_capture_order": list(CAPTURE_LEGS),
            "adhesion_actuator_ids": self.adhesion_ids,
            "tarsal_geom_names": self.geom_names,
            "contact_definition": "a MuJoCo contact-list entry involving TARSUS or CLAW, excluding TIBIA",
            "tracked_geom": "largest summed nonnegative normal force, smallest geom id breaks ties; -1 when absent",
            "abdomen_dofs": AB.DOF_NAMES if self.abd_qadr else [],
            "abdomen_qpos_addresses": self.abd_qadr,
            "abdomen_body_names": self.abd_body_names,
            "abdomen_body_masses": self.masses.tolist(),
            "physics_nq": int(model.nq),
            "physics_joint_names": [(model.joint(i).name or "").split("/", 1)[-1]
                                    for i in range(model.njnt)],
            "physics_joint_qpos_addresses": model.jnt_qposadr.tolist(),
            "coordinate_convention": "world xyz in model millimetres; centre uses xipos and body_mass"}

    def read(self, applied_onoff):
        import mujoco

        model, data = self.sim.mj_model, self.sim.mj_data
        forces = {}
        buf = np.zeros(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            ids = [g for g in (int(contact.geom1), int(contact.geom2)) if g in self.tarsal]
            if not ids:
                continue
            mujoco.mj_contactForce(model, data, i, buf)
            for geom in ids:
                forces[geom] = forces.get(geom, 0.0) + max(0.0, float(buf[0]))
        flag = np.zeros(6, dtype=bool)
        geom_ids = np.full(6, -1, dtype=np.int32)
        xyz = np.full((6, 3), np.nan)
        normal = np.zeros(6)
        for leg in range(6):
            ids = [g for g in forces if self.tarsal[g] == leg]
            if ids:
                geom = min(ids, key=lambda g: (-forces[g], g))
                flag[leg], geom_ids[leg] = True, geom
                xyz[leg] = data.geom_xpos[geom]
                normal[leg] = sum(forces[g] for g in ids)
        achieved = {"joint_rad": data.qpos[self.abd_qadr].copy(),
                    "centre_xyz_mm": (np.average(data.xipos[self.abd_body_ids],
                                                 axis=0, weights=self.masses)
                                      if len(self.masses) else np.full(3, np.nan))}
        return (np.asarray(applied_onoff, dtype=bool).copy(),
                data.actuator_force[self.adhesion_ids].copy(),
                {"contact": flag, "geom_id": geom_ids, "xyz_mm": xyz, "normal_force": normal},
                achieved)
