"""Continue native jump-muscle force after takeoff using the existing twitch.

The force scale101uN, side-specific moment arms, twitch and upper joint stop
match jumpfly_cns.JumpFlyBody.advance_to. After takeoff the twitch is evaluated
at each physics step; the existing flight pose controller continues to act.
"""


def apply_after_takeoff(body, model, data, receipt):
    if model is not body.h['model'] or data is not body.h['data'] or body.t_off is None:
        return
    twitches = body.twitch_now(float(data.time))
    active = False
    for side, joint in body.h['ctr'].items():
        torque = 0.
        if twitches[side] > 0. and float(data.qpos[joint['qadr']]) < body.stop:
            torque = 101. * twitches[side] * body.lever[side]
            active = True
        data.qfrc_applied[joint['dofadr']] = torque
        receipt['peak_torque_by_side'][side] = max(receipt['peak_torque_by_side'][side], abs(float(torque)))
    receipt['post_takeoff_physics_steps'] += 1
    if active:
        receipt['active_twitch_physics_steps'] += 1
        if receipt['first_active_time_ms'] is None:
            receipt['first_active_time_ms'] = float(data.time * 1000.)
        receipt['last_active_time_ms'] = float(data.time * 1000.)
