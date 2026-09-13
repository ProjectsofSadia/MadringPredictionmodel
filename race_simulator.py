"""Simplified lap-by-lap Monte Carlo race simulator.

This is not a race-strategy simulator. It is a transparent stochastic model with
every assumption declared in src/simulation_assumptions.yaml.

Per simulation:
  1. each driver's underlying pace = model prediction + a bootstrapped out-of-fold
     residual, held constant for the race
  2. lap 1 reorders the grid using an empirical position-change distribution
  3. laps 2..N: base pace + lap-to-lap noise + estimated tyre-age trend
  4. a queue model holds a car behind a slower car unless it passes
  5. pit stops at pre-drawn laps with an estimated pit-lane time loss
  6. per-lap DNF hazard from team retirement rates
  7. Safety Car periods drawn from a 2026-season prior (no Madring-specific value)

Overtaking difficulty cannot be calibrated for Madring, so the whole simulation is
run under LOW / BASE / HIGH pass probability and the spread is reported.
"""

from __future__ import annotations

import numpy as np


def _choose_n_stops(n_laps: int, deg_s: float, pit_loss_s: float, max_stops: int = 3) -> int:
    best, best_cost = 1, np.inf
    for n in range(1, max_stops + 1):
        stint = n_laps / (n + 1)
        cost = n * pit_loss_s + deg_s * (n + 1) * stint * (stint - 1) / 2.0
        if cost < best_cost:
            best, best_cost = n, cost
    return best


def _draw_pit_laps(rng, n_sims, n_drivers, n_laps, n_stops, jitter):
    targets = [round(n_laps * (k + 1) / (n_stops + 1)) for k in range(n_stops)]
    laps = np.empty((n_sims, n_drivers, n_stops), dtype=np.int32)
    for k, t in enumerate(targets):
        laps[:, :, k] = np.clip(
            t + rng.integers(-jitter, jitter + 1, size=(n_sims, n_drivers)),
            3,
            n_laps - 2,
        )
    return np.sort(laps, axis=2)


def _draw_safety_car(rng, n_sims, n_laps, sc_prior):
    """Boolean (n_sims, n_laps+1) mask of laps run under Safety Car."""
    mask = np.zeros((n_sims, n_laps + 2), dtype=bool)
    p = sc_prior.get("p_race_has_sc") or 0.0
    if p <= 0:
        return mask
    periods = max(1, int(round(sc_prior.get("mean_periods_when_present") or 1)))
    share = sc_prior.get("mean_lap_share_when_present") or 0.05
    dur = max(1, int(round(share * n_laps / periods)))
    has = rng.random(n_sims) < p
    for _ in range(periods):
        start = rng.integers(3, max(4, n_laps - dur - 2), size=n_sims)
        for d in range(dur):
            lap = np.clip(start + d, 1, n_laps)
            mask[np.arange(n_sims), lap] |= has
    return mask


def simulate(
    *,
    pred_pace_pct: np.ndarray,
    residuals: np.ndarray,
    grid_position: np.ndarray,
    dnf_per_race: np.ndarray,
    ref_race_lap_s: float,
    n_laps: int,
    priors: dict,
    assumptions: dict,
    pass_prob: float,
    n_sims: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    D = len(pred_pace_pct)
    S = n_sims

    deg_pct = priors["degradation_pct_per_lap"]["pooled"]
    lap_noise = priors["lap_noise_s"]["value"]
    pit_loss = priors["pit_loss_s"]["value"]
    lap1_gap = priors["lap1_gap_s"]["value"]
    lap1_pool = np.asarray(priors["lap1_position_delta"]["samples"], dtype=float)
    for name, v in (
        ("degradation", deg_pct),
        ("lap noise", lap_noise),
        ("pit loss", pit_loss),
        ("lap-1 gap", lap1_gap),
    ):
        if v is None:
            raise RuntimeError(f"{name} prior could not be estimated from 2026 data - stopping")

    deg_s = deg_pct * ref_race_lap_s / 100.0
    follow_gap = assumptions["following_gap_s"]
    dirty_air = assumptions["dirty_air_penalty_s"]
    pace_threshold = assumptions["pass_pace_threshold_s"]
    pass_cost = assumptions["pass_time_cost_s"]
    sc_gap = assumptions["sc_gap_s"]
    sc_discount = assumptions["sc_pit_loss_discount"]

    # 1. underlying pace
    pace = pred_pace_pct[None, :] + rng.choice(residuals, size=(S, D), replace=True)
    base_lap = ref_race_lap_s * (1.0 + pace / 100.0)

    # 2. strategy
    n_stops = _choose_n_stops(n_laps, deg_s, pit_loss)
    pit_laps = _draw_pit_laps(rng, S, D, n_laps, n_stops, assumptions["pit_lap_jitter"])
    sc_mask = _draw_safety_car(rng, S, n_laps, priors["safety_car"])

    # 3. lap 1
    delta = rng.choice(lap1_pool, size=(S, D)) if lap1_pool.size else np.zeros((S, D))
    score = grid_position[None, :] + delta + rng.normal(0, 0.25, size=(S, D))
    order_rank = np.argsort(np.argsort(score, axis=1), axis=1)
    cum = base_lap + order_rank * lap1_gap
    tyre_age = np.ones((S, D), dtype=np.int32)
    alive = np.ones((S, D), dtype=bool)
    laps_done = np.ones((S, D), dtype=np.int32)

    hazard = 1.0 - (1.0 - np.clip(dnf_per_race, 0, 0.9)) ** (1.0 / n_laps)

    # 4. laps 2..N
    for lap in range(2, n_laps + 1):
        under_sc = sc_mask[:, lap]

        lt = base_lap + rng.normal(0, lap_noise, size=(S, D)) + deg_s * tyre_age
        pitting = (pit_laps == lap).any(axis=2)
        loss = np.where(under_sc[:, None], pit_loss * sc_discount, pit_loss)
        lt = lt + pitting * loss
        tyre_age = np.where(pitting, 0, tyre_age + 1)

        cum = cum + lt * alive
        laps_done = laps_done + alive

        dnf = (rng.random((S, D)) < hazard[None, :]) & alive
        alive = alive & ~dnf

        # track position
        cum_eff = np.where(alive, cum, np.inf)
        idx = np.argsort(cum_eff, axis=1)
        cum_sorted = np.take_along_axis(cum_eff, idx, axis=1)
        base_sorted = np.take_along_axis(base_lap, idx, axis=1)

        for k in range(1, D):
            ahead = cum_sorted[:, k - 1]
            behind = cum_sorted[:, k]
            both_running = np.isfinite(behind) & np.isfinite(ahead)
            gap = np.subtract(
                behind, ahead, out=np.full_like(behind, np.inf), where=both_running
            )
            close = both_running & (gap < follow_gap)
            faster = base_sorted[:, k] < (base_sorted[:, k - 1] - pace_threshold)
            passes = close & faster & (rng.random(S) < pass_prob)
            blocked = close & ~passes

            cum_sorted[:, k] = np.where(
                blocked, ahead + follow_gap + dirty_air, cum_sorted[:, k]
            )
            if passes.any():
                new_behind = np.where(passes, ahead + pass_cost, cum_sorted[:, k])
                new_ahead = np.where(passes, behind + pass_cost, cum_sorted[:, k - 1])
                cum_sorted[:, k] = new_behind
                cum_sorted[:, k - 1] = new_ahead

        if under_sc.any():
            leader = cum_sorted[:, 0]
            compressed = leader[:, None] + np.arange(D)[None, :] * sc_gap
            cum_sorted = np.where(
                under_sc[:, None] & np.isfinite(cum_sorted), compressed, cum_sorted
            )

        cum = np.where(alive, np.take_along_axis(cum_sorted, np.argsort(idx, axis=1), axis=1), cum)

    # 5. classification: finishers by time, retirements by laps completed
    rank_key = np.where(alive, cum, 1e9 + (n_laps - laps_done) * 1e3 + cum / 1e6)
    finish = np.argsort(np.argsort(rank_key, axis=1), axis=1) + 1

    return {
        "finish_positions": finish,
        "alive": alive,
        "n_stops": n_stops,
        "pass_prob": pass_prob,
    }


def summarise(finish: np.ndarray, drivers: list[str], grid: np.ndarray) -> list[dict]:
    S = finish.shape[0]
    rows = []
    for i, drv in enumerate(drivers):
        pos = finish[:, i]
        win = float((pos == 1).mean())
        pod = float((pos <= 3).mean())
        t10 = float((pos <= 10).mean())
        rows.append(
            {
                "driver": drv,
                "grid_position": int(grid[i]),
                "win_pct": 100 * win,
                "win_mc_stderr_pct": 100 * float(np.sqrt(win * (1 - win) / S)),
                "podium_pct": 100 * pod,
                "podium_mc_stderr_pct": 100 * float(np.sqrt(pod * (1 - pod) / S)),
                "top10_pct": 100 * t10,
                "expected_finish": float(pos.mean()),
                "median_finish": float(np.median(pos)),
                "p10_finish": float(np.percentile(pos, 10)),
                "p90_finish": float(np.percentile(pos, 90)),
                "expected_positions_gained": float(grid[i] - pos.mean()),
            }
        )
    return sorted(rows, key=lambda r: -r["win_pct"])
