"""
Advancing the per-step time scalars of a case between the steps of a multi-step run.

A unit test reads these scalars from the capture every step (time_scalars.py, `spel_t_<name>`);
a multi-step proptest run has no capture to read, so it advances them itself, by one time step
`dtime_mod` on ELM's no-leap calendar. The values of step 1 are the domain's (`[domain.time]`,
then domain.TIME_DEFAULTS, ...); every name stands for the time at the end of the step, as ELM's
`get_curr_date`/`get_curr_calday` do.

What elm_drv sets each step (elm_driver.F90: `nstep_mod = get_nstep()`, `get_curr_date(year_curr,
mon_curr, day_curr, secs_curr)`, `jday_mod = get_curr_calday()`, `dayspyr_mod`) advances as ELM
does. The others ELM leaves at what the capture or the domain gave them (its
`timeinfoMod::increment_time_vars` is never called); here they follow the same date, so that they
agree with it: `thiscalday_mod` (calendar day with its fraction), `nextsw_cday_mod` (the next
step's), `end_cd_mod` (the step ends at midnight), the `*_prev` date (the previous step's), and
`spel_nstep` (spel_timeMod, = nstep). `doalb`, `first`, `dtime_mod`, the step sizes and anything
else stay as they are.
"""
from __future__ import annotations

from typing import Any

DAYS_PER_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # ELM's no-leap calendar
SECONDS_PER_DAY = 86400


def calday(mon: int, day: int, secs: float) -> float:
    """Calendar day of the year, 1-based, with the fraction of the day (1 Jan 00:00 = 1.0)."""
    return float(sum(DAYS_PER_MONTH[: int(mon) - 1]) + int(day)) + float(secs) / SECONDS_PER_DAY


def advance(values: dict[str, Any]) -> dict[str, Any]:
    """The time scalars of the next step: `values` (name -> value; only the names present are
    changed) one `dtime_mod` later."""
    v = {k.lower(): x for k, x in values.items()}
    out = dict(v)
    dt = float(v.get("dtime_mod", 1800.0))
    if dt <= 0 or abs(dt - round(dt)) > 1e-9:
        raise ValueError(f"dtime_mod = {dt}: a multi-step run needs a positive whole number of seconds")
    year, mon, day, secs = (int(v.get(k, d)) for k, d in
                            (("year_curr", 2000), ("mon_curr", 1), ("day_curr", 1), ("secs_curr", 0)))
    if not (1 <= mon <= 12 and 1 <= day <= DAYS_PER_MONTH[mon - 1] and 0 <= secs < SECONDS_PER_DAY):
        raise ValueError(f"not a date of the no-leap calendar: {year}-{mon:02d}-{day:02d} {secs} s")
    prev = (year, mon, day, secs)
    secs += int(round(dt))
    while secs >= SECONDS_PER_DAY:
        secs -= SECONDS_PER_DAY
        day += 1
        if day > DAYS_PER_MONTH[mon - 1]:
            day, mon = 1, mon + 1
            if mon > 12:
                mon, year = 1, year + 1
    new = {"year_curr": year, "mon_curr": mon, "day_curr": day, "secs_curr": secs,
           "year_prev": prev[0], "mon_prev": prev[1], "day_prev": prev[2], "secs_prev": prev[3]}
    cd = calday(mon, day, secs)
    nxt = cd + dt / SECONDS_PER_DAY
    if nxt >= 366.0:
        nxt -= 365.0
    new.update(jday_mod=int(cd), thiscalday_mod=cd, nextsw_cday_mod=nxt, end_cd_mod=(secs == 0),
               dayspyr_mod=365.0)
    if "nstep_mod" in v:
        new["nstep_mod"] = int(v["nstep_mod"]) + 1
    if "spel_nstep" in v:
        new["spel_nstep"] = int(new.get("nstep_mod", int(v["spel_nstep"]) + 1))
    for k, x in new.items():
        if k in v:
            out[k] = type(v[k])(x) if isinstance(v[k], (bool, int, float)) and not isinstance(x, type(v[k])) else x
    return out


def inconsistent(values: dict[str, Any]) -> list[str]:
    """Messages for start values that disagree with their own date (the date wins when advancing)."""
    v = {k.lower(): x for k, x in values.items()}
    if not all(k in v for k in ("mon_curr", "day_curr", "secs_curr")):
        return []
    cd = calday(int(v["mon_curr"]), int(v["day_curr"]), float(v["secs_curr"]))
    out = []
    if "jday_mod" in v and int(v["jday_mod"]) != int(cd):
        out.append(f"jday_mod = {v['jday_mod']} but the date gives {int(cd)}")
    if "thiscalday_mod" in v and abs(float(v["thiscalday_mod"]) - cd) > 1e-9:
        out.append(f"thiscalday_mod = {v['thiscalday_mod']} but the date gives {cd}")
    return out
