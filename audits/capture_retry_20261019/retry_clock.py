"""Experimental game-observer scheduler. Does not alter the production runtime.

This class schedules attempts, not physical integration or radar pulses. The
callback must evaluate current inputs; it must not cache earlier visibility.
Integer global ticks are explicit to prevent fitting a per-entity epoch.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable

@dataclass(frozen=True)
class Decision:
    tick: int
    state: str
    attempted: bool
    searching: bool
    accepted: bool
    reason: str
    next_search_tick: int | None
    last_accepted_tick: int | None

class RetryClock:
    def __init__(self, period_ticks: int = 48, phase_tick: int = 1,
                 hold_ticks: int = 48):
        for name, value in [('period_ticks', period_ticks), ('phase_tick', phase_tick), ('hold_ticks', hold_ticks)]:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + ' must be an integer')
        if period_ticks <= 0 or not 0 <= phase_tick < period_ticks or hold_ticks < 0:
            raise ValueError('Invalid period, phase or hold duration')
        self.period = period_ticks
        self.phase = phase_tick
        self.hold_ticks = hold_ticks
        self.first_attempted = False
        self.ever_accepted = False
        self.last_accepted_tick = None
        self.next_search_tick = None
        self.last_tick = None
        self.last_decision = None
        self.attempt_count = 0

    def next_phase(self, tick: int, *, strictly_after: bool = True) -> int:
        lo = tick + int(strictly_after)
        return lo + ((self.phase - lo) % self.period)

    def step(self, tick: int, *, enabled: bool, search_ready: bool,
             evaluate: Callable[[bool], bool]) -> Decision:
        """Evaluate at most once per tick; evaluate(searching) returns acceptance.

        search_ready must depend only on available input state, e.g. an INS
        range estimate. It is not a precomputed future capture label. During
        tracking/hold, this search-only readiness gate is deliberately ignored.
        Disabled periods or a range exit preserve previous failed-search state.
        """
        if isinstance(tick, bool) or not isinstance(tick, int):
            raise TypeError('global tick must be an integer')
        if self.last_tick is not None:
            if tick < self.last_tick:
                raise ValueError('time must not move backwards')
            if tick == self.last_tick:
                return self.last_decision
        holding = (self.last_accepted_tick is not None and
                   tick - self.last_accepted_tick <= self.hold_ticks)
        searching = not holding
        attempted = accepted = False
        if not enabled:
            state, reason = 'INACTIVE', 'not_enabled'
        elif holding:
            attempted = True
            accepted = bool(evaluate(False))
            state, reason = ('TRACK', 'accepted') if accepted else ('HOLD', 'no_accepted_observation')
        elif not search_ready:
            state, reason = 'WAIT_PRIOR', 'search_not_ready'
        elif not self.first_attempted:
            attempted = True
            accepted = bool(evaluate(True))
            self.first_attempted = True
            state, reason = ('TRACK', 'initial_attempt_accepted') if accepted else ('RETRY', 'initial_attempt_failed')
        else:
            if self.next_search_tick is None or self.next_search_tick < tick:
                # Never perform a missed historical check with today's geometry.
                self.next_search_tick = self.next_phase(tick, strictly_after=False)
            if tick == self.next_search_tick:
                attempted = True
                accepted = bool(evaluate(True))
                state, reason = ('TRACK', 'retry_accepted') if accepted else ('RETRY', 'retry_failed')
            else:
                state, reason = 'RETRY', 'await_global_phase'
        if attempted:
            self.attempt_count += 1
            if accepted:
                self.ever_accepted = True
                self.last_accepted_tick = tick
                self.next_search_tick = None
            elif searching:
                self.next_search_tick = self.next_phase(tick)
        self.last_tick = tick
        result = Decision(tick, state, attempted, searching, accepted, reason,
                          self.next_search_tick, self.last_accepted_tick)
        self.last_decision = result
        return result

    def snapshot(self) -> dict:
        return {} if self.last_decision is None else asdict(self.last_decision)
