from collections import deque
from typing import Deque

from vllm.sequence import SequenceGroup

from vllm.core.vtc import VTC

from functools import partial
class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class FairnessPolicy(Policy):
    def get_priority(self, vtc: VTC, seq_group: SequenceGroup):
        return vtc.vtc[seq_group.user_id]
    
    def sort_by_priority(
        self,
        vtc: VTC,
        seq_groups: Deque[SequenceGroup],
    )-> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(vtc, seq_group),
                reverse=False,
            )
        )

    def time_elapsed(self, now, seq_group: SequenceGroup):
        return now - seq_group.metrics.arrival_time

    def get_one_highest_priority(
        self,
        now,
        vtc: VTC,
        seq_groups: Deque[SequenceGroup], # assumed sorted by time
    )->SequenceGroup:
        compare_with_now = partial(self.time_elapsed, now=now)
        user_id_order = vtc.get_user_id_order()
        for user_id in user_id_order:
            for seq_group in seq_groups:
                if seq_group.user_id == user_id:
                    return seq_group

    def sort_by_time(self, now, seq_groups: Deque[SequenceGroup]):
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.time_elapsed(now, seq_group),
                reverse=True,
            ))

class PolicyFactory:
    _POLICY_REGISTRY = {'fcfs': FCFS,
                        'fair': FairnessPolicy}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)



