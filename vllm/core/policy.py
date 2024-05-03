from collections import deque
from typing import Deque

from vllm.sequence import SequenceGroup

from vllm.core.vtc import VTC
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


class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS,
                        'fair': FairnessPolicy}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)



