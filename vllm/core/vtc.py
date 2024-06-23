
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from collections import deque

class VTC:
    def __init__(self, w_p = 1, w_q = 2):
        self.vtc: Dict[int, float] = {}
        self.vtc[0] = 0
        self.w_p = w_p
        self.w_q = w_q
        
        # user_id to requests
        self.user_id_to_request:Dict[int, Deque[SequenceGroup]] = {} # naturally first come first serve
        
        self.user_id_to_priority:Dict[int, float] = {}

    def set_user_priority(self, priority_dict:Dict[int, float]):
        for user_id, priority in priority_dict.items():
            self.user_id_to_priority[user_id] = priority

    def new_seq_come(self, seq_group: SequenceGroup, waiting: Deque[SequenceGroup], last_uid_left = 0):
        # bring up awaken user or new user
        if seq_group.user_id not in self.vtc:
            self.vtc[seq_group.user_id] = 0
        if seq_group.user_id not in self.user_id_to_request:
            self.user_id_to_request[seq_group.user_id] = deque([seq_group])
        else:
            self.user_id_to_request[seq_group.user_id].append(seq_group)

        if seq_group.user_id not in self.user_id_to_priority:
            self.user_id_to_priority[seq_group.user_id] = 1.0

        user_not_waiting = True
        for seq in waiting:
            if seq.user_id == seq_group.user_id:
                user_not_waiting = False
                break

        if user_not_waiting:
            if len(waiting) == 0:
                self.vtc[seq_group.user_id] = max(self.vtc[seq_group.user_id], 
                                                  self.vtc[last_uid_left])
            else:
                others_count_min = 2**31
                for seq in waiting:
                    temp = self.vtc[seq.user_id]
                    if others_count_min > temp:
                        others_count_min = temp
                self.vtc[seq_group.user_id] = max(self.vtc[seq_group.user_id], 
                                                  others_count_min)

    def update_count(self, seq_group_metadata_list: List[SequenceGroupMetadata]):
        for metadata in seq_group_metadata_list:
            if metadata.is_prompt:
                assert metadata.token_chunk_size is not None
                self.vtc[metadata.user_id] += self.w_p * metadata.token_chunk_size / self.user_id_to_priority[metadata.user_id]
            else:
                self.vtc[metadata.user_id] += self.w_q / self.user_id_to_priority[metadata.user_id]

        # print(self.vtc)
    def free_finished_seq_groups(self, seq_group_list: Deque[SequenceGroup]):
        for seq_group in seq_group_list:
            if seq_group.is_finished():
                self.user_id_to_request[seq_group.user_id].remove(seq_group)
    
    def get_user_id_order(self):
        # smallest vtc to highest vtc
        return [k for k, v in sorted(self.vtc.items(), key=lambda item: item[1])]
