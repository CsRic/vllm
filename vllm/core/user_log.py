from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
from vllm.sequence import SequenceGroup

import pandas as pd

class RequestTestProfile:
    def __init__(self, user_id: int, prompt_len: int, now: float):
        self.user_id = user_id
        self.prompt_len = prompt_len
        self.timestamp: List[float] = [now] # first for request submission. then each as a decode step
        self.is_finish: bool = False

    def add_timestamp(self, now, is_finish: bool = False):
        self.timestamp.append(now)
        self.is_finish = is_finish


class LLMEngineProfile:
    def __init__(self):
        self.step_count = 0
        self.step_batchsize_list: List[tuple[int]] = [] # each step, how many prefills, how many decodes
    
    def add_step(self, running: Deque[SequenceGroup]):
        self.step_count += 1
        num_prefills = 0
        num_decodes = 0
        for sq in running:
            if sq.is_prefill():
                num_prefills += 1
            else:
                num_decodes += 1
        self.step_batchsize_list.append((num_prefills, num_decodes))
        
class UserLog:
    def __init__(self):
        self.request_profiles: Dict[int, RequestTestProfile] = {}
        self.user_id_to_request_ids: Dict[int, List[int]] = {}
        self.summary_count = 0
        
        self.engine_profile = LLMEngineProfile()
        
        self.first_now = 10**12
        self.last_now = 0

    def submit_request(self, request_id: int, user_id: int, prompt_len: int, now: float):
        self.request_profiles[request_id] = RequestTestProfile(user_id, prompt_len, now)
        if user_id not in self.user_id_to_request_ids:
            self.user_id_to_request_ids[user_id] = [request_id]
        else:
            self.user_id_to_request_ids[user_id].append(request_id)

    def add_timestamp(self, request_id: int, now: float, is_finish: bool = False):
        if request_id not in self.request_profiles:
            return
        self.request_profiles[request_id].add_timestamp(now, is_finish)
        if now < self.first_now:
            self.first_now = now
        self.last_now = now

    def add_running_schedule(self, running: Deque[SequenceGroup]):
        self.engine_profile.add_step(running)

    def _avg_prompt_throughput(self, request_ids: List[int]):
        total_time = 0
        total_len = 0
        for request_id in request_ids:
            single_profile = self.request_profiles[request_id]
            if len(single_profile.timestamp) < 2:
                continue
            total_time += single_profile.timestamp[1] - single_profile.timestamp[0]
            total_len += single_profile.prompt_len + 1 # because the first decode token is also included
        if(total_time == 0):
            return 0
        return total_len / total_time
    
    def _avg_generation_throughput(self, request_ids: List[int]):
        total_time = 0
        total_len = 0
        for request_id in request_ids:
            single_profile = self.request_profiles[request_id]
            if len(single_profile.timestamp) < 3:
                continue
            total_time += single_profile.timestamp[-1] - single_profile.timestamp[1]
            total_len += len(single_profile.timestamp[2:])
        if(total_time == 0):
            return 0
        return total_len / total_time
    
    def _avg_first_token_time(self, request_ids: List[int]):
        total_time = 0
        total_requests = 0
        for request_id in request_ids:
            single_profile = self.request_profiles[request_id]
            if len(single_profile.timestamp) < 2:
                continue
            total_time += single_profile.timestamp[1] - single_profile.timestamp[0]
            total_requests += 1
        if(total_requests == 0):
            return 0
        return total_time / total_requests
    
    def _avg_total_time(self, request_ids: List[int]):
        total_time = 0
        total_requests = 0
        for request_id in request_ids:
            single_profile = self.request_profiles[request_id]
            if not single_profile.is_finish:
                continue
            total_time += single_profile.timestamp[-1] - single_profile.timestamp[0]
            total_requests += 1
        if(total_requests == 0):
            return 0
        return total_time / total_requests
    
    def _avg_per_token_time(self, request_ids: List[int]):
        # generation phase only
        total_time = 0
        total_len = 0
        for request_id in request_ids:
            single_profile = self.request_profiles[request_id]
            if len(single_profile.timestamp) < 2:
                continue
            total_time += single_profile.timestamp[-1] - single_profile.timestamp[1]
            total_len += len(single_profile.timestamp[2:])
        if(total_len == 0):
            return 0
        return total_time / total_len

    def _avg_batch_size(self):
        total_prefill = 0
        total_decode = 0
        for record in self.engine_profile.step_batchsize_list:
            total_prefill += record[0]
            total_decode += record[1]
        avg_prefill = total_prefill / self.engine_profile.step_count
        avg_decode = total_decode / self.engine_profile.step_count
        return avg_prefill, avg_decode

    def print_summary(self):
        print(f"----------------------summary {self.summary_count}----------------------")
        '''
        print(f"prefill / decode batchsize: {self._avg_batch_size()}")
        for user_id, request_ids in self.user_id_to_request_ids.items():
            print(f"user_id: {user_id}")
            p0 = self._avg_prompt_throughput(request_ids)
            p1 = self._avg_generation_throughput(request_ids)
            p2 = self._avg_first_token_time(request_ids)
            p3 = self._avg_total_time(request_ids)
            p4 = self._avg_per_token_time(request_ids)
            print(f"avg_prompt_throughput: {p0}")
            print(f"avg_generation_throughput: {p1}")
            print(f"avg_first_token_time: {p2}")
            print(f"avg_total_time: {p3}")
            print(f"avg_per_token_time: {p4}")
            print(f"{user_id},{p0},{p1},{p2},{p3},{p4}")
        '''
        batch_size = self._avg_batch_size()
        print(f"{batch_size[0]}, {batch_size[1]}")
        for user_id, request_ids in self.user_id_to_request_ids.items():
            p0 = self._avg_prompt_throughput(request_ids)
            p1 = self._avg_generation_throughput(request_ids)
            p2 = self._avg_first_token_time(request_ids)
            p3 = self._avg_total_time(request_ids)
            p4 = self._avg_per_token_time(request_ids)
            print(f"{user_id},{p0},{p1},{p2},{p3},{p4}")
        self.summary_count += 1

    def calc_average_interval(self, interval, save_path = "user_log.csv"):
        time_start = self.first_now
        time_end = self.first_now + interval
        def get_time_slot(time_stamp):
            return int((time_stamp - self.first_now) / interval)

        total_slots = int((self.last_now - self.first_now) / interval) + 1
        user_records = {}
        for user_id, request_ids in self.user_id_to_request_ids.items():
            prompt_throughput = [[0,0] for _ in range(total_slots)] # total time, total length
            first_token_time = [[0,0] for _ in range(total_slots)]
            generation_throughput = [[0,0] for _ in range(total_slots)]
            total_time = [[0,0] for _ in range(total_slots)]
            per_token_time = [[0,0] for _ in range(total_slots)]
            for request_id in request_ids:
                single_profile = self.request_profiles[request_id]
                # prompt_throughput
                # first_token_time
                if len(single_profile.timestamp) < 2:
                    continue
                time_slot = get_time_slot(single_profile.timestamp[1])
                prompt_throughput[time_slot][0] += single_profile.prompt_len + 1
                # prompt_throughput[time_slot][1] += single_profile.timestamp[1] - single_profile.timestamp[0]
                prompt_throughput[time_slot][1] = interval
                
                first_token_time[time_slot][0] += single_profile.timestamp[1] - single_profile.timestamp[0]
                first_token_time[time_slot][1] += 1

                # total_time
                if single_profile.is_finish:
                    time_slot = get_time_slot(single_profile.timestamp[-1])
                    total_time[time_slot][0] += single_profile.timestamp[-1] - single_profile.timestamp[0]
                    total_time[time_slot][1] += 1
                
                # generation_throughput
                # per_token_time
                if len(single_profile.timestamp) < 3:
                    continue
                for i in range(2, len(single_profile.timestamp)):
                    time_slot = get_time_slot(single_profile.timestamp[i])
                    generation_throughput[time_slot][0] += 1
                    # generation_throughput[time_slot][1] += single_profile.timestamp[i] - single_profile.timestamp[i-1]
                    generation_throughput[time_slot][1] = interval
    
                    per_token_time[time_slot][0] += single_profile.timestamp[i] - single_profile.timestamp[i-1]
                    per_token_time[time_slot][1] += 1
            
            # finsh up
            for i in range(total_slots):
                for l in [prompt_throughput, first_token_time, generation_throughput, total_time, per_token_time]:
                    l[i] = l[i][0] / l[i][1] if l[i][1] != 0 else 0
            user_records[user_id] = [prompt_throughput, first_token_time, generation_throughput, total_time, per_token_time]
        # save
        rows = []
        for user_id, metrics in user_records.items():
            for slot in range(total_slots):
                row = [user_id, slot]
                for metric in metrics:
                    row.append(metric[slot])
                rows.append(row)
        columns = ["user_id", "time_slot", "prompt_throughput", "first_token_time", "generation_throughput", "total_time", "per_token_time"]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(save_path, index=False)
        return user_records

    def clean_finished(self):
        keys_to_remove = [key for key, value in self.request_profiles.items() if value.is_finish]
        remaining_profiles  = {key: value for key, value in self.request_profiles.items() if not value.is_finish}
        self.request_profiles = remaining_profiles
        for request_list in self.user_id_to_request_ids.values():
            for key in keys_to_remove:
                if key in request_list:
                    request_list.remove(key)
