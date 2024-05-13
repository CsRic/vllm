from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

class RequestTestProfile:
    def __init__(self, user_id: int, prompt_len: int, now: float):
        self.user_id = user_id
        self.prompt_len = prompt_len
        self.timestamp: List[float] = [now] # first for request submission. then each as a decode step
        self.is_finish: bool = False
    
    def add_timestamp(self, now, is_finish: bool = False):
        self.timestamp.append(now)
        self.is_finish = is_finish

class UserLog:
    def __init__(self):
        self.request_profiles: Dict[int, RequestTestProfile] = {}
        self.user_id_to_request_ids: Dict[int, List[int]] = {}

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

    def print_summary(self):
        for user_id, request_ids in self.user_id_to_request_ids.items():
            print(f"\n\nuser_id: {user_id}")
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