import numpy as np


class OfflineEnv(object):
    
    def __init__(self, 
                users_dict: dict[int, list[tuple[int, int]]], # user_id : [(movie_id, rating), ...] -> 총 4832 user, id는 1부터 시작, 앞에 있는 movie가 최근에 본 movie
                users_history_lens: int, # user_id 별 history 길이
                movies_id_to_movies: dict[str, tuple[str, str]], # movie_id : (title, genre)
                state_size, # N
                fix_user_id=None):

        self.users_dict = users_dict                    
        self.users_history_lens = users_history_lens    
        self.items_id_to_name = movies_id_to_movies     
        
        self.state_size = state_size                    # state size = 10 : 최근에 본 movie의 개수 기준 설정 
        self.available_users = self._generate_available_users() # 현재 추천가능한 user list 추리기

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)      # 이번 에피소드에서 movie를 추천할 user
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}   # 해당 user가 본 {movie: rating}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]   # 해당 user의 history 중 최근 10개의 movie의 id
        self.done = False
        self.recommended_items = set(self.items)      # 추천된 movie들의 id set
        self.done_count = 3000                        # trajectory의 최대 길이
        
    def _generate_available_users(self):
        """
        self.state_size (10) 보다 긴 history를 가진 user 들만 available_users에 추가
        episode가 진행될 수록 available user가 감소하는건 아닌건가? 그러면 불필요하게 한 user에 대해서 여러번 훈련될 수도 있나? 
        """
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users
    
    def reset(self):
        """
        init 반복
        """
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]} # {movie : rating}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
        
    def step(self, action, top_k=False):
        
        reward = -0.5

        # 일단 top_k 안함 
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append((self.user_items[act] - 3)/2)
                else:
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards

        else:
            # 추천한 movie가 user가 봤던 history에 있고, 최근 본 10개 중에 없다면.
            if action in self.user_items.keys() and action not in self.recommended_items:
                reward = self.user_items[action] -3  # reward : rating이 1~5까지니까 4, 5인 경우에 + reward를 받게 됨
            
            # reward가 0 보다 크다면, 추천한 movie를 history에 추가 -> 이건 현재 episode에서만 반영됨 (user가 중복 학습 안된다고 하면 문제 없음)
            # self.items: 지금은 10개가 순서 무관하게 동일하게 취급되지만, 최근꺼를 더 반영하는 식으로 수정가능할 듯
            if reward > 0:
                self.items = self.items[1:] + [action]
            
            # 추천받은 movie에 action 추가
            self.recommended_items.add(action)

        # 추천받은 item개수(=trajectory 길이)가 done_count보다 크거나, 추천받은 item 개수가 user의 history 길이보다 크거나 같다면
        # self.user -1 인 이유는 user id가 1부터 시작해서
        # 추천받은 item 개수가 user의 history 길이보다 크거나 같은지는 왜 체크하지?
        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user-1]:
            self.done = True
            
        # self.items : 이게 state가 만약 적절한 추천을 해줬다면 다음 self.items에 추가되었을 것이고 state만드는데 사용됨
        return self.items, reward, self.done, self.recommended_items

    # 여기서 쓰이는 곳은 없음
    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names