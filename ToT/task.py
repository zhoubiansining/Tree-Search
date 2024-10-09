import random
from tasks.science import SearchTask
from ToT.base import Node
from models.get_response import *
from ToT.bfs import BFS
from ToT.dfs import DFS


class ToT_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', algorithm='dfs', branch=3, select_branch=2,
                 max_depth=8, end_gate=0.9, select_method='greedy',
                 temperature=0.7, max_tokens=1000,
                 seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=512, use_case_prompt=False, low=0, high=1, evaluate=False):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'tot'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.algorithm = algorithm
        self.branch = branch
        self.select_branch = select_branch
        self.max_depth = max_depth
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.select_method = select_method
        self.end_gate = end_gate
        self.node_count = 1

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def get_next_step(self, y, step_n):
        if self.use_case_prompt:
            prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
        else:
            prompt = self.zero_single_propose_wrap(self.question, y, step_n)

        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('获得下一步失败！\n')
            return ''

        p = ''
        for _ in response:
            p = p + _
        p = p.strip()

        if '下一步:' in p:
            stp = p.split('下一步:')[1].strip()
            if len(stp) < 2:
                print('输出步骤过短！\n')
                return ''
            if stp in y:
                print('输出步骤重复！\n')
                return ''

            revised_ = '步骤' + str(step_n) + ':' + stp
            print(f'标准化后新的步骤:{revised_}\n')
            return revised_ + '\n'

        elif '步骤' in p and ':' in p:
            pre_len = len(p.split(':')[0])
            p_ = p[pre_len:]
            p_ = p_.split('步骤')[0].strip()
            if len(p_) < 3:
                print('输出步骤过短！\n')
                return ''
            if p_[1:] in y:
                print('输出步骤重复！\n')
                return ''

            revised_ = '步骤' + str(step_n) + p_
            print(f'标准化后新的步骤:{revised_}\n')
            return revised_ + '\n'

        else:
            print('输出格式有误！\n')
            return ''

    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]
        prompt = self.value_prompt_wrap(self.question, y)

        if self.value_method == 'local':
            prompt_answer = self.question + '【答案】' + y
            value = get_value(prompt_answer, self.value_method, self.temperature, self.max_tokens, self.seed,
                              self.max_length, self.low, self.high)
            print(f'获得评分:{value}\n')
            self.value_cache.update({y: value})
            return value

        else:
            response = get_value(prompt, self.value_method, self.temperature, self.max_tokens, self.seed,
                                 self.max_length, self.low, self.high)
            value = self.value_outputs_unwrap(response, self.low, self.high)
            print(f'获得评分:{value}\n')
            self.value_cache.update({y: value})
            return value

    def get_summary(self, y):
        if self.evaluate:
            prompt = self.evaluate_summary_prompt_wrap(self.question, y)
        else:
            prompt = self.summary_prompt_wrap(self.question, y)

        response = get_proposal(prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)

        if not response:
            print('获得综述失败！\n')
            return ''
        p = ''
        for _ in response:
            p = p + _
        p = p.strip()

        if self.evaluate:
            if len(p) < 1:
                print('获得综述过短！\n')
                return ''

            if '综上所述，最终答案是:' not in p:
                summ = '综上所述，最终答案是:' + p
                print(f'获得综述:{summ}\n')
                return summ
            else:
                summ = '综上所述，最终答案是:' + p.split('综上所述，最终答案是:')[-1]
                print(f'获得综述:{summ}\n')
                return summ

        else:
            if len(p) < 1:
                print('获得综述过短！\n')
                return ''

            if '综上所述，' not in p:
                summ = '综上所述，' + p
                print(f'获得综述:{summ}\n')
                return summ
            else:
                summ = '综上所述，' + p.split('综上所述，')[-1]
                print(f'获得综述:{summ}\n')
                return summ

    def run(self):
        self.clear_cache()
        if self.algorithm == 'dfs':
            solution, root = DFS(self)
        elif self.algorithm == 'bfs':
            solution, root = BFS(self)
        else:
            print('Unsupported algorithm!\n')
            return {}
        summary = self.get_summary(solution)
        final_answer = {'content': self.question, 'solution': solution, 'summary': summary}
        return final_answer, root
