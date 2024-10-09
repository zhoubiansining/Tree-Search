import random
from tasks.science import SearchTask
from MCTS.base import treeNode
from models.get_response import *
from MCTS.mcts import MCTS


class MCTS_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', branch=3, end_gate=0.9, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=None, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=1000, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=512, use_case_prompt=False, use_reflection=False, low=0, high=1,
                 evaluate=False):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.branch = branch
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'

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

    def get_next_step_use_reflection(self, y, step_n, reflection):  # 暂不支持 case-prompt
        propose_prompt = self.zero_single_propose_wrap_use_reflection(self.question, y, step_n, reflection)
        response = get_proposal(propose_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
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

    def get_reflection(self, y, step_n):
        reflection_prompt = self.single_reflection_wrap(self.question, y, step_n)
        response = get_proposal(reflection_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('获得意见失败！\n')
            return ''

        p = ''
        for _ in response:
            p = p + _
        p = p.strip()

        if '已解决' in p or '已经解决' in p:
            print('此步问题已解决，停止下探。\n')
            return '<end>'

        if '意见:' not in p:
            print('输出格式有误！\n')
            return ''
        revised_ = p.split('意见:')[1]
        print(f'标准化后的意见:{revised_}\n')
        return revised_

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
        self.set_limit_type()
        solution, finish, root = MCTS(self)
        summary = self.get_summary(solution)
        final_answer = {'content': self.question, 'solution': solution, 'summary': summary, 'finish': finish}
        return final_answer, root
