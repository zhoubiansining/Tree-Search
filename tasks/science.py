import re
import os
from tasks.prompts import *


# data: question: str
# mode: 'cot', 'tot', 'mcts'
# method: 'glm', 'gpt', 'local'
class SearchTask(object):
    def __init__(self, data, propose_method='glm', value_method='glm'):
        super().__init__()
        self.question = data
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}

    def clear_cache(self):
        self.value_cache = {}

    @staticmethod
    def summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'summary', '==============================', '\n')
        print('summary_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤的综述为:')
        prompt = summary_prompt + x + '\n已有步骤:\n' + y + '\n输出:'
        return prompt

    @staticmethod
    def evaluate_summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'summary', '==============================', '\n')
        print('summary_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤的综述为:')
        prompt = evaluate_summary_prompt + x + '\n已有步骤:\n' + y + '\n输出:'
        return prompt

    @staticmethod
    def single_propose_prompt_wrap(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:')
        prompt = single_proposal_prompt + x + '\n已有步骤:\n' + y + '\n输出:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:')
        prompt = zero_single_proposal_prompt + x + '\n已有步骤:\n' + y + '\n输出:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap_use_reflection(x: str, y: str = '', step: int = 0, ref: str = '') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:')
        prompt = zero_single_proposal_prompt_use_reflection + x + '\n已有步骤:\n' + y + '\n意见:' + ref + '\n输出:'
        return prompt

    @staticmethod
    def single_reflection_wrap(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:')
        prompt = single_reflection_prompt + x + '\n已有步骤:\n' + y + '\n输出:'
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'critic', '==============================', )
        value_prompt = critic_simplified + x + '\n已有步骤:\n' + y.strip() + '输出:\n'
        return value_prompt

    @staticmethod
    def cot_prompt_wrap(x: str) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\n')
        prompt = cot_prompt + x + "\n解答过程:"
        print('propose_prompt: \n', prompt, '\n')
        return prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, low=0.0, high=1.0) -> float:
        out_value = low
        all_out = ''
        for _ in value_outputs:
            all_out = all_out + _
        if '分数' not in all_out:
            print('分数输出不合法!\n')
            return out_value
        stp = all_out.split('分数')[-1].strip()
        try:
            match = re.findall(r'-?[0-9]+\.?[0-9]*', stp)[-1]
            out_value = float(match)
            out_value = min(max(low, out_value), high)
        except Exception as e:
            print(f'分数输出有误！错误类型:{e}\n')
            return low
        return out_value
