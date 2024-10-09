from tasks.science import SearchTask
from models.get_response import *


class CoT_Task(SearchTask):
    def __init__(self, data, propose_method='glm', value_method='glm', temperature=0.7, max_tokens=1000, seed=170,
                 max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=512, evaluate=False):
        super().__init__(data, propose_method, value_method)
        self.mode = 'cot'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.evaluate = evaluate

    @staticmethod
    def get_summary(solution: str):
        if "综上所述，" in solution:
            summ = solution.split("综上所述，")[-1]
            return "综上所述，" + summ
        elif '。' in solution:
            summ = solution.split("。")[-2]
            return "综上所述，" + summ + '。'
        else:
            return ''

    def run(self):
        self.clear_cache()
        prompt = self.cot_prompt_wrap(self.question)
        solution = get_proposal(prompt, self.propose_method, temperature=self.temperature, max_tokens=self.max_tokens,
                                seed=self.seed, max_length=self.max_length, truncation=self.truncation,
                                do_sample=self.do_sample, max_new_tokens=self.max_new_tokens)
        out = ''
        for _ in solution:
            out = out + _
        out = out.strip()
        print(f'获得解答:{out}\n')
        summary = self.get_summary(out)
        return {'content': self.question, 'solution': out, 'summary': summary}
