import os
import pathlib
from CoT.task import CoT_Task
from ToT.task import ToT_Task
from MCTS.task import MCTS_Task
import argparse
from utils.visualize import visualize
from utils.json_operator import *
from utils.verify_answer import *


def run(arguments):
    print('-'*30, '开始测试', '-'*30, '\n')
    file = f'data/{arguments.task_name}/{arguments.file}.json'
    try:
        data_list = read_json(file)
        data_len = len(data_list)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    assert 'content' in data_list[0].keys() and 'answer' in data_list[0].keys(), "Key error, Make sure json object contain correct keys!\n"

    output_list = []
    correct_count = 0
    for i in range(data_len):
        # solve
        print(f'开始解答第{i+1}题...\n')
        data = data_list[i]['content']
        answer = data_list[i]['answer']
        if arguments.mode == 'cot':
            Task = CoT_Task(data, arguments.propose_method, arguments.value_method, arguments.temperature, evaluate=arguments.evaluate)
            output = Task.run()
        elif arguments.mode == 'tot':
            Task = ToT_Task(data, arguments.propose_method, arguments.value_method, arguments.algorithm,
                            arguments.branch, arguments.select_branch, arguments.max_depth, arguments.end_gate,
                            arguments.select_method, arguments.temperature, use_case_prompt=arguments.use_case_prompt,
                            low=arguments.low, high=arguments.high, evaluate=arguments.evaluate)
            output, root = Task.run()
            if arguments.visualize:
                visualize(root, Task, arguments.task_name, arguments.file, i + 1)
        else:
            Task = MCTS_Task(data, arguments.propose_method, arguments.value_method, arguments.branch, arguments.end_gate,
                             arguments.roll_policy, arguments.roll_branch, arguments.roll_forward_steps, arguments.time_limit,
                             arguments.iteration_limit, arguments.exploration_constant, arguments.alpha, arguments.inf,
                             arguments.temperature, use_case_prompt=arguments.use_case_prompt, use_reflection=arguments.use_reflection,
                             low=arguments.low, high=arguments.high, evaluate=arguments.evaluate)
            output, root = Task.run()
            if arguments.visualize:
                visualize(root, Task, arguments.task_name, arguments.file, i + 1)

        # evaluate metrics
        if arguments.evaluate:
            result = verify_float(answer, output['summary'])
            output.update({'answer': answer, 'accurate': result})
            if result:
                print(f'模型第{i+1}题解答正确。\n')
                correct_count += 1
            else:
                print(f'模型第{i+1}题解答错误。\n')
        print(f'第{i+1}题解答结束。\n')

        # output
        base_dir = os.getcwd()
        output_dir = pathlib.Path(f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}')
        output_file = f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}/{Task.propose_method}_{Task.value_method}.json'
        output_list.append(output)
        pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
        dump_json(output_file, output_list)

    print('_' * 60)
    # accuracy
    if args.evaluate:
        print(f'测试准确率:{correct_count / data_len}\n')
        print(f'正确题目数:{correct_count}\n总题目数:{data_len}\n')
    print('_' * 60)


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='scibench')
    base_args.add_argument('--file', type=str, default='calculus_standardized')
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'local'], default='glm')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='glm')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='cot')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=None)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.4)
    base_args.add_argument('--roll_forward_steps', type=int, default=2)
    base_args.add_argument('--end_gate', type=float, default=0.9)
    base_args.add_argument('--branch', type=int, default=3)
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--inf', type=float, default=1.0)
    base_args.add_argument('--evaluate', type=bool, default=False)
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', type=bool, default=False)
    base_args.add_argument('--use_case_prompt', type=bool, default=False)
    base_args.add_argument('--use_reflection', type=bool, default=False)
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--algorithm', type=str, choices=['dfs', 'bfs'], default='dfs')
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=10)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    run(args)
