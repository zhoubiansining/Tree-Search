from CoT.task import CoT_Task
from ToT.task import ToT_Task
from MCTS.task import MCTS_Task
from utils.visualize import visualize


def simple_run(q, mode, iteration_limit=50):
    if mode == 'mcts':
        Task = MCTS_Task(q, iteration_limit=iteration_limit)
        final_answer, root = Task.run()
        print('-' * 100)
        print(final_answer['solution'] + final_answer['summary'])
        visualize(root, Task, 'simpleRun', 'test', 1)
    elif mode == 'tot':
        Task = ToT_Task(q)
        final_answer, root = Task.run()
        print('-' * 100)
        print(final_answer['solution'] + final_answer['summary'])
        visualize(root, Task, 'simpleRun', 'test', 1)
    elif mode == 'cot':
        Task = CoT_Task(q)
        final_answer = Task.run()
        print('-' * 100)
        print(final_answer['solution'])
    else:
        print("Unsupported mode!\n")


if __name__ == '__main__':
    question = '''
        A fluid has density $870 \mathrm{~kg} / \mathrm{m}^3$ and flows with velocity $\mathbf{v}=z \mathbf{i}+y^2 \mathbf{j}+x^2 \mathbf{k}$, where $x, y$, and $z$ are measured in meters and the components of $\mathbf{v}$ in meters per second. 
        Find the rate of flow outward through the cylinder $x^2+y^2=4$, $0 \leqslant z \leqslant 1$. 
        The unit of the answer should be  $\mathrm{kg}/\mathrm{s}$.
        '''
    simple_run(question, 'tot')
