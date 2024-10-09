import time
import math
import random
import numpy
from functools import partial
import copy
from MCTS.base import treeNode


def get_next_steps_roll(y: str, step_n: int, mcts_task):
    next_steps = []
    reflection = mcts_task.get_reflection(y, step_n)
    for i in range(mcts_task.roll_branch):
        proposal = ''
        cnt = 15
        while not proposal and cnt:
            if mcts_task.use_reflection:
                proposal = mcts_task.get_next_step_use_reflection(y, step_n, reflection)
            else:
                proposal = mcts_task.get_next_step(y, step_n)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps


def get_next_steps_expand(node: treeNode, mcts_task):
    next_steps = []
    reflection = node.reflection
    for i in range(mcts_task.branch):
        proposal = ''
        cnt = 15
        while not proposal and cnt:
            if mcts_task.use_reflection:
                proposal = mcts_task.get_next_step_use_reflection(node.y, node.depth + 1, reflection)
            else:
                proposal = mcts_task.get_next_step(node.y, node.depth + 1)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps


def randomPolicy(node: treeNode, mcts_task):
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    reflection = mcts_task.get_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        print('此步问题已解决，不需要模拟。\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps):
        next_steps = get_next_steps_roll(strs, cur_step, mcts_task)
        if not next_steps:
            break
        action = random.choice(next_steps)  # str
        strs = strs + action
        cur_step += 1
        value = mcts_task.get_step_value(strs)
        if value > max_V:
            max_V = value
        cur_ref = mcts_task.get_reflection(strs, cur_step)
        if cur_ref == '<end>':
            break
    return max_V


def greedyPolicy(node: treeNode, mcts_task):
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    reflection = mcts_task.get_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        print('此步问题已解决，不需要模拟。\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps):
        actions = get_next_steps_roll(strs, cur_step, mcts_task)  # str_list
        if not actions:
            break
        new_ys = [strs + action for action in actions]
        cur_step += 1
        values = [mcts_task.get_step_value(new_y) for new_y in new_ys]
        idx = numpy.argmax(values)
        strs = new_ys[idx]
        value = values[idx]
        if value > max_V:
            max_V = value
        cur_ref = mcts_task.get_reflection(strs, cur_step)
        if cur_ref == '<end>':
            break
    return max_V


def MCTS_search(mcts_task):
    root = treeNode('')

    if mcts_task.limit_type == 'time':
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            print(f'<开始新搜索轮次，目前总时间:{time.time() - time_start}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('已找到解决方案！\n')
                return root, node, time.time() - time_start
    else:
        for i in range(mcts_task.iteration_limit):
            print(f'<开始新搜索轮次，目前已完成轮次数:{i}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('已找到解决方案！\n')
                return root, node, i + 1
    return root, None, None


def executeRound(root, mcts_task):
    """
        execute a selection-expansion-simulation-backpropagation round
    """
    print('-' * 40)
    print('选择节点阶段\n')
    flag, node = selectNode(root, mcts_task)
    if flag:
        return True, node, root

    print('-' * 40)
    print('扩充阶段\n')
    if node.reflection == '<end>':
        print('跳过此阶段。\n')
    else:
        node = expand(node, mcts_task)

    print('-' * 40)
    print('模拟搜索阶段\n')
    if node.reflection == '<end>':
        print('跳过此阶段。\n')
    else:
        roll_node = getBestChild(node, mcts_task)
        best_V = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node,
                                                                                                           mcts_task)
        roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
        roll_node.numVisits += 1

    print('-' * 40)
    print('反向传播阶段\n')
    back_propagate(node)
    return False, node, root


def isTerminal(node, mcts_task):
    return node.V >= mcts_task.end_gate


def selectNode(node, mcts_task):
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
    if isTerminal(node, mcts_task):
        node.final_ans_flag = 1
        return True, node
    else:
        return False, node


def expand(node: treeNode, mcts_task):
    if not node.reflection:
        reflection = mcts_task.get_reflection(node.y, node.depth + 1)
        node.update_reflection(reflection)
    if node.reflection == '<end>':
        return node
    actions = get_next_steps_expand(node, mcts_task)
    if not actions:
        node.update_reflection('<end>')
        return node

    for action in actions:
        if action not in node.children.keys():
            node.append_children(action)
            child = node.children[action]
            value = mcts_task.get_step_value(child.y)
            child.update_value(value)
            child.visit_sequence = mcts_task.node_count
            mcts_task.update_count()
    node.isFullyExpanded = True
    return node


def back_propagate(node):
    while node is not None:
        node.numVisits += 1
        if node.isFullyExpanded:
            child_Vs = [child.V * child.numVisits for child in node.children.values()]
            node.V = sum(child_Vs) / sum([child.numVisits for child in node.children.values()])
        node = node.parent


def getBestChild(node, mcts_task):
    bestValue = mcts_task.low
    bestNodes = []
    for child in node.children.values():
        nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
            2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)


def MCTS(mcts_task):
    root, node, finish = MCTS_search(mcts_task)

    if finish is not None:
        print(f'已找到最终解!\nSolution:{node.y}\n')
        return node.y, finish, root

    else:
        best_node, best_V = root.getBestV()
        print(f'在规定时间/轮次内未找到满足要求价值的解答，采用最高价值价值解答代替。\nSolution:{best_node.y}\n')
        return best_node.y, -1, root
