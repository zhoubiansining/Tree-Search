from ToT.base import Node, rand_select


def BFS(tot_task):
    root = Node('')
    cur_nodes = [root]
    for depth in range(tot_task.max_depth):
        candidates = []
        for node in cur_nodes:
            for i in range(tot_task.branch):
                new_pcd = ''
                cnt = 15
                while not new_pcd and cnt:
                    new_pcd = tot_task.get_next_step(node.y, node.depth + 1)
                    cnt -= 1
                if not new_pcd:
                    continue

                node = node.append_children(new_pcd)
                child = node.children[i]
                value = tot_task.get_step_value(child.y)
                child.update_value(value)
                child.visit_sequence = tot_task.node_count
                tot_task.update_count()
                candidates.append(child)

        if not candidates:
            break
        ranked_candidates = sorted(candidates, key=lambda item: item.V, reverse=True)
        if ranked_candidates[0].V >= tot_task.end_gate:
            print('已找到最终解!\n')
            ranked_candidates[0].final_ans_flag = 1
            return ranked_candidates[0].y, root

        if tot_task.select_method == 'greedy':
            cur_nodes = ranked_candidates[:min(tot_task.select_branch, tot_task.branch, len(ranked_candidates))]

        else:
            idx_list = []
            cur_nodes = []
            for j in range(min(tot_task.select_branch, tot_task.branch)):
                idx, node = rand_select(ranked_candidates, [item.V for item in ranked_candidates])
                if idx not in idx_list:
                    idx_list.append(idx)
                    cur_nodes.append(node)
            cur_nodes = sorted(cur_nodes, key=lambda item: item.V, reverse=True)

    print('未找到满足要求价值的解答，采用最高价值价值解答代替。\n')
    max_node, max_V = root.getBestV()
    max_node.final_ans_flag = 1
    return max_node.y, root
