import copy


class treeNode(object):
    def __init__(self, pcd, parent=None, depth=0):
        self.pcd = pcd  # str
        self.y = ''  # str
        self.parent = parent  # treeNode
        self.numVisits = 0  # int
        self.V = 0  # float
        self.children = {}  # dict{str:treeNode}
        self.depth = depth  # int
        self.isFullyExpanded = False  # expanded
        self.visit_sequence = 0
        self.final_ans_flag = 0
        self.reflection = ''
        self.isTerminal = False  # value acceptable

    def __str__(self):
        s = ["numVisits: %d" % self.numVisits, f'V:{self.V}', "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))

    def append_children(self, new_pcd: str):
        node = treeNode(new_pcd, self, self.depth + 1)
        node.update_y_from_parent()
        self.children.update({new_pcd: node})
        return self

    def update_y_from_parent(self):
        if self.parent is None:
            self.y = self.pcd
        else:
            self.y = self.parent.y + self.pcd

    def update_value(self, value):
        self.V = value

    def update_reflection(self, reflection):
        self.reflection = reflection

    def getBestV(self):  # 获取子树最大价值节点
        if not self.isFullyExpanded:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children.values():
            subNode, subValue = child.getBestV()
            if subValue > max_V:
                max_V = subValue
                max_node = subNode
        return max_node, max_V
