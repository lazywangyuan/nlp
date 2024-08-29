class Node():
    def _init__(self, node_name, parent_node=None, path_to_this=None):
        self.node_name = node_name  # 当前节点存储的元素值
        self.children_node_map = None  # 当前节点的子节点信息
        self.parent_node = parent_node  # 父节点引用，用于从任意一个节点出发，获取到达这里的路径。微观上，这种节点可以支持双链表
        self.is_leaf = False
        self.path_to_this = path_to_this  #

    # 是否为叶子结点
    def if_leaf(self):
        return self.is_leaf

    def add_children_node(self, a_node):
        if self.children_node_map == None: self.children_node_map = {}
        self.children_node_map[a_node.node_name] = a_node

    def set_as_leaf(self):
        self.is_leaf = True


class TrieHashMap():
    def __init__(self):
        self.root_node = Node('root')


def add_term(self, element_list):
    current_node = self.root_node
    parent_path = ""
    for element in element_list:
        if current_node.children_node_map == None or element not in current_node.children_node_map:
            new_node = Node(element)
            current_node.add_children_node(new_node)
        current_node = current_node.children_node_map[element]
        parent_path += element
    print("这个模式的末尾节点是", current_node.node_name)
    current_node.set_as_leaf()


def containsKey(self, element_list):
    current_node = self.root_nodefor
    for element in element_list:
        if element not in current_node.children_node_map:
            return False
        current_node = current_node.children_node_map[element]
        if current_node.if_leaf():
            return True
    else:
        return False
