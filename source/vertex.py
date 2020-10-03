import numpy as np

empty = np.array((), dtype = np.int64)

class Vertex(dict):
    
    def __init__(self, index, children = empty, parents = empty):
        self.children = children
        self.parents = parents
        self.index = index
        super().__init__(self)

    def __delitem__(self, key):
        value = super().pop(key)
        super().pop(value, None)

    def add_child(self, index):
        if np.any(self.children == index):
            raise ValueError('Graph does not support parallel edges, {ind} is already a child of vertex {curr}'.format(ind = index, curr = self.index))
        else:
            self.children = np.append(self.children, index)

    def add_parent(self, index):
        if np.any(self.parents == index):
            raise ValueError('Graph does not support parallel edges, {ind} is already a parent of vertex {curr}'.format(ind = index, curr = self.index))
        else:
            self.parents = np.append(self.parents, index)
    
    def __setitem__(self, key, value):
        if key in self:
            del self[self[key]]
        if value in self:
            del self[value]
        super().__setitem__(key, value)
        super().__setitem__(value, key)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"
