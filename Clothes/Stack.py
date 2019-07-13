class Stack():
    # stack
    def __init__(self, max_len):
        self._array = [[0,0] for i in range(max_len)]
        self._pointer = -1
    
    def push(self, point):
        self._array[self._pointer+1] = point
        self._pointer += 1
    
    def pop(self):
        self._pointer -= 1
        return self._array[self._pointer+1]
    
    # def get_length(self):
    #     return self._pointer+1

    # def __init__(self, max_len):
    #     self._array = []
    
    # def push(self, point):
    #     self._array.append(point)
    
    # def pop(self):
    #     return self._array.pop()
    
    # def get_length(self):
    #     return len(self._array)

    # # list
    # def __init__(self, max_len):
    #     self._array = []
    
    # def push(self, point):
    #     self._array.append(point)
    
    # def pop(self):
    #     tail = self._array[-1]
    #     self._array.remove(self._array[-1])
    #     return tail
    
    # def get_length(self):
    #     return len(self._array)-1

    # # set
    # def __init__(self, max_len):
    #     self._array = set()
    
    # def push(self, point):
    #     self._array.add(point)
    
    # def pop(self):
    #     return self._array.pop()
    
    # def get_length(self):
    #     return len(self._array)-1
    