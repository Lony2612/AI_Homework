class Stack():
    def __init__(self, max_len):
        self._array = [[0,0] for i in range(max_len)]
        self._pointer = -1
    
    def push(self, point):
        self._array[self._pointer+1] = point
        self._pointer += 1
    
    def pop(self):
        self._pointer -= 1
        return self._array[self._pointer+1]
    
    def get_length(self):
        return self._pointer+1