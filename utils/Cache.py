class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = []

    def get(self, item):
        raise NotImplementedError

    def put(self, item):
        raise NotImplementedError


class FIFOCache(Cache):
    def get(self, item):
        if item in self.cache:
            return item
        else:
            return None

    def put(self, item):
        if len(self.cache) >= self.capacity:
            self.cache.pop(0)
        self.cache.append(item)


class LRUCache(Cache):
    def get(self, item):
        if item in self.cache:
            self.cache.remove(item)
            self.cache.append(item)
            return item
        else:
            return None

    def put(self, item):
        if item in self.cache:
            self.cache.remove(item)
        elif len(self.cache) >= self.capacity:
            self.cache.pop(0)
        self.cache.append(item)