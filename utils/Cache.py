import numpy as np
from collections import deque

class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = deque()

    def get(self, item):
        raise NotImplementedError

    def put(self, item):
        raise NotImplementedError


class FIFOCache(Cache):
    def get(self, item):
        for cached_item in self.cache:
            if np.array_equal(item, cached_item):
                return cached_item
        return None

    def put(self, item):
        # 如果缓存已满，移除最早的项
        if len(self.cache) >= self.capacity:
            self.cache.popleft()
        self.cache.append(item)


class LRUCache(Cache):
    def get(self, item):
        for idx, cached_item in enumerate(self.cache):
            if np.array_equal(item, cached_item):
                # 移动到队列末尾，表示最近使用
                self.cache.remove(cached_item)
                self.cache.append(cached_item)
                return cached_item
        return None

    def put(self, item):
        for cached_item in self.cache:
            if np.array_equal(item, cached_item):
                # 如果存在，移除旧项
                self.cache.remove(cached_item)
                break
        else:
            # 如果缓存已满，移除最久未使用的项
            if len(self.cache) >= self.capacity:
                self.cache.popleft()
        # 添加新项到队列末尾
        self.cache.append(item)
