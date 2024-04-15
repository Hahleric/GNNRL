from collections import OrderedDict

from utils.cache_utils import cache_hit_ratio


class BaseCache():
    """
    cache_base
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.requests = 0

    def get(self, key):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def put(self, key, value):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def hit_ratio(self):
        """ Calculate and return the cache hit ratio. """
        if self.requests == 0:
            return 0  # Prevent division by zero
        print(self.requests, self.hits, self.misses)
        return self.hits / self.requests

class LRUCache(BaseCache):
    def get(self, key):
        self.requests += 1
        if key not in self.cache:
            self.misses += 1
            return -1
        else:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value


class FIFOCache(BaseCache):
    def get(self, key):
        self.requests += 1
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return -1

    def put(self, key, value):
        if key not in self.cache:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

class MRUCache(BaseCache):
    def get(self, key):
        self.requests += 1
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return -1

    def put(self, key, value):
        if key in self.cache:
            # Move the key to the end to mark it as the most recently used before potentially removing it
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove the most recently used item which is at the end of the cache
            self.cache.popitem(last=True)
        self.cache[key] = value

class RegularEnvironment:
    def __init__(self, cache_size, popular_files, request_dataset):
        self.cache_size = cache_size
        self.request_dataset = request_dataset
        self.popular_files = popular_files
        self.lru_cache = LRUCache(cache_size)
        self.fifo_cache = FIFOCache(cache_size)
        self.mru_cache = MRUCache(cache_size)
        self.cache_items = []
        self.request_num = 0
        self.cache_hit_ratio = []

    def calculate_base_policy(self):
        print("Calculating Base Policy")
        print("Request Dataset: ", self.request_dataset.shape)
        for i in range(len(self.popular_files)):
            self.lru_cache.put(i, self.popular_files[i])
            self.fifo_cache.put(i, self.popular_files[i])
            self.mru_cache.put(i, self.popular_files[i])
        for i in range(self.request_dataset.shape[0]):
            for j in range(self.request_dataset.shape[1]):
                self.lru_cache.get(self.request_dataset[i][j])
                self.fifo_cache.get(self.request_dataset[i][j])
                self.mru_cache.get(self.request_dataset[i][j])
        self.cache_hit_ratio.append([self.lru_cache.hit_ratio(), self.fifo_cache.hit_ratio(), self.mru_cache.hit_ratio()])
        print("LRU Cache Hit Ratio: ", self.lru_cache.hit_ratio())
        print("FIFO Cache Hit Ratio: ", self.fifo_cache.hit_ratio())
        print("MRU Cache Hit Ratio: ", self.mru_cache.hit_ratio())
        return self.cache_hit_ratio


