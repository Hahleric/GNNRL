from collections import Counter


def cache_hit_ratio(test_dataset, cache_items, request_num):
    """
    计算缓存命中率
    :param test_dataset: user_group_test[0-9] dataset
    :param cache_items: 缓存内容
    :param request_num: 请求次数
    :return: 缓存命中率
    """

    # TODO why is this here test_dataset[:, 1]? it may be a bug which should be
    # TODO generalized to other
    request_items = test_dataset[:request_num, :]
    cache_hits = 0
    for item in request_items:
        count = Counter(item)
        for cache_item in cache_items:
            cache_hits += count[cache_item]
    print("cache_hits: ", cache_hits)
    print("request_nums", request_items.shape[0] * request_items.shape[1])
    hit_ratio = cache_hits / (request_items.shape[0] * request_items.shape[1]) * 100
    # print("popular_items: ", popular_items)
    return hit_ratio


def cache_hit_ratio2(test_dataset, cache_items, cache_items2, request_num):
    """
    计算缓存命中率
    :param test_dataset: user_group_test[0-9] dataset
    :param cache_items: 缓存内容
    :return: 缓存命中率
    """
    requset_items = test_dataset[:request_num, 1]
    count = Counter(requset_items)
    CACHE_HIT_NUM = 0
    for item in cache_items:
        if item not in cache_items2:
            CACHE_HIT_NUM += count[item]
    CACHE_HIT_RATIO = CACHE_HIT_NUM / len(requset_items) * 100

    return CACHE_HIT_RATIO
