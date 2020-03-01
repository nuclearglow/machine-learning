def intersection(list1, list2):
    return list(set(list1) & set(list2))


def difference(list1, list2):
    return set(list1) - set(list2)
