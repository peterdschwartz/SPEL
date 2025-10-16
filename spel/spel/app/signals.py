from django.core.cache import cache

from .calltree import Node


def _tree_key(sub_names: list[str]) -> str:
    return "'calltree:" + "|".join(sub_names)


def retrieve_cache_tree(sub_names: list[str]):
    key = _tree_key(sub_names)
    return cache.get(key, None)


def cache_tree(roots: list[Node], timeout=180):
    key = _tree_key([root.name for root in roots])
    cache.set(key, roots, timeout)
    return


# @receiver(post_save, sender=SubroutineCalltree)
# def _calltree_saved(sender, **kwargs):
#     _zap_callorder_cache()
#
#
# @receiver(post_delete, sender=SubroutineCalltree)
# def _calltree_deleted(sender, **kwargs):
#     _zap_callorder_cache()
