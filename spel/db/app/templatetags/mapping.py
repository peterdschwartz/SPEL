from django import template

register = template.Library()


@register.filter
def get_item(d, key):
    """Safe dict lookup for templates: {{ mydict|get_item:key }}"""
    if isinstance(d, dict):
        return d.get(key)
    return None
