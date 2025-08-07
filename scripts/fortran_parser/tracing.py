import functools


class Trace:
    enabled = False
    _trace_level = 0  # Static variable to keep the indent level
    _trace_indent_placeholder = "|  "

    @classmethod
    def ident_level(cls):
        return cls._trace_indent_placeholder * (cls._trace_level - 1)

    @classmethod
    def trace_print(cls, message: str):
        if cls.enabled:
            print(f"{cls.ident_level()}{message}")

    @classmethod
    def inc_ident(cls):
        cls._trace_level += 1

    @classmethod
    def dec_ident(cls):
        cls._trace_level -= 1

    @classmethod
    def trace(cls, msg: str):
        if cls.enabled:
            cls.inc_ident()
            cls.trace_print(f"BEGIN {msg}")
            return msg
        else:
            return ""

    @classmethod
    def untrace(cls, msg: str):
        if cls.enabled:
            cls.trace_print(f"END {msg}")
            cls.dec_ident()

    @classmethod
    def trace_decorator(cls, msg: str):
        """A decorator to wrap functions for automatic tracing."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cls.trace(msg)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    cls.untrace(msg)

            return wrapper

        return decorator
