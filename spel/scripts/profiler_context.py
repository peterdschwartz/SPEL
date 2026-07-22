import contextlib
import cProfile
import pstats


@contextlib.contextmanager
def profile_ctx(enabled: bool, section: str):
    if enabled:
        pr = cProfile.Profile()
        pr.enable()
        try:
            yield pr
        finally:
            pr.disable()
            print("===================== " + section + " ==================")
            stats = pstats.Stats(pr)
            stats.strip_dirs().sort_stats("cumtime").print_stats(10)
    else:
        # When disabled, yield a dummy value.
        yield None
