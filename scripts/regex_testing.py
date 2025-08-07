import re

from scripts.config import _bc


def test_regex(input: list[str], expected: list[bool], pattern: re.Pattern) -> None:
    """
    Function that applies pattern on each input str. Expected only specifies if
    something should match or not.
    """
    for n, test in enumerate(input):
        m_ = pattern.search(test)
        if not m_ and expected[n]:
            print(_bc.FAIL + f"Error No Match: {test}" + _bc.ENDC)
        elif m_ and not expected[n]:
            print(_bc.FAIL + f"Error Matched: {m_.group()} in {test}")
        elif not m_ and not expected[n]:
            print(_bc.OKGREEN + f"No Match: {test}")
        elif m_ and expected[n]:
            print(_bc.OKGREEN + f"Matched: {m_.group()} in {test}")
    return
