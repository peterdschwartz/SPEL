from pathlib import Path

import pytest

from spel.scripts.functional_unit_test import insert_at_token, with_prof_ranges
from spel.scripts.run_fut import profile_command


def test_with_prof_ranges_wraps_calls():
    calls = ["call foo(a, b)\n", "! comment\n", "  CALL Bar()\n"]
    assert with_prof_ranges(calls) == [
        "call spel_prof_push('foo')\n",
        "call foo(a, b)\n",
        "call spel_prof_pop()\n",
        "! comment\n",
        "call spel_prof_push('Bar')\n",
        "  CALL Bar()\n",
        "call spel_prof_pop()\n",
    ]


def test_call_sub_ranges_keep_call_order():
    # prepare_main inserts the calls reversed; the ranges must still bracket them
    calls = ["call first()\n", "call second()\n"]
    lines = insert_at_token(
        ["program main\n", "!#CALL_SUB\n", "end\n"],
        "!#CALL_SUB",
        with_prof_ranges(list(reversed(calls))),
    )
    body = [l.strip() for l in lines[2:-1]]
    assert body == [
        "call spel_prof_push('second')",
        "call second()",
        "call spel_prof_pop()",
        "call spel_prof_push('first')",
        "call first()",
        "call spel_prof_pop()",
    ]


def test_profile_command(monkeypatch):
    monkeypatch.setattr(
        "spel.scripts.run_fut.find_nvidia_tool", lambda name: f"/bin/{name}"
    )
    out = Path("/tmp/rep")
    assert profile_command("none", ["4"], out) == ["./build/elmtest", "4"]

    nsys = profile_command("nsys", ["4"], out, stats=True, tool_args=["--x"])
    assert nsys[:2] == ["/bin/nsys", "profile"]
    assert "--trace=cuda,nvtx,openacc,osrt" in nsys
    assert f"--output={out}" in nsys and "--stats=true" in nsys
    assert nsys[-3:] == ["--x", "./build/elmtest", "4"]

    ncu = profile_command("ncu", [], out)
    assert ncu[0] == "/bin/ncu"
    assert ncu[ncu.index("--nvtx-include") + 1] == "compute/"
    assert ncu[-1] == "./build/elmtest"

    with pytest.raises(ValueError):
        profile_command("vtune", [], out)
