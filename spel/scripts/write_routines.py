"""OpenACC directive helpers for the generated unit-test sources."""


def acc_clause_lines(directive: str, entries: list[str], tabs: str = "") -> list[str]:
    """`!$acc <directive>(e1, ..., eN)` split over `!$acc&` lines, with no trailing comma."""
    if not entries:
        return []
    lines = [f"{tabs}!$acc {directive}( &\n"]
    lines.extend(f"{tabs}!$acc&   {name}, &\n" for name in entries[:-1])
    lines.append(f"{tabs}!$acc&   {entries[-1]})\n")
    return lines


def acc_update_device_lines(variables, tabs: str) -> list[str]:
    entries = []
    for var in variables:
        dim_str = ",".join(":" for _ in range(var.dim))
        entries.append(f"{var.name}({dim_str})" if dim_str else var.name)
    lines = acc_clause_lines("update device", entries, tabs)
    return lines + ["\n"] if lines else lines
