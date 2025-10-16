"""
Functions to process the git output and show differences between commits
"""


def check_new_additions():
    from edit_files import process_for_unit_test

    files, lines = git_diff()
    needed_mods = []

    for f in files:
        sublist = process_for_unit_test(
            fname=f,
            casename="casename",
            mods=needed_mods,
            overwrite=False,
            verbose=False,
            singlefile=True,
        )

    for s in sublist:
        s.examineLoops()


def git_diff(commit="067be7cbc99cd349770d4eeb8270b58f47498dfe"):
    import re
    import subprocess as sp

    cmd = f"git show {commit}"
    log = sp.getoutput(cmd)
    log = log.split("\n")
    # use to find which files the modifications are for
    regex_gitdiff = re.compile("^diff --git ")
    # Find new set of diffs.
    regex_lines = re.compile("^@@")
    regex_ln = re.compile("[0-9]+?,[0-9]+\s*?")
    files = []
    lines = []
    for l in log:
        match_gitdiff = regex_gitdiff.search(l)
        if match_gitdiff:
            larr = l
            larr = larr.replace("diff --git", "")
            larr = larr.split()
            file = larr[0]
            file = file[1:]
            file = "." + file
            files.append(file[:])

        match_lines = regex_lines.search(l)
        if match_lines:
            l = l.split("@@")
            nums = l[1]
            m_ln = regex_ln.findall(nums)
            # Second element should be the new file
            lines.append(m_ln[1])

    print(files)
    print(lines)
    return files, lines


if __name__ == "__main__":
    import config

    config.elm_files = ""
    check_new_additions()
