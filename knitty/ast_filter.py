import json
from .stitch.stitch import Stitch
import psutil
import traceback


# -------------------------------------------
# Pandoc JSON AST filter
# -------------------------------------------
def safe_spawn(func):
    """
    Safely run function: if func spawns child processes they are closed even on python error.

    It can be useful when calling Stitch from Atom. For some reason RTerm.exe does not close
    and Node.js isn't aware of it. So spawned Node.js process cannot exit.
    """
    # noinspection PyBroadException
    try:
        func()
    except Exception:
        traceback.print_exc()

    procs = psutil.Process().children(recursive=True)
    for p in procs:
        p.terminate()
    gone, still_alive = psutil.wait_procs(procs, timeout=50)
    for p in still_alive:
        p.kill()
        print("Killed process that was still alive after 'timeout=50' from 'terminate()' command.")


def knitty_pandoc_filter(json_ast: str, **kwargs) -> str:
    """
    Changes Pandoc JSON AST string
    """
    ast = json.loads(json_ast)
    stitcher = Stitch(**kwargs)

    def work():
        nonlocal ast
        ast = stitcher.stitch_ast(ast)

    safe_spawn(work)

    return json.dumps(ast)
