import json
from .main import Plait
import psutil
import traceback
import panflute as pf
import io

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


def pandoc_filter(json_ast: str, **kwargs) -> str:
    """
    Changes Pandoc JSON AST string
    """
    ast = json.loads(json_ast)
    f = io.StringIO(json_ast)
    plaiter = Plait(pf.load(f), **kwargs)

    def work():
        nonlocal ast
        ast = plaiter.plait_ast()

    safe_spawn(work)
    with io.StringIO() as out:
        pf.dump(ast, out)
        output = out.getvalue()
    return output
