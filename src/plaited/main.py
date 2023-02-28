import os
import sys
import copy
import base64
import mimetypes
from queue import Empty
from distutils.util import strtobool

import panflute as pf
from nbconvert.utils.base import NbConvertBase
from jupyter_client import MultiKernelManager, KernelClient
from jupyter_client.kernelspec import NoSuchKernel


class Plait:
    """
    Class that executes the code blocks within ("plaits") a single document

    use_prompt : bool, default ``False``
        Whether to use prompt.

    results : str, default ``'default'``
        * ``'default'``: default behaviour
        * ``'pandoc'``: same as 'default' but plain text is parsed via Pandoc:
          if the output is a stdout message that is
          not warning/error or if it has ``'text/plain'`` key.
          Pandoc setings can be set like
          ``{results='pandoc -f markdown-link_attributes --flag'}``
          Markdown outputs are also parsed by Pandoc
          (with appropriate settigns).
        * ``'hide'``: evaluate chunk but hide results

    # Document or Cell
    warning: bool = True
    error: str="continue" ("continue" or "raise")
    prompt: Optional[str]=None
    echo: bool = True
    eval: bool = True
    """

    def __init__(self, doc: pf.Doc, name="p_files"):
        """
        Parameters
        ----------
        name : str
            Name of directory for supporting files
        """

        if not hasattr(doc, "result") or doc.result not in ["pandoc", "hide"]:
            doc.result = "default"

        self.multi_kernel_manager = MultiKernelManager()
        self.name = name
        self.resource_dir = f"{name}_files"

        self.doc = doc

        self.lang_map = {"py": "python", "py2": "python2", "r": "ir"}
        if "kernels-map" in self.doc.metadata:
            self.lang_map.update(self.doc.metadata["kernels-map"])

        self.doc.metadata["tables"] = True
        self.doc.error = "default"

        self.untitled_count = 1

    def __getattr__(self, attr):
        return getattr(self.doc, attr)

    def get_lang_from_name(self, name):
        try:
            return self.lang_map[name]
        except KeyError:
            return str(name)

    def get_kernel_client(self, kernel_name) -> KernelClient:
        """
        Get a kernel from ``kernel_managers`` by ``kernel_name``,
        creating it if needed.

        Parameters
        ----------
        kernel_name : str

        Returns
        -------
        KernelClient
        """
        try:
            return self.multi_kernel_manager.get_kernel(kernel_name).client()
        except KeyError:
            try:
                km_name = self.multi_kernel_manager.start_kernel(
                    kernel_name=kernel_name, kernel_id=kernel_name
                )
            except NoSuchKernel:
                print(
                    f'No kernel found with name "{kernel_name}", skipping',
                    file=sys.stderr,
                )
                return None
            kc = self.multi_kernel_manager.get_kernel(km_name).client()
            if kernel_name == "python":
                initialize_python_kernel(kc)
            return kc

    def shutdown_all_kernels(self):
        for uuid in self.multi_kernel_manager.list_kernel_ids():
            km = self.multi_kernel_manager.get_kernel(uuid)
            km.shutdown_kernel(now=True)

    def get_option(self, elem, option, default) -> bool:
        value = pf.get_option(
            elem.attributes, option, self.doc, ("plaited-options." + option), default
        )
        return bool(strtobool(str(value)))

    def filter_code_blocks(self, elem, doc):
        # is this a code element, and does it have at least one class?
        if (
            isinstance(elem, pf.Code) or isinstance(elem, pf.CodeBlock)
        ) and elem.classes:
            # should we try executing the code in this element?
            if "eval" not in elem.attributes or elem.attributes["eval"]:
                # first class of the element should be the name of the language
                lang = elem.classes[0]
                kc = self.get_kernel_client(self.get_lang_from_name(lang))
                # did we successfully obtain a kernel client?
                if kc is not None:
                    """
                    # Can this code block be executed?
                    Must be Code or a CodeBlock, and must not have ``eval=False`` in the block
                    arguments, and ``lang`` (kernel_name) must be specified and not None
                    """
                    messages = run_code(elem.text, kc)
                    del kc
                else:
                    messages = []

            # determine target base class of output elements
            if isinstance(elem, pf.CodeBlock):
                target_base_class = pf.Block
                echo_default = True
            elif isinstance(elem, pf.Code):
                target_base_class = pf.Inline
                echo_default = False

            out_elements = []

            # decide whether or not to echo code
            if self.get_option(elem, "echo", echo_default):
                out_elements.append(elem)

            # determine if code output should be inserted into AST
            if (
                bool(messages)
                and (messages[0] is not None)
                and (
                    "results" not in elem.attributes
                    or elem.attributes["results"] != "hide"
                )
            ):
                for e in self.wrap_output(elem, messages):
                    if not isinstance(e, list):
                        e = [e]
                    for i in e:
                        if isinstance(i, target_base_class):
                            out_elements.append(i)

            # return output elements, to replace original element
            return out_elements

    def plait_ast(self) -> pf.Doc:
        """
        Main function for filter

        Returns
        -------
        doc : panflute Document
        """
        self.doc_actions = []

        self.doc.walk(self.filter_code_blocks, self.doc)

        for action in self.doc_actions:
            action[0].content.insert(*action[1])

        # force all Jupyter kernels to shutdown ASAP
        self.shutdown_all_kernels()

        return self.doc

    def wrap_output(self, elem, messages) -> list:
        """
        Wrap the messages of a code-block.

        Parameters
        ----------
        elem : code element that was executed
        messages : list of dicts

        Returns
        -------
        output_elements : list

        Notes
        -----
        Messages with mimetypes (e.g. matplotlib figures)
        are outputted using Jupyter's display priority.
        See ``NbConvertBase.display_data_priority``.
        """

        md_format, md_extra_args = "markdown", None

        # messsage_pairs can come from stdout or the io stream (maybe others?)

        output_messages = [
            x for x in messages if not (x["msg_type"] == "execute_input")
        ]
        display_messages = [
            x for x in output_messages if not is_stdout(x) and not is_stderr(x)
        ]

        out_elems = []
        LB_contents = []

        # Handle all stdout first...
        for message in output_messages:
            is_warning = is_stderr(message)  # and self.get_option("warning", attrs)
            if is_stdout(message) or is_warning:
                text = message["content"]["text"]
                if text.strip() != "":
                    for output in plain_output(elem, text):
                        LB_contents.append(output)
        if len(LB_contents) > 0:
            for element in LB_contents:
                out_elems.append(element)

        priority = list(enumerate(NbConvertBase().display_data_priority))
        priority.append((len(priority), "application/javascript"))
        order = dict((x[1], x[0]) for x in priority)

        for message in display_messages:
            if message["header"]["msg_type"] == "error":
                error = self.doc.error
                if error == "raise":
                    raise Exception(message["content"]["traceback"])
                LB_contents.append(plain_output(elem, message["content"]["traceback"]))
            else:
                all_data = message["content"]["data"]
                """
                if not all_data:  # some R output
                    # results = self.get_option('results', attrs)
                    continue
                """
                key = min(all_data.keys(), key=lambda k: order[k])
                data = all_data[key]

                if self.doc.format in ("latex", "beamer"):
                    if "text/latex" in all_data.keys():
                        key = "text/latex"
                        data = all_data[key]
                if key == "text/plain":
                    # ident, classes, kvs
                    out_elems.append(plain_output(data))
                elif key == "text/latex":
                    out_elems.append(pf.RawBlock(data, format="latex"))
                elif key == "text/html":
                    out_elems.append(pf.RawBlock(data, format="html"))
                elif key == "application/javascript":
                    script = f"<script type=text/javascript>{data}</script>"
                    out_elems.append(pf.RawBlock(script, format="html"))
                elif key.startswith("image") or key == "application/pdf":
                    out_elems.append(self.wrap_image_output(elem, data, key))
                elif key == "text/markdown":
                    out_elems.append(tokenize_block(data, md_format, md_extra_args))
                else:
                    out_elems.append(tokenize_block(data, self.doc.format))

        return out_elems

    def wrap_image_output(self, elem, data, key):
        """
        Extra handling for images

        Parameters
        ----------
        elem : pf.Element
        data, key : str
        attrs: dict

        Returns
        -------
        pf.Para(pf.Image)
        """
        chunk_name = f"untitled_fig_{self.untitled_count}"
        self.untitled_count += 1

        # TODO: interaction of output type and standalone.
        # TODO: this can be simplified, do the file-writing in one step
        # noinspection PyShadowingNames
        def b64_encode(data):
            return base64.encodebytes(data.encode("utf-8")).decode("ascii")

        image_keys = {"width", "height"}
        """
        caption = attrs.get('fig.cap', '')
        """

        def transform_key(k):
            # fig.width -> width, fig.height -> height;
            return k.split("fig.", 1)[-1]

        # we are saving to filesystem
        ext = mimetypes.guess_extension(key)
        filepath = os.path.join(self.resource_dir, f"{chunk_name}{ext}")
        os.makedirs(self.resource_dir, exist_ok=True)
        if ext == ".svg":
            with open(filepath, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            with open(filepath, "wb") as f:
                f.write(base64.decodebytes(data.encode("utf-8")))
        # Image :: alt text (list of inlines), target
        # Image :: Attr [Inline] Target
        # Target :: (string, string)  of (URL, title)
        block = pf.Para(pf.Image(pf.Str(chunk_name), url=filepath))
        # TODO: ..., title=caption...

        return block


def format_input_prompt(prompt, code, number):
    """
    Format the actual input code-text.
    """
    if prompt is None:
        return format_ipython_prompt(code, number)
    lines = code.split("\n")
    formatted = "\n".join([prompt + line for line in lines])
    return formatted


def wrap_input_code(elem, use_prompt, prompt, execution_count, code_style=None):
    new = copy.deepcopy(elem)
    code = elem.content
    """
    if use_prompt or prompt is not None:
        new['c'][1] = format_input_prompt(prompt, code, execution_count)
    """
    if isinstance(code_style, str) and code_style != "":
        try:
            new.classes.append(code_style)
        except (KeyError, IndexError):
            pass
    return new


def tokenize_block(source: str, pandoc_format: str = "markdown") -> list:
    """
    Convert a Jupyter output to Pandoc's JSON AST
    """
    converted = pf.convert_text(source, input_format=pandoc_format)

    if isinstance(converted, list):
        if len(converted) > 1:
            return pf.Div(*converted)
        else:
            return converted[0]
    else:
        return converted


def plain_output(
    elem,
    text,
    pandoc: bool = False,
) -> list:
    if isinstance(elem, pf.CodeBlock):
        if isinstance(text, str):
            text = text.splitlines()
        if isinstance(text, list):
            return [pf.LineBlock(*[pf.LineItem(*[pf.Str(x) for x in text])])]
        else:
            raise TypeError("Plain output requires a string or list of strings.")
    else:
        return [pf.Str(x) for x in text]


def is_stdout(message):
    return message["content"].get("name") == "stdout"


def is_stderr(message):
    return message["content"].get("name") == "stderr"


def run_code(code: str, kc: KernelClient, timeout=None):
    """
    Execute a code chunk, capturing the output.

    Parameters
    ----------
    code : str
    kc : KernelClient
    timeout : int

    Returns
    -------
    outputs : List

    Notes
    -----
    See https://github.com/jupyter/nbconvert/blob/master/nbconvert
      /preprocessors/execute.py
    """
    kc.wait_for_ready()

    msg_id = kc.execute(code)
    messages = []

    while True:  # until message that corresponds to executed code
        # We've already waited for execute_reply, so all output
        # should already be waiting. However, on slow networks, like
        # in certain CI systems, waiting < 1 second might miss messages.
        # So long as the kernel sends a status:idle message when it
        # finishes, we won't actually have to wait this long, anyway.
        msg = kc.get_iopub_msg(timeout=4)

        if msg["parent_header"]["msg_id"] != msg_id:
            # not an output from our execution
            continue

        msg_type = msg["msg_type"]
        content = msg["content"]

        if msg_type == "status":
            if content["execution_state"] == "idle":
                break
            else:
                continue

        elif msg_type in (
            "execute_input",
            "execute_result",
            "display_data",
            "stream",
            "error",
        ):
            # Keep `execute_input` just for execution_count if there's
            # no result
            messages.append(msg)

        elif msg_type == "clear_output":
            messages = []

    return messages


def initialize_python_kernel(kc):
    # TODO: set_matplotlib_formats takes *args
    # TODO: do as needed? Push on user?
    # valid_formats = ["png", "jpg", "jpeg", "pdf", "svg"]
    init_code = """\
    %colors NoColor
    try:
        %matplotlib inline
    except:
        pass
    try:
        import pandas as pd
        pd.options.display.latex.repr = True
    except:
        pass
    """
    kc.execute(init_code, store_history=False)
