import io
import sys
import json
from contextlib import redirect_stdout

import pytest
import panflute as pf
from plaited.plaited import main


@pytest.fixture
def basic_code_test():
    test_doc = '#a\nb\n```python\nprint("c")\n```\nd'

    expected_input = pf.convert_text(test_doc, output_format="json").encode("utf-8")

    # leave out the pandoc-api-version key, because it will vary by environment
    expected_output = {
        "meta": {"tables": {"t": "MetaBool", "c": True}},
        "blocks": [
            {
                "t": "Para",
                "c": [
                    {"t": "Str", "c": "#a"},
                    {"t": "SoftBreak"},
                    {"t": "Str", "c": "b"},
                ],
            },
            {"t": "CodeBlock", "c": [["", ["python"], []], 'print("c")']},
            {"t": "LineBlock", "c": [[{"t": "Str", "c": "c"}]]},
            {"t": "Para", "c": [{"t": "Str", "c": "d"}]},
        ],
    }

    return {"input": expected_input, "output": expected_output}


def test_main(basic_code_test):
    # this will pretend to be sys.stdin.buffer
    dummy_reader = io.BufferedReader(io.BytesIO(basic_code_test["input"]))

    # sys.stdin.buffer is a read-only attribute, but I want to overwrite it
    # so, here is a dummy class to replace sys.stdin entirely
    # what could go wrong?
    class DummyStdin:
        buffer = dummy_reader

    # no-one will notice...
    sys.stdin = DummyStdin()

    # ...heh but that's not all
    # I want to also trick panflute into writing out into this test
    dummy_stdout_buffer = io.BytesIO()
    with redirect_stdout(io.TextIOWrapper(dummy_stdout_buffer)) as f:
        main()

    # parse the output as json, and create a dictionary
    json_output = json.loads(dummy_stdout_buffer.getvalue().decode("utf-8"))

    # remove pandoc-api-version, this varies by environment
    del json_output["pandoc-api-version"]

    assert json_output == basic_code_test["output"]
