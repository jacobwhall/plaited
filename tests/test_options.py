import json
import panflute as pf

import pytest
from textwrap import dedent
from plaited.plait.plait import Plait

"""
def pre_stitch_ast(source: str) -> dict:
        return json.loads(pf.convert_text(knitty_preprosess(source),
                                          input_format='markdown',
                                          output_format='json'))
"""

@pytest.fixture
def doc_meta():
    data = {
        "date": "2016-01-01",
        "title": "My Title",
        "author": "Jack",
        "self_contained": True,
        "standalone": False,
    }
    doc = dedent(
        """\
    ---
    title: {title}
    author: {author}
    date: {date}
    self_contained: {self_contained}
    standalone: {standalone}
    ---

    # Hi
    """
    )
    return doc.format(**data), data

"""
Test that default plaited options are as expected
"""
def test_defaults(self):
    in_text = "hello world"
    in_doc = pf.convert_text(in_text, standalone = True)
    p = Plait(in_doc)
    out_doc = p.plait_ast()
    assert out_doc.warning
    assert out_doc.error == 'continue'

def test_override():
    input_text = dedent(
        """\
    ---
    title: My Title
    standalone: False
    warning: False
    error: raise
    abstract: |
      This is the abstract.

      It consists of two paragraphs.
    ---

    # Hail and well met
    """
    )
    input_doc = pf.convert_text(input_text, standalone = True)
    p = Plait(input_doc)
    out_doc = p.plait_ast()

    assert out_doc.warning is False
    assert out_doc.error == "raise"
    assert getattr(out_doc, "abstract", None) is None

"""

@pytest.mark.parametrize(
    "key", ["title", "author", "date", "self_contained", "standalone"]
)
def test_meta(key, doc_meta):
    doc, meta = doc_meta
    p = Plait("", "html")
    p.plait(pre_stitch_ast(doc))
    result = getattr(s, key)
    expected = meta[key]
    assert result == expected
"""

"""
Test to see if captions of output figures can be set in CodeBlock classes
"""

def test_fig_cap():
    input_text = dedent(
        """\
    ```{python, fig.cap="This is a caption"}
    import matplotlib.pyplot as plt
    plt.plot(range(4), range(4))
    ```"""
    )
    input_doc = pf.convert_text(input_text, standalone=True)
    p = Plait(input_doc)
    out_doc = p.plait_ast()
    # assert result == "This is a caption"
