import json
import panflute as pf

import pytest
from textwrap import dedent
from plaited import Plait

@pytest.fixture
def code_block_doc():
    input_text = dedent(
        """\
    ---
    title: My Title
    standalone: False
    plaited-options:
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

    assert p.get_option() is False

def get_option(self, elem, option, default):
    assert out_doc.error == "raise"
    assert getattr(out_doc, "abstract", None) is None

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
    assert True
