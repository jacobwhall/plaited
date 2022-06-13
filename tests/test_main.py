import pytest
import panflute as pf

from plaited import Plait
from plaited.main import is_code


def test_is_code():
    code_inline_elem = pf.Code(
        text='print("a")', identifier="abc", classes=["python"], attributes={}
    )
    assert is_code(code_inline_elem)

    code_block_elem = pf.CodeBlock(
        text='print("a")', identifier="abc", classes=["python"], attributes={}
    )
    assert is_code(code_block_elem)

def test_bad_is_code():
    code_no_class_elem = pf.Code(
        text='print("a")', identifier="abc", classes=[], attributes={}
    )
    assert not is_code(code_no_class_elem)

