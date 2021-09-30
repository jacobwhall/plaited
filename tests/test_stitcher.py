import os
import os.path as p
import json
import uuid
import shutil
import datetime
from textwrap import dedent

import pytest
import panflute as pf
from traitlets import TraitError

from knitty.api import knitty_preprosess
import knitty.stitch.stitch as R

HERE = p.dirname(__file__)


def pre_stitch_ast(source: str) -> dict:
        return json.loads(pf.convert_text(knitty_preprosess(source),
                                          input_format='markdown',
                                          output_format='json'))


@pytest.fixture(scope='module')
def global_python_kernel():
    """
    A python kernel anyone can use.
    """
    return R.kernel_factory('python')


@pytest.fixture(scope='function')
def clean_python_kernel(global_python_kernel):
    """
    Takes ``global_python_kernel`` and resets all variables,
    returning the clean kernel.
    """
    R.run_code('%reset -f', global_python_kernel)
    return global_python_kernel


@pytest.fixture
def clean_name():
    name = str(uuid.uuid1())
    yield name
    shutil.rmtree(name + '_files')


@pytest.fixture
def clean_stdout():
    yield
    shutil.rmtree('stdout_files')


@pytest.fixture
def document_path():
    "Path to a markdown document"
    return p.join(HERE, 'data', 'small.md')


@pytest.fixture
def document():
    "In-memory markdown document"
    with open(p.join(HERE, 'data', 'small.md')) as f:
        doc = f.read()
    return doc


@pytest.fixture
def as_json(document):
    "JSON representation of the markdown document"
    return json.loads(pf.convert_text(document, output_format='json',
                                      input_format='markdown'))


@pytest.fixture(params=['python', 'R'], ids=['python', 'R'])
def code_block(request):
    if request.param == 'python':
        code = 'def f(x):\n    return x * 2\n\nf(2)'
    elif request.param == 'R':
        code = 'f <- function(x){\n  return(x * 2)\n}\n\nf(2)'
    block = {'t': 'CodeBlock',
             'c': [['', ['{}'.format(request.param)], []],
                   code]}
    return block


@pytest.fixture
def python_kp():
    return R.kernel_factory('python')


class TestTesters:

    @pytest.mark.parametrize('block, expected', [
        ({'t': 'CodeBlock',
          'c': [['', ['{python}'], []],
                'def f(x):\n    return x * 2\n\nf(2)']}, True),
        ({'c': [{'c': 'With', 't': 'Str'},
                {'c': [], 't': 'Space'},
                {'c': 'options', 't': 'Str'}], 't': 'Para'}, False),
    ])
    def test_is_code_block(self, block, expected):
        result = R.is_code_block(block)
        assert result == expected

    @pytest.mark.parametrize('output, attrs, expected', [
        ([], {}, False),
        ([None], {}, False),
        ([{'text/plain': '4'}], {}, True),
        ([{'text/plain': '4'}], {'results': 'hide'}, False),
    ])
    def test_is_stitchable(self, output, attrs, expected):
        result = R.is_stitchable(output, attrs)
        assert result == expected

    @pytest.mark.parametrize('block, lang, attrs, expected', [
        ({'t': 'CodeBlock',
          'c': [['', ['{python}'], []],
                'def f(x):\n    return x * 2\n\nf(2)']}, 'python', {}, True),

        ({'c': [{'c': 'With', 't': 'Str'},
         {'c': [], 't': 'Space'},
         {'c': 'options', 't': 'Str'}], 't': 'Para'}, '', {}, False),

        ({'t': 'CodeBlock',
          'c': [['', ['{r}'], []],
                '2+2']}, 'r', {'eval': False}, False),
    ])
    def test_is_executable(self, block, lang, attrs, expected):
        result = R.is_executable(block, lang, attrs)
        assert result is expected

    @pytest.mark.parametrize('message, expected', [
        ({'content': {'name': 'stdout'}}, True),
        ({'content': {'name': 'stderr'}}, False),
        ({'content': {}}, False),
    ])
    def test_is_stdout(self, message, expected):
        result = R.is_stdout(message)
        assert result == expected

    @pytest.mark.parametrize('message, expected', [
        ({'content': {'name': 'stdout'}}, False),
        ({'content': {'name': 'stderr'}}, True),
        ({'content': {}}, False),
    ])
    def test_is_stderr(self, message, expected):
        result = R.is_stderr(message)
        assert result == expected

    @pytest.mark.parametrize('message, expected', [
        ({'msg_type': 'execute_input'}, True),
        ({'msg_type': 'idle'}, False),
    ])
    def test_is_execute_input(self, message, expected):
        result = R.is_execute_input(message)
        assert result == expected


class TestKernelArgs:

    @pytest.mark.parametrize('block, expected', [
        ({'t': 'CodeBlock', 'c': [['', ['python'], []], 'foo']}, 'python'),
        ({'t': 'CodeBlock', 'c': [['', ['ir'], ['foo']], 'foo']}, 'ir'),
        ({'t': 'CodeBlock', 'c': [['', ['ir'], [['foo', 'bar']]], 'foo']},
         'ir'),
    ])
    def test_extract_kernel_name(self, block, expected):
        result = R.extract_kernel_name(block)
        assert result == expected

    @pytest.mark.parametrize('block, lang, attrs, expected', [
        ({'t': 'CodeBlock',
          'c': [['', ['{python}'], []],
                'def f(x):\n    return x * 2\n\nf(2)']}, 'python', {}, True),

        ({'c': [{'c': 'With', 't': 'Str'},
         {'c': [], 't': 'Space'},
         {'c': 'options', 't': 'Str'}], 't': 'Para'}, '', {}, False),

        ({'t': 'CodeBlock',
          'c': [['', ['{r}'], []],
                '2+2']}, 'r', {'eval': False}, False),
    ])
    def test_is_executable(self, block, lang, attrs, expected):
        result = R.is_executable(block, lang, attrs)
        assert result is expected

    @pytest.mark.parametrize('code_block, expected', [
        ({'c': [['', ['python'], []], '3'], 't': 'CodeBlock'},
         (('python', None), {})),
        ({'c': [['', ['python', 'name'], []], '3'], 't': 'CodeBlock'},
         (('python', 'name'), {})),
        ({'c': [['', ['r', 'n'], [['foo', 'bar']]], '3'], 't': 'CodeBlock'},
         (('r', 'n'), {'foo': 'bar'})),
        ({'c': [['', [], [['foo', 'bar']]], '4'], 't': 'CodeBlock'},
         ((None, None), {'foo': 'bar'})),
    ])
    def test_parse_kernel_arguments(self, code_block, expected):
        result = R.parse_kernel_arguments(code_block)
        assert result == expected


class TestFormatters:

    def test_format_input(self):
        code = '2 + 2'
        expected = '>>> 2 + 2'
        result = R.format_input_prompt('>>> ', code, None)
        assert result == expected

    def test_format_input_multi(self):
        code = dedent('''\
            def f(x):
                return x''')
        expected = dedent('''\
            >>> def f(x):
            >>>     return x''')
        result = R.format_input_prompt('>>> ', code, None)
        assert result == expected

    def test_format_ipython_input(self):
        code = '2 + 2'
        expected = 'In [1]: 2 + 2'
        result = R.format_ipython_prompt(code, 1)
        assert result == expected

    def test_format_input_none(self):
        code = 'abcde'
        result = R.format_ipython_prompt(code, None)
        assert result == code

    def test_format_ipython_input_multi(self):
        code = dedent('''\
        def f(x):
            return x + 2

        f(2)
        ''').strip()
        expected = dedent('''\
        In [10]: def f(x):
            ...:     return x + 2
            ...:
            ...: f(2)
        ''').strip()
        result = R.format_ipython_prompt(code, 10)
        assert result == expected

    def test_wrap_input__code(self):
        block = {'t': 'code', 'c': ['a', ['b'], 'c']}
        result = R.wrap_input_code(block, True, None, None)
        assert block is not result

    @pytest.mark.parametrize('messages,expected', [
        ([{'content': {'data': {},
                       'execution_count': 4},
           'header': {'msg_type': 'execute_result'}}],
         4),

        ([{'content': {'execution_count': 2},
           'header': {'msg_type': 'execute_input'}}],
         2),

        ([{'content': {'data': {'text/plain': 'foo'}}},
          {'content': {'execution_count': 2}}],
         2),
    ])
    def test_extract_execution_count(self, messages, expected):
        assert R.extract_execution_count(messages) == expected

    @pytest.mark.parametrize('output, message, expected', [
        ([{'text/plain': '2'}],
         {'content': {'execution_count': '1'}},
         {'t': 'Div', 'c': (['', ['output'], []],
                            [{'t': 'Para',
                              'c': [{'t': 'Str',
                                     'c': 'Out[1]: 2'}]}])}),
    ])
    @pytest.mark.xfail
    def test_wrap_output(self, output, message, expected):
        result = R.Stitch('stdout', 'html').wrap_output(output, message, {})
        assert result == expected


class TestIntegration:

    def test_from_file(self, document_path, clean_stdout):
        with open(document_path, 'r', encoding='utf-8') as f:
            R.Stitch('stdout', 'html', self_contained=False).stitch_ast(pre_stitch_ast(f.read()))

    def test_from_source(self, document, clean_stdout):
        R.Stitch('stdout', 'html', self_contained=False).stitch_ast(pre_stitch_ast(document))

    @pytest.mark.parametrize("to, value", [
        ("html", "data:image/png;base64,"),
        ("latex", 'unnamed_chunk_0'),  # TODO: chunk name
    ])
    def test_image(self, to, value, global_python_kernel):
        code = dedent('''\
        ```{python}
        %matplotlib inline
        import matplotlib.pyplot as plt
        plt.plot(range(4), range(4))
        plt.title('Foo — Bar');  # That's an em dash
        ```
        ''')
        result = R.Stitch('foo', to).stitch_ast(pre_stitch_ast(code))
        blocks = result['blocks']
        assert blocks[1]['c'][0]['t'] == 'Image'

    def test_image_chunkname(self):
        code = dedent('''\
        ```{python, chunk}
        %matplotlib inline
        import matplotlib.pyplot as plt
        plt.plot(range(4), range(4));
        ```
        ''')
        result = R.Stitch('foo', 'latex', standalone=False).stitch_ast(pre_stitch_ast(code))
        blocks = result['blocks']
        assert 'chunk' in blocks[1]['c'][0]['c'][0][0]

    def test_image_attrs(self):
        code = dedent('''\
        ```{python, chunk, fig.width=10, fig.height=10px}
        %matplotlib inline
        import matplotlib.pyplot as plt
        plt.plot(range(4), range(4));
        ```
        ''')
        result = R.Stitch('foo', 'html', standalone=False).stitch_ast(pre_stitch_ast(code))
        blocks = result['blocks']
        attrs = blocks[1]['c'][0]['c'][0][2]
        assert ('width', '10') in attrs
        assert ('height', '10px') in attrs

    def test_image_no_self_contained(self, clean_python_kernel, clean_name):
        code = dedent('''\
        ```{python}
        %matplotlib inline
        import matplotlib.pyplot as plt
        plt.plot(range(4))
        ```
        ''')
        s = R.Stitch(clean_name, 'html', self_contained=False)
        s._kernel_pairs['python'] = clean_python_kernel
        result = s.stitch_ast(pre_stitch_ast(code))
        blocks = result['blocks']
        expected = ('{}_files/unnamed_chunk_0.png'.format(clean_name),
                    r'{}_files\unnamed_chunk_0.png'.format(clean_name))
        result = blocks[-1]['c'][0]['c'][2][0]
        assert result in expected

    @pytest.mark.parametrize('fmt', ['png', 'svg', 'pdf'])
    def test_image_no_self_contained_formats(self, clean_python_kernel,
                                             clean_name, fmt):
        code = dedent('''\
        ```{{python}}
        %matplotlib inline
        from IPython.display import set_matplotlib_formats
        import numpy as np
        import matplotlib.pyplot as plt
        set_matplotlib_formats('{fmt}')

        x = np.linspace(-np.pi / 2, np.pi / 2)
        plt.plot(x, np.sin(x))
        plt.plot(x, np.cos(x))
        ```
        ''').format(fmt=fmt)
        s = R.Stitch(clean_name, 'html', self_contained=False)
        s._kernel_pairs['python'] = clean_python_kernel
        s.stitch_ast(pre_stitch_ast(code))
        expected = p.join(clean_name + '_files',
                          'unnamed_chunk_0.' + fmt)
        assert p.exists(expected)

    @pytest.mark.parametrize('warning, length', [
        (True, 3),
        (False, 2),
    ])
    def test_warning(self, clean_python_kernel, warning, length):
        code = dedent('''\
        ```{python}
        import warnings
        warnings.warn("Hi")
        2
        ```
        ''')
        r = R.Stitch('foo', 'html', warning=warning)
        r._kernel_pairs['python'] = clean_python_kernel
        result = r.stitch_ast(pre_stitch_ast(code))
        assert len(result['blocks']) == length

    @pytest.mark.parametrize('to', ['latex', 'beamer'])
    def test_rich_output(self, to, clean_python_kernel):
        code = dedent('''\
        ```{python}
        import pandas as pd
        pd.options.display.latex.repr = True
        pd.DataFrame({'a': [1, 2]})
        ```
        ''')
        stitch = R.Stitch('foo', to)
        stitch._kernel_pairs['python'] = clean_python_kernel
        blocks = stitch.stitch_ast(pre_stitch_ast(code))['blocks']
        result = blocks[1]['c'][1]
        assert '\\begin{tabular}' in result

    def test_error_raises(self):
        s = R.Stitch('stdout', 'html', error='raise')
        code = dedent('''\
        ```{python}
        1 / 0
        ```
        ''')
        with pytest.raises(R.KnittyError):
            s.stitch_ast(pre_stitch_ast(code))

        s.error = 'continue'
        s.stitch_ast(pre_stitch_ast(code))

    @pytest.mark.parametrize('to', [
        'html', 'latex', 'docx',
    ])
    def test_ipython_display(self, clean_python_kernel, to):
        s = R.Stitch('stdout', to)
        code = dedent('''\
        from IPython import display
        import math
        display.Markdown("$\\alpha^{pi:1.3f}$".format(pi=math.pi))
        ''')
        messages = R.run_code(code, clean_python_kernel)
        wrapped = s.wrap_output('', messages, None)[0]
        assert wrapped['t'] == 'Para'
        assert wrapped['c'][0]['c'][0]['t'] == 'InlineMath'


class TestStitcher:
    def test_error(self):
        s = R.Stitch('stdout', 'html')
        assert s.error == 'continue'
        s.error = 'raise'
        assert s.error == 'raise'

        with pytest.raises(TraitError):
            s.error = 'foo'

    def test_getattr(self):
        s = R.Stitch('stdout', 'html')
        assert getattr(s, 'fig.width') is None
        assert s.fig.width is None
        with pytest.raises(AttributeError):
            assert getattr(s, 'foo.bar')

        with pytest.raises(AttributeError):
            assert getattr(s, 'foo')

    def test_has_trait(self):
        s = R.Stitch('stdout', 'html')
        assert s.has_trait('fig.width')
        assert not s.has_trait('fake.width')
        assert not s.has_trait('fig.fake')


def test_empty_message():
    # GH 52
    messages = [
        {'parent_header': {
            'username': 't', 'session': 'a',
            'msg_type': 'execute_request', 'msg_id': '3',
            'date': datetime.datetime(2016, 9, 27, 7, 20, 13, 790481),
            'version': '5.0'
        }, 'metadata': {}, 'buffers': [], 'msg_type': 'display_data',
           'header': {'username': 't', 'session': 'a',
                      'msg_type': 'display_data', 'version': '5.0',
                      'date': '2016-09-27T07:20:17.461893',
                      'msg_id': '6'},
           'content': {'metadata': {}, 'data': {}}, 'msg_id': '6'}
    ]
    s = R.Stitch('foo', 'html')
    result = s.wrap_output('bar', messages, {})
    assert result == []
