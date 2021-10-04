# plaited 🪢

plaited is a [Pandoc](https://pandoc.org/) [filter](https://pandoc.org/filters.html) that uses [Jupyter](https://jupyter.org/) [kernels](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels) to execute code inline. It's a flexible tool for writing and publishing code notebooks in a variety of languages and formats. It is a fork of [Knitty](https://github.com/kiwi0fruit/knitty), which is a fork of [Stitch](https://github.com/pystitch/stitch), which used code from [knitpy](https://github.com/jankatins/knitpy) and nbconvert.

## Installation

_coming soon!_


## Getting Started

plaited is plug-and-play with Pandoc:

```bash
pandoc --filter plaited -o out.html input.md
```

## Contributing

You are more than welcome to submit a pull request to this repository, or open an issue, or send me an email…I'd love to hear from you!

## Thanks

plaited builds upon the work of hundreds of people! Here are a handful of them:

- [John MacFarlane](https://johnmacfarlane.net/) and other contributors to [Pandoc](https://pandoc.org/)
- [Sergio Correia](http://scorreia.com/) and other contributors to [Panflute](https://github.com/sergiocorreia/panflute)
- [Tom Augspurger](https://github.com/TomAugspurger), who wrote [Stitch](https://github.com/pystitch/stitch)
- [Peter Zagubisalo](https://github.com/kiwi0fruit), who created [Knitty](https://github.com/kiwi0fruit/knitty)
- [Jan Katins](https://www.katzien.de/en/), who wrote [knitpy](https://github.com/jankatins/knitpy)

## License

There seems to have been a misunderstanding by previous developers of this project regarding license compatibility.
[Stitch](https://github.com/pystitch/stitch) and [Knitty](https://github.com/kiwi0fruit/knitty), by Tom Augspurger and Peter Zagubisalo respectively, were released using the [MIT License](https://en.wikipedia.org/wiki/MIT_License).
Their code is adapted from the [knitpy](https://github.com/jankatins/knitpy) and [IPython](https://github.com/ipython/ipython) projects, both released under BSD licenses.
I am not a lawyer, but I do not believe that BSD licenses are compatible with the MIT license.
I hope that by relicensing this project under the Modified (3-Clause) BSD License, work by all prior contributors is being used according to their original licenses.
This is not legal advice, and I welcome any feedback or complaints about the licensure of this repository.
