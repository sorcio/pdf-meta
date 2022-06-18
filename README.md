# pdf-meta

Reads metadata from a PDF, in pure Python with no third party dependencies.

I only made this because I had some very weird and specific requirements. If
you need to parse a PDF file with Python you are very likely better served by
something like [PyPDF2](https://pypdf2.readthedocs.io/) or any of the thousands
of packages that PyPI indexes when searching for "pdf".

However, since I spent a minute making this, I thought it would be nice to
make it available to others.

## Goals

The only thing this code does right now is to get the creation date from XMP
metadata in a PDF file. Probably it can be extended to do more, should the need
ever arise.

Basics:

* pure Python implementation (tested on 3.10)
* no third-party dependencies, only standard library
* reasonably fast (meaning that large files won't slow it down, not that it's
  fast in absolute when compared to other implementations)
* reasonably tolerant to malformed or extremely large files
* contained memory usage (by only requiring small chunks of the file to be
  in memory)
* single-file, or in any case easily vendorable

## Non-goals and limitations

All of the limitations? I only tested this on a number of files. I made it work
for those. It complies to the standard only on some points.

I can see other use cases, like parsing and interpreting more metadata fields,
or extracting some general information from a document. As long as the
implementation stays compact, this is fair.

This project is probably a bad starting point if you need to parse anything
about the contents of a PDF. And it's a no-go if you need to manipulate or
write PDF files.

## Installation

At this time, this is not packaged for being installable. There is a single
file, `parsepdf.py`, which you can download or include in your project.

## Usage

As a CLI script, it will take a list of PDF files and output their file name
followed by the create date (extracted from the file) in YYYY-MM-DD format:

> **python parsepdf.py** *file* ...

E.g.:

```shell
$ python parsepdf.py file1.pdf file2.pdf
file1.pdf 2022-06-18
file2.pdf 2022-06-17
```

You can also import it as a module, but the interface is even less polished.
Something that should work is:

```python
>>> from pathlib import Path
>>> from parsepdf import get_create_date_for_pdf
>>> get_create_date_for_pdf(Path("file1.pdf"))
datetime.datetime(2022, 6, 18, 11, 49, 42)
```

## Contributing

This project is mostly a single-person hack, with no documentation or tests.
Still, you are welcome to contribute if you feel like it! You can open an issue
or make a pull request, even if this has been inactive for a long time.

Make sure to comply with the [Code of
Conduct](https://github.com/sorcio/.github/blob/main/.github/CODE_OF_CONDUCT.md)
when interacting on any project space.

## License

This repository is available under the terms of the [MIT license](LICENSE).

Please open an issue if you have different licensing needs.
