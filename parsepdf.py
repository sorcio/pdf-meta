"""
Extract date from PDF metadata.

(With no external libraries)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from io import BufferedReader, BytesIO, IOBase
from mmap import mmap
from pathlib import Path
import sys
from typing import Any, Callable, Generator
import xml.sax
from xml.sax.xmlreader import InputSource
from xml.sax.handler import ContentHandler, feature_namespaces


# Python docs suggest to check pyexpat.EXPAT_VERSION to ensure that
# expat has countermeasures against some known vulnerabilities.
# https://docs.python.org/3/library/xml.html#xml-vulnerabilities
ALLOW_VULNERABLE_XML_PARSER = False
if not ALLOW_VULNERABLE_XML_PARSER:
    from pyexpat import EXPAT_VERSION

    assert EXPAT_VERSION > "expat_2.4.1"


class BaseHandler:
    def start_dictionary(self):
        pass

    def end_dictionary(self):
        pass

    def start_array(self):
        pass

    def end_array(self):
        pass

    def name(self, value: bytes):
        pass

    def number(self, number: int | float):
        pass

    def string(self, value: bytes):
        pass

    def bool(self, value: bool):
        pass

    def null(self):
        pass

    def reference(self, obj: int, gen: int):
        pass

    def indirect_object(self, obj: int, gen: int):
        pass

    def stream(self):
        # Must implement in child classes if you need to support streams.
        raise NotImplementedError


class InvalidPDF(Exception):
    pass


class PdfParser:
    """Very approximative PDF parser with little-to-no validation"""

    def __init__(self, scanner: "PdfScanner", handler: BaseHandler):
        self._scanner = scanner
        self._current_token = scanner.next_token()
        self._previous_token = None
        self._eof: bool = False
        self.handler = handler

    def indirect_object(self):
        obj_number = self.consume_number()
        obj_gen = self.consume_number()
        self.consume_keyword(b"obj")
        self.handler.indirect_object(obj_number, obj_gen)
        self.object()
        self.consume_keyword(b"endobj")

    def trailer(self):
        tok = self.consume(TokenType.SEQUENCE)
        assert tok.value == b"trailer"
        self.object()

    def object(self):
        if self.match(TokenType.SEQUENCE):
            # possibly a keyword?
            tok = self.previous()
            if tok.value in (b"true", b"false"):
                self.boolean_object()
            elif tok.value == b"null":
                self.null_object()
            else:
                self.error(tok, "unexpected sequence")
        elif self.match(TokenType.DICT_BEGIN):
            self.dictionary_or_stream_object()
        elif self.match(TokenType.SOLIDUS):
            self.name_object()
        elif self.match(TokenType.ARRAY_BEGIN):
            self.array_object()
        elif self.match(TokenType.STRING):
            self.string_object()
        elif self.match(TokenType.NUMBER):
            if self.check(TokenType.NUMBER):
                self.indirect_reference()
            else:
                self.number_object()
        else:
            self.error(self.peek(), "invalid object")

    def indirect_reference(self):
        tok1 = self.previous()
        tok2 = self.consume(TokenType.NUMBER)
        tok3 = self.consume(TokenType.SEQUENCE)
        assert tok1.value is not None
        assert tok2.value is not None
        assert tok3.value == b"R"
        obj_number = int(tok1.value)
        gen_number = int(tok2.value)
        self.handler.reference(obj_number, gen_number)

    def boolean_object(self):
        value = self.previous().value == b"true"
        self.handler.bool(value)

    def number_object(self):
        raw = self.previous().value
        assert raw is not None
        value = int(raw)
        self.handler.number(value)

    def dictionary_or_stream_object(self):
        self.dictionary_object()
        if self.check(TokenType.SEQUENCE):
            if self.peek().value == b"stream":
                self.stream_object()

    def stream_object(self):
        self._scanner.prepare_for_binary_stream()
        # We don't consume the 'stream' token because the scanner breaks if we
        # try to lookahead into the stream. This is a bit of a fragile state so
        # we need to be careful.
        self.handler.stream()
        # Assuming the handler did the right thing and advanced the scanner, we
        # can "consume" the current token now and go to the next one AFTER the
        # stream bytes.
        self.consume_keyword(b"stream")
        self.consume_keyword(b"endstream")

    def string_object(self):
        raw = self.previous().value
        assert raw is not None
        self.handler.string(raw)

    def name_object(self):
        tok = self.consume(TokenType.SEQUENCE)
        raw = tok.value
        assert raw is not None
        self.handler.name(raw)

    def array_object(self):
        self.handler.start_array()
        while not self.check(TokenType.ARRAY_END):
            self.object()
        self.consume(TokenType.ARRAY_END)
        self.handler.end_array()

    def null_object(self):
        self.handler.null()

    def dictionary_object(self):
        self.handler.start_dictionary()
        while not self.check(TokenType.DICT_END):
            self.object()
            self.object()
        self.consume(TokenType.DICT_END)
        self.handler.end_dictionary()

    # Parser helpers:

    def match(self, *types):
        if any(map(self.check, types)):
            self.advance()
            return True
        return False

    def check(self, type):
        if self.is_at_end():
            return False
        return self.peek().type == type

    def advance(self):
        token = self._current_token
        try:
            self._previous_token = token
            self._current_token = self._scanner.next_token()
        except EOFError:
            self._eof = True
        return token

    def is_at_end(self):
        return self._eof

    def peek(self):
        if self.is_at_end():
            raise RuntimeError
        return self._current_token

    def previous(self):
        if self._previous_token is None:
            raise RuntimeError
        return self._previous_token

    def consume(self, type, message: str = ""):
        if self.check(type):
            return self.advance()
        self.error(self.peek(), message or f"expected {type}")

    def consume_keyword(self, kw: bytes, message: str = ""):
        if self.check(TokenType.SEQUENCE):
            token = self.peek()
            if token.value == kw:
                return self.advance()
        self.error(self.peek(), message or f"expected keyword '{kw}'")

    def consume_number(self, message: str = ""):
        token = self.consume(TokenType.NUMBER)
        assert token.value is not None
        return int(token.value, 10)

    def error(self, token: "Token", message: str):
        raise InvalidPDF(f"error at token {token!r}: {message}")


class TokenType(Enum):
    # Single character
    ARRAY_BEGIN = auto()
    ARRAY_END = auto()
    CURLY_OPEN = auto()
    CURLY_CLOSE = auto()
    SOLIDUS = auto()
    # Double character
    DICT_BEGIN = auto()
    DICT_END = auto()
    # Composite
    STRING = auto()
    COMMENT = auto()
    SEQUENCE = auto()
    NUMBER = auto()


@dataclass
class Token:
    type: TokenType
    value: bytes | None = field(default=None)


class PdfScanner:
    def __init__(self, data: bytes | mmap, start: int = 0, end: int = -1):
        self._data = data
        self._pos = start
        if end < 0:
            self._length = len(data)
        else:
            self._length = end

    def _consume(self):
        char = self._peek()
        self._pos += 1
        return char

    def _peek(self):
        if self._pos >= self._length:
            return b""
        return self._data[self._pos : self._pos + 1]

    def tokenize(self):
        try:
            while True:
                yield self.next_token()
        except EOFError:
            pass

    def next_token(self):
        while (tok := self._scan_token()) is None:
            pass
        return tok

    def _scan_token(self) -> Token | None:
        c = self._consume()
        if not c:
            raise EOFError
        if c == b"[":
            return Token(TokenType.ARRAY_BEGIN)
        if c == b"]":
            return Token(TokenType.ARRAY_END)
        if c == b"{":
            return Token(TokenType.CURLY_CLOSE)
        if c == b"}":
            return Token(TokenType.CURLY_CLOSE)
        if c == b"/":
            return Token(TokenType.SOLIDUS)
        if c == b"%":
            # comment
            value = b""
            while self._peek() not in b"\r\n":
                c = self._consume()
                if len(value) < 128:
                    value += c
            return Token(TokenType.COMMENT, value)
        if c.isdigit():
            # TODO: real numbers
            value = c
            while self._peek().isdigit():
                value += self._consume()
            return Token(TokenType.NUMBER, value)
        if c.isspace():
            return None
        if c == b"<":
            if self._peek() == b"<":
                self._consume()
                return Token(TokenType.DICT_BEGIN)
            return self._hex_string()
        if c == b">":
            if self._consume() != b">":
                raise InvalidPDF("expected >>")
            return Token(TokenType.DICT_END)
        if c == b"(":
            return self._string()
        # "All characters except the white-space characters and delimiters are
        # referred to as regular characters. These characters include bytes
        # that are outside the ASCII character set. A sequence of consecutive
        # regular characters comprises a single token."
        return self._sequence(c)

    def _hex_string(self):
        hex_value = b""
        while True:
            c = self._consume()
            if not c:
                raise InvalidPDF("unexpected EOF while reading hex string")
            if c == b">":
                break
            hex_value += c
        value = bytes.fromhex(hex_value.decode("ascii"))
        return Token(TokenType.STRING, value)

    def _string(self):
        value = b""
        while True:
            c = self._consume()
            if not c:
                raise InvalidPDF("unexpected EOF while reading string")
            if c == b")":
                break
            if c == b"\r":
                if self._peek() == b"\n":
                    self._consume()
                c = b"\n"
            if c == b"\\":
                next_c = self._consume()
                if next_c == b"n":
                    c = b"\n"
                elif next_c == b"r":
                    c = b"\r"
                elif next_c == b"t":
                    c = b"\t"
                elif next_c == b"b":
                    c = b"\b"
                elif next_c == b"f":
                    c = b"\f"
                elif next_c == b"(":
                    c = b"("
                elif next_c == b")":
                    c = b")"
                elif next_c == b"\\":
                    c = b"\\"
                elif next_c.isdigit():
                    octal_digits = next_c
                    octal_digits += self._consume()
                    octal_digits += self._consume()
                    if len(octal_digits) != 3 and not all(
                        x in b"01234567" for x in octal_digits
                    ):
                        raise InvalidPDF("invalid octal escape sequence")
                    c = bytes([int(octal_digits, 8)])
                else:
                    raise InvalidPDF("invalid escape sequence")
            value += c
        return Token(TokenType.STRING, value)

    def _sequence(self, c: bytes):
        seq = c
        while True:
            peeked_value = self._peek()
            if (
                not peeked_value
                or peeked_value.isspace()
                or peeked_value in b"()<>[]{}/%"
            ):
                break
            seq += self._consume()
        return Token(TokenType.SEQUENCE, seq)

    def read_xref_sub_section(self, length: int):
        while self._peek().isspace():
            self._consume()
        for i in range(length):
            # Each row is:
            # nnnnnnnnnn ggggg n EOL
            raw = self._data[self._pos : self._pos + 20]
            self._pos += 20
            number = int(raw[0:10], 10)
            generation = int(raw[11:16], 10)
            in_use_or_free = raw[17:18]
            if in_use_or_free == b"n":
                yield (i, number, generation)

    def prepare_for_binary_stream(self):
        # This just means that we need to exhaust the whitespace at the end of
        # the last token, ONLY until the first newline.
        while self._peek() in b"\0\t\f ":
            self._consume()
        if self._peek() == b"\r":
            self._consume()
        if self._peek() == b"\n":
            self._consume()

    def read_binary_stream(self, length: int) -> bytes:
        if self._pos + length >= self._length:
            raise InvalidPDF("stream length exceeds document boundaries")
        data = self._data[self._pos : self._pos + length]
        self._pos += length
        return data


class PrintHandler:
    def __getattr__(self, name):
        def method(*args):
            print(f">>> {name} {args}")

        return method


@dataclass(frozen=True)
class IndirectReference:
    obj: int
    gen: int


@dataclass(frozen=True, slots=True)
class Name:
    name: str


class BuildObjectHandler(BaseHandler):
    """Transform the event stream into an equivalent Python object."""

    NO_RESULT = object()

    class StopObject(Exception):
        pass

    def __init__(self):
        self._stack: list[Generator[None, None, Any]] = []
        self.result: Any = self.NO_RESULT
        self._push(self._result_handler)

    def _result_handler(self):
        assert self.result is self.NO_RESULT
        self.result = yield

    def _dictionary_handler(self):
        d = {}
        try:
            while True:
                key = yield
                value = yield
                d[key] = value
        except self.StopObject:
            return d

    def _list_handler(self):
        l = []
        try:
            while True:
                l.append((yield))
        except self.StopObject:
            return l

    def _push(self, coro_func: Callable[[], Generator[None, None, Any]]):
        coro = coro_func()
        coro.send(None)
        self._stack.append(coro)

    def _pop(self):
        coro = self._stack.pop()
        try:
            coro.throw(self.StopObject)
        except StopIteration as e:
            result = e.args[0]
        else:
            raise RuntimeError("coroutine did not stop when requested")
        self._new_object(result)

    def _new_object(self, obj):
        try:
            self._stack[-1].send(obj)
        except StopIteration:
            pass

    def start_dictionary(self):
        self._push(self._dictionary_handler)

    def end_dictionary(self):
        self._pop()

    def start_array(self):
        self._push(self._list_handler)

    def end_array(self):
        self._pop()

    def number(self, number: int | float):
        self._new_object(number)

    def string(self, value: bytes):
        self._new_object(value)

    def reference(self, obj: int, gen: int):
        self._new_object(IndirectReference(obj, gen))

    def null(self):
        self._new_object(None)

    def name(self, value: bytes):
        self._new_object(Name(value.decode("ascii")))


class StreamObjectHandler(BuildObjectHandler):
    def __init__(self, handle_stream: Callable[[int], None]):
        super().__init__()
        self._handle_stream = handle_stream

    def stream(self):
        assert isinstance(self.result, dict)
        length = self.result[Name("Length")]
        if not isinstance(length, int):
            raise InvalidPDF("stream 'Length' is not an integer")
        self._handle_stream(length)


def read_xref_table(m: mmap, start: int, end: int, pdf_size: int):
    # TODO: support Cross-Reference Streams (7.5.8)
    scanner = PdfScanner(m, start, end)
    stream = scanner.tokenize()

    table = {}

    tok = next(stream)
    while True:
        # Sections begin with 'xref'
        if tok != Token(TokenType.SEQUENCE, b"xref"):
            break
        tok = next(stream)

        while True:
            # Sub-sections begin with two numbers
            if tok.type != TokenType.NUMBER:
                break
            assert tok.value is not None
            sub_section_start = int(tok.value)
            tok = next(stream)
            if tok.type != TokenType.NUMBER:
                break
            assert tok.value is not None
            sub_section_length = int(tok.value)
            for i, n, g in scanner.read_xref_sub_section(sub_section_length):
                ref = IndirectReference(sub_section_start + i, g)
                table[ref] = n

            # only legal place to hit EOF
            try:
                tok = next(stream)
            except StopIteration:
                break
    return table


def find_xref_table(fp: IOBase) -> bytes:
    import mmap

    with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as m:
        # "Conforming readers should read a PDF file from its end", says the
        # standard. We are not conforming, but we want to read the xref table
        # anyway.
        length = len(m)
        if length < 128:
            raise InvalidPDF("not enough data to read")
        rpos = length - 1
        eof_marker = m.rfind(b"%%EOF", rpos - 7, rpos)
        if eof_marker < 0:
            raise InvalidPDF("missing EOF marker")
        rpos = eof_marker
        startxref = m.rfind(b"startxref", rpos - 24, rpos)
        if startxref < 0:
            raise InvalidPDF("missing 'startxref'")
        try:
            _, xrefpos_ascii = m[startxref:rpos].split()
            xrefpos = int(xrefpos_ascii)
        except ValueError:
            raise InvalidPDF("unreadable startxref value")

        rpos = startxref

        # Let's read the trailer dictionary.

        trailer_end = m.rfind(b">>", rpos - 4, rpos) + 2
        rpos = trailer_end - 2

        # arbitrary limit, increase if parsing fails for some files:
        MAX_TRAILER_DICT_SIZE = 4096
        trailer_safe_start = max(rpos - MAX_TRAILER_DICT_SIZE, 0)
        NEWLINES = (10, 13)

        trailer_found = False
        while rpos > trailer_safe_start:
            rpos = m.rfind(b"trailer", trailer_safe_start, rpos)
            if m[rpos - 1] in NEWLINES and m[rpos + 7] in NEWLINES:
                trailer_found = True
                break

        if trailer_found:
            scanner = PdfScanner(m, rpos, trailer_end)
            build_dictionary = BuildObjectHandler()
            parser = PdfParser(scanner, build_dictionary)
            parser.trailer()
            trailer_dict = build_dictionary.result
            if trailer_dict is None:
                raise InvalidPDF("could not parse trailer dictionary")

            try:
                pdf_size = trailer_dict[Name("Size")]
                pdf_root = trailer_dict[Name("Root")]
            except KeyError as ke:
                raise InvalidPDF(f"missing key in trailer dictionary: {ke.args[0]}")

            # We have what we need to read the xref table
            xref_table = read_xref_table(m, xrefpos, rpos, pdf_size)

            # "Root" points to the document catalog
            if not isinstance(pdf_root, IndirectReference):
                raise InvalidPDF("'Root' is not an object reference")
        else:
            # TODO: find a cross-reference stream
            raise InvalidPDF(
                "cross-reference streams are currently not supported"
            )

        MAX_CATALOG_SIZE = 4096
        root_offset = xref_table[pdf_root]
        scanner = PdfScanner(m, root_offset, root_offset + MAX_CATALOG_SIZE)
        build_dictionary = BuildObjectHandler()
        parser = PdfParser(scanner, build_dictionary)
        parser.indirect_object()
        catalog_dict = build_dictionary.result
        if catalog_dict is None:
            raise InvalidPDF("could not parse catalog dictionary")

        # and "Metadata" (in the catalog) points to the XMP metadata
        try:
            metadata_ref = catalog_dict[Name("Metadata")]
        except KeyError:
            raise InvalidPDF("catalog has no 'Metadata' entry")

        if not isinstance(metadata_ref, IndirectReference):
            raise InvalidPDF("'Metadata' is not an object reference")

        MAX_METADATA_SIZE = 65536
        metadata_offset = xref_table[metadata_ref]
        scanner = PdfScanner(m, metadata_offset, metadata_offset + MAX_METADATA_SIZE)
        raw_metadata: bytes | None = None

        def read_stream(length: int):
            nonlocal raw_metadata
            raw_metadata = scanner.read_binary_stream(length)

        stream_handler = StreamObjectHandler(read_stream)
        parser = PdfParser(scanner, stream_handler)
        parser.indirect_object()
        if not raw_metadata:
            raise InvalidPDF("could not read metadata")
        assert isinstance(raw_metadata, bytes)
        return raw_metadata


def extract_raw_xmp(fp: BufferedReader):
    return find_xref_table(fp)


class XmpMetadataContentHandler(ContentHandler):
    def __init__(self):
        super().__init__()
        self.in_create_date = False
        self.data = ""

        self.create_date = None

    def startElementNS(self, name, qname, attrs):
        if name == ("http://ns.adobe.com/xap/1.0/", "CreateDate"):
            self.in_create_date = True
            self.data = ""
        elif name == ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "Description"):
            try:
                create_date = attrs[("http://ns.adobe.com/xap/1.0/", "CreateDate")]
            except KeyError:
                pass
            else:
                self.create_date = create_date

    def characters(self, content):
        self.data += content

    def endElementNS(self, name, qname):
        if name == ("http://ns.adobe.com/xap/1.0/", "CreateDate"):
            self.in_create_date = False
            self.create_date = self.data


def parse_metadata(raw_xmp: bytes):
    # suddenly, we are writing Java (maybe using xml.parsers.expat directly
    # would be nicer?)
    handler = XmpMetadataContentHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    parser.setFeature(feature_namespaces, True)
    input_source = InputSource()
    input_source.setByteStream(BytesIO(raw_xmp))
    parser.parse(input_source)

    if handler.create_date:
        raw_date = handler.create_date.removesuffix("Z")
        create_date = datetime.fromisoformat(raw_date)
        return create_date
    else:
        return None


class ParsePdfError(Exception):
    pass


class InvalidMetadata(ParsePdfError):
    pass


class MissingCreateDate(ParsePdfError):
    pass


def get_create_date_for_pdf(p: Path):
    with open(p, "rb") as f:
        raw_xmp = extract_raw_xmp(f)

    if raw_xmp is None:
        raise InvalidMetadata

    create_date = parse_metadata(raw_xmp)
    if create_date is None:
        raise MissingCreateDate

    return create_date


def error(*args):
    print(*args, file=sys.stderr)


def abort(message: str):
    error(message)
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        abort(f"Usage: {sys.argv[0]} file ...")

    paths = [Path(x) for x in sys.argv[1:]]

    for path in paths:
        try:
            create_date = get_create_date_for_pdf(path)
        except FileNotFoundError:
            error(path, "error: file not found")
        except InvalidPDF as err:
            error(path, "error: I was not able to parse the PDF:", err)
        except InvalidMetadata:
            error(path, "error: I was not able to find XMP metadata")
        except MissingCreateDate:
            error(path, "error: I was not able to find the create date")
        except Exception:
            error(path, "error: unexpected error while parsing")
            raise
        else:
            print(str(path), create_date.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main()
