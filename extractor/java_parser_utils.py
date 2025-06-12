"""Light-weight helpers built on top of *javalang* to obtain a (partial) API
surface from Java source files.

We purposefully restrict ourselves to the syntactic level – enough for API
change tracking – and avoid *semantic* tasks such as resolving imported simple
names to fully-qualified names.  For Spring's internal consistency this is
usually acceptable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import javalang  # type: ignore[import]

PUBLIC_LIKE = {"public", "protected"}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_javadoc_deprecated(comment: str | None) -> bool:
    if comment is None:
        return False
    return "@deprecated" in comment.lower()


def parse_java_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return API items defined in *path* keyed by *stable id*.

    The stable id format is
        ``<fqcn>#<memberSignature>`` for methods/constructors/fields or
        ``<fqcn>`` for top-level types.
    Example: ``org.springframework.foo.Bar#baz(String,int)``
    """
    code = path.read_text(encoding="utf-8", errors="ignore")
    lines = code.splitlines()

    # Skip generated files or resources quickly to save time
    if "@Generated" in code and "generated" in path.name.lower():
        return {}

    try:
        tree = javalang.parse.parse(code)
        package_name = tree.package.name if tree.package else ""
    except Exception:  # noqa: BLE001
        # Any failure (syntax or lexer) -> fallback to regex heuristics
        return _regex_fallback_parse(code, path)

    api_items: Dict[str, Dict[str, Any]] = {}

    def _snippet(pos):
        if pos and pos.get("line"):
            return _capture_block(lines, pos["line"])
        return ""

    for _, node in tree.filter(
        (
            javalang.tree.ClassDeclaration,
            javalang.tree.InterfaceDeclaration,
            javalang.tree.EnumDeclaration,
        )
    ):
        if not (set(node.modifiers) & PUBLIC_LIKE):
            # Only record public/protected top-level elements
            continue

        fqcn = f"{package_name}.{node.name}" if package_name else node.name
        javadoc = _get_node_doc(_node_position(node), code)
        deprecated = "Deprecated" in node.annotations or _extract_javadoc_deprecated(javadoc)
        # api_items[fqcn] = {
        #     "kind": node.__class__.__name__.replace("Declaration", "").lower(),
        #     "deprecated": deprecated,
        #     "modifiers": sorted(node.modifiers),
        #     "signature": _class_signature(node),
        #     "position": _node_position(node),
        #     "doc": javadoc or "",
        #     "source_code": _snippet(_node_position(node)),
        # }

        # Members ----------------------------------------------------------------
        for member in node.body:
            if isinstance(member, javalang.tree.MethodDeclaration):
                if not (set(member.modifiers) & PUBLIC_LIKE):
                    continue
                sig_id = _method_stable_id(fqcn, member)
                docstr = _get_node_doc(_node_position(member), code)
                source_code = _snippet(_node_position(member))
                if len(source_code) < 10 or len(member.parameters) < 2:
                    continue
                api_items[sig_id] = {
                    "kind": "method",
                    "return": _type_to_str(member.return_type),
                    "params": [_type_to_str(p.type) for p in member.parameters],
                    "deprecated": (
                        _has_deprecated_annotation(member.annotations)
                        or _extract_javadoc_deprecated(docstr)
                    ),
                    "modifiers": sorted(member.modifiers),
                    "signature": _method_signature(member),
                    "position": _node_position(member),
                    "doc": docstr or "",
                    "source_code": source_code,
                }
            # elif isinstance(member, javalang.tree.ConstructorDeclaration):
            #     if not (set(member.modifiers) & PUBLIC_LIKE):
            #         continue
            #     sig_id = _ctor_stable_id(fqcn, member)
            #     docstr = _get_node_doc(_node_position(member), code)
            #     api_items[sig_id] = {
            #         "kind": "constructor",
            #         "params": [_type_to_str(p.type) for p in member.parameters],
            #         "deprecated": (
            #             _has_deprecated_annotation(member.annotations)
            #             or _extract_javadoc_deprecated(docstr)
            #         ),
            #         "modifiers": sorted(member.modifiers),
            #         "signature": _ctor_signature(member, fqcn.split(".")[-1]),
            #         "position": _node_position(member),
            #         "doc": docstr or "",
            #         "source_code": _snippet(_node_position(member)),
            #     }
            # elif isinstance(member, javalang.tree.FieldDeclaration):
            #     if not (set(member.modifiers) & PUBLIC_LIKE):
            #         continue
            #     for declarator in member.declarators:
            #         field_id = f"{fqcn}.{declarator.name}"
            #         docstr = _get_node_doc(_node_position(member), code)
            #         api_items[field_id] = {
            #             "kind": "field",
            #             "type": _type_to_str(member.type),
            #             "deprecated": (
            #                 _has_deprecated_annotation(member.annotations)
            #                 or _extract_javadoc_deprecated(docstr)
            #             ),
            #             "modifiers": sorted(member.modifiers),
            #             "signature": _field_signature(member, declarator.name),
            #             "position": _node_position(member),
            #             "doc": docstr or "",
            #             "source_code": _snippet(_node_position(member)),
            #         }
    return api_items

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _type_to_str(typ) -> str:  # type: ignore[override]
    if typ is None:
        return "void"

    if isinstance(typ, str):
        return typ

    if hasattr(typ, "name"):
        # For simple and parameterized types, `.name` holds the raw identifier.
        name = typ.name  # type: ignore[attr-defined]
        # If the type itself has generic arguments (ParameterizedType in
        # javalang) capture them recursively so that we emit the full
        # signature, e.g. "Constructor<?>" instead of just "Constructor".
        if hasattr(typ, "arguments") and typ.arguments:  # type: ignore[attr-defined]
            args = ", ".join(_type_to_str(a) for a in typ.arguments)
            name += f"<{args}>"
    else:
        # Fallback – str() often already contains the generic arguments.
        name = str(typ)
    if typ.dimensions:  # type: ignore[attr-defined]
        name += "[]" * len(typ.dimensions)
    return name


def _method_stable_id(fqcn: str, node: "javalang.tree.MethodDeclaration") -> str:  # type: ignore[name-defined]
    params = ",".join(_type_to_str(p.type) for p in node.parameters)
    return f"{fqcn}#{node.name}({params})"


def _ctor_stable_id(fqcn: str, node: "javalang.tree.ConstructorDeclaration") -> str:  # type: ignore[name-defined]
    params = ",".join(_type_to_str(p.type) for p in node.parameters)
    return f"{fqcn}#<init>({params})"

def _get_node_doc(position: Dict[str, int], source: str | None = None) -> str | None:
    """Return the raw Javadoc string that immediately precedes *node*.

    We rely on the *position* attribute provided by *javalang* AST nodes to
    look *upwards* in the source file and capture the closest `/** … */` block
    (ignoring annotation lines and blank space).  Falls back to the previous
    regex-based heuristic for nodes without a recorded position.
    """

    if source is None:
        return None
    
    lines = source.splitlines()
    # print(f"node.position: {position}")
    line_no = position["line"] - 1  # zero-based index of declaration line

    # Walk *upwards* skipping blank lines or annotation lines (starting with '@').
    i = line_no - 1
    while i >= 0 and (lines[i].strip() == "" or lines[i].lstrip().startswith("@")):
        i -= 1

    # We now expect to be either on the line with '*/' terminating the Javadoc
    if i >= 0 and "*/" in lines[i]:
        # Collect lines until we hit the opening '/**'
        doc_lines: list[str] = []
        while i >= 0:
            doc_lines.append(lines[i])
            if "/**" in lines[i]:
                break
            i -= 1
        doc_lines.reverse()
        return "\n".join(doc_lines)

    # Fallback: original regex heuristic ------------------------------------
    # match = re.search(r"/\*\*((?:.|\n)*?)\*/[\s\S]{0,120}?" + re.escape(name), source)
    # if match:
    #     return _clean_javadoc(match.group(1).splitlines())
    return None


def _has_deprecated_annotation(annots: Sequence[Any]) -> bool:
    for ann in annots:
        if getattr(ann.name, "value", ann.name) == "Deprecated":
            return True
    return False

# ---------------------------------------------------------------------------
# regex fallback
# ---------------------------------------------------------------------------


_CLASS_RE = re.compile(
    r"(?m)^[ \t]*(public|protected)[ \t\w<>]*?\b(class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)",
)

# captures: modifiers.. return type .. name (group3) .. params (group4)
_METHOD_RE = re.compile(
    r"(?m)^[ \t]*(public|protected)[ \t\w<>,&@]*?\s+([A-Za-z_][\w<>\[\]]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)",
)

# field: modifiers .. type .. name
_FIELD_RE = re.compile(
    r"(?m)^[ \t]*(public|protected)[ \t\w<>,&@]*?\s+([A-Za-z_][\w<>\[\]]*)\s+([A-Za-z_][A-Za-z0-9_]*)(\s*[;=])",
)


def _regex_fallback_parse(code: str, path: Path) -> Dict[str, Dict[str, Any]]:
    """Very crude regex-based extraction used when *javalang* fails.

    Only looks for top-level types, methods, and fields with public/protected
    visibility.  Accuracy is lower but better than nothing.
    """
    package_match = re.search(r"(?m)^\s*package\s+([\w\.]+)\s*;", code)
    package_name = package_match.group(1) if package_match else ""

    api: Dict[str, Dict[str, Any]] = {}

    # First, collect declared type simple names so we can later detect ctors
    declared_types: list[str] = []
    # for m in _CLASS_RE.finditer(code):
    #     kind = m.group(2)
    #     name = m.group(3)
    #     declared_types.append(name)
    #     fqcn = f"{package_name}.{name}" if package_name else name
    #     api[fqcn] = {
    #         "kind": kind.lower(),
    #         "deprecated": False,
    #         "modifiers": [m.group(1)],
    #         "signature": f"{m.group(1)} {kind} {name}",
    #         "position": _index_position(code, m.start()),
    #         "doc": _get_node_doc(_index_position(code, m.start()), code),
    #         "source_code": _capture_block(code.splitlines(), _index_position(code, m.start())["line"]),
    #     }

    # # Fields
    # for m in _FIELD_RE.finditer(code):
    #     type_str = m.group(2)
    #     name = m.group(3)
    #     fq_field = f"{package_name}.{name}" if package_name else name
    #     api[fq_field] = {
    #         "kind": "field",
    #         "type": type_str,
    #         "deprecated": False,
    #         "modifiers": [m.group(1)],
    #         "signature": f"{m.group(1)} {type_str} {name}",
    #         "position": _index_position(code, m.start()),
    #         "doc": _get_node_doc(_index_position(code, m.start()), code),
    #         "source_code": _capture_block(code.splitlines(), _index_position(code, m.start())["line"]),
    #     }

    # Methods / constructors
    for m in _METHOD_RE.finditer(code):
        ret_type = m.group(2)
        method_name = m.group(3)
        params_raw = m.group(4).strip()
        # Preserve the *original* "type name" fragments for the signature while
        # extracting just the *names* for the lightweight `params` field that is
        # later used for change detection.
        params = [p.strip().split()[-1] if " " in p.strip() else p.strip() for p in params_raw.split(",") if p.strip()]

        fqcn_prefix = package_name + "." if package_name else ""

        if method_name not in declared_types:  # constructor
        #     sig_id = f"{fqcn_prefix}{method_name}#<init>({','.join(params)})"
        #     api[sig_id] = {
        #         "kind": "constructor",
        #         "params": params,
        #         "deprecated": False,
        #         "modifiers": [m.group(1)],
        #         "signature": f"{m.group(1)} {method_name}({params_raw})",
        #         "position": _index_position(code, m.start()),
        #         "doc": _get_node_doc(_index_position(code, m.start()), code),
        #         "source_code": _capture_block(code.splitlines(), _index_position(code, m.start())["line"]),
        #     }
        # else:
            # method
            # Need class context to generate id; we approximate by using first declared type in file for now
            owner = declared_types[0] if declared_types else "<unknown>"
            fqcn = f"{fqcn_prefix}{owner}#{method_name}({','.join(params)})"
            source_code = _capture_block(code.splitlines(), _index_position(code, m.start())["line"])
            if len(code.splitlines()) < 10 or len(params) < 2:
                continue
            api[fqcn] = {
                "kind": "method",
                "return": ret_type,
                "params": params,
                "deprecated": False,
                "modifiers": [m.group(1)],
                # Keep full type+name fragments in the signature for fidelity.
                "signature": f"{m.group(1)} {ret_type} {method_name}({params_raw})",
                "position": _index_position(code, m.start()),
                "doc": _get_node_doc(_index_position(code, m.start()), code),
                "source_code": source_code,
            }

    return api 

# Signature helpers


def _class_signature(node) -> str:
    mods = " ".join(sorted(node.modifiers))
    kind = node.__class__.__name__.replace("Declaration", "").lower()
    return f"{mods} {kind} {node.name}".strip()


def _method_signature(node) -> str:
    mods = " ".join(sorted(node.modifiers))
    # Include parameter *types* together with their *names* to preserve the original
    # source-level method signature (e.g. "Constructor<?> ctor" instead of just
    # "ctor").
    params = ", ".join(f"{_type_to_str(p.type)} {p.name}" for p in node.parameters)
    return_type = _type_to_str(node.return_type)
    return f"{mods} {return_type} {node.name}({params})".strip()


def _ctor_signature(node, class_name: str) -> str:
    mods = " ".join(sorted(node.modifiers))
    # Ditto for constructors – keep both type and parameter name.
    params = ", ".join(f"{_type_to_str(p.type)} {p.name}" for p in node.parameters)
    return f"{mods} {class_name}({params})".strip()


def _field_signature(node, field_name: str) -> str:
    mods = " ".join(sorted(node.modifiers))
    type_str = _type_to_str(node.type)
    return f"{mods} {type_str} {field_name}".strip()

# ---------------------------------------------------------------------------
# position helpers
# ---------------------------------------------------------------------------


def _node_position(node) -> Dict[str, int] | None:  # type: ignore[override]
    """Return line/column mapping from a *javalang* AST node (or *None*)."""
    if getattr(node, "position", None):
        line, column = node.position  # type: ignore[attr-defined]
        return {"line": int(line), "column": int(column)}
    return None


def _index_position(code: str, index: int) -> Dict[str, int]:
    """Translate *index* in *code* (character offset) to a (line, column) dict."""
    # Count newlines before *index* to get 1-based line number
    line = code.count("\n", 0, index) + 1
    # Column = distance from previous newline (or start of file)
    last_nl = code.rfind("\n", 0, index)
    column = index - last_nl
    return {"line": line, "column": column}

# ---------------------------------------------------------------------------
# source code snippet helpers
# ---------------------------------------------------------------------------

def _capture_block(lines: List[str], start_line: int) -> str:
    """Return a best-effort code snippet for the element starting at *start_line* (1-based).

    For declarations followed by a body `{ … }` we capture until the matching closing
    brace.  For single-line declarations (e.g. fields or abstract methods) we only
    return that line.
    """

    # Guard against out-of-range indices ------------------------------------
    if start_line < 1 or start_line > len(lines):
        return lines[start_line - 1] if 0 <= start_line - 1 < len(lines) else ""

    snippet_lines: List[str] = []
    brace_depth = 0

    i = start_line - 1  # zero-based index
    # First line always included
    first = lines[i]
    snippet_lines.append(first)

    # Fast-path: interface method / field / annotation line ending with ';'
    if first.strip().endswith(";"):
        return "\n".join(snippet_lines)

    # Initialize brace depth based on first line (account for inline opening brace)
    brace_depth += first.count("{") - first.count("}")
    overall_brace = first.count("{")

    i += 1
    while i < len(lines) and (brace_depth > 0 or overall_brace == 0):
        snippet_lines.append(lines[i])
        brace_depth += lines[i].count("{") - lines[i].count("}")
        overall_brace += lines[i].count("{")
        i += 1

    return "\n".join(snippet_lines)