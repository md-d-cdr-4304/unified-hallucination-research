"""Convert the thesis v3 chapter sources (HTML) into a LaTeX project.

Output: thesis/v3/latex/thesis.tex, refs.bib, figures/ (copied).
The conversion is exact: the same chapter files, the same citation keys, the
same figures. Numbering of chapters, sections, tables, and figures is left to
LaTeX, and every "Table 5.1" / "Figure 4.1" / "Section 4.6" / "Chapter 6" in
the text becomes a \\ref to the corresponding label, so the numbers stay
consistent. Citations become \\cite{key}; refs.bib is generated from the
verified bibliography with a DOI / arXiv / URL link per entry, which the IEEE
style prints and hyperref makes clickable.

The document follows the BTH degree-project template (DP_thesis_tmpl/: bth.cls,
changepage.sty, bthnotext.pdf, bibliography style IEEEtranS). The template's
front page, inner page, and closing pages are reproduced unchanged; the thesis
data (degree, date, faculty, title, authors, supervisor) comes from
chapters/00_front.html. Build: pdflatex -> bibtex -> pdflatex -> pdflatex.
"""
import html, json, re, shutil
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
CH = HERE / "chapters"; BIB = HERE / "bib"; OUT = HERE / "latex"
OUT.mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)

# ------------------------------------------------------------------ text escaping
SYM = {"→": r"$\rightarrow$", "−": r"$-$", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "≈": r"$\approx$",
       "κ": r"$\kappa$", "Δ": r"$\Delta$", "α": r"$\alpha$", "±": r"$\pm$", "·": r"$\cdot$", "—": "---", "–": "--",
       "’": "'", "‘": "`", "“": "``", "”": "''", "‑": "-", "\u202f": "~", "\u00a0": "~", "…": r"\ldots{}", "•": r"\textbullet{}",
       "≠": r"$\neq$", "∈": r"$\in$", "τ": r"$\tau$", "β": r"$\beta$", "λ": r"$\lambda$", "μ": r"$\mu$", "σ": r"$\sigma$", "→": r"$\rightarrow$", "†": r"\dag{}", "✓": r"$\checkmark$"}
ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def esc(t: str) -> str:
    out = []
    for ch in t:
        if ch in ESC: out.append(ESC[ch])
        elif ch in SYM: out.append(SYM[ch])
        else: out.append(ch)
    return "".join(out)


# ------------------------------------------------------------------ bibliography
def load_bib():
    bib = {}
    for f in sorted(BIB.glob("lit_*.json")):
        d = json.loads(f.read_text()); d = d if isinstance(d, list) else d.get("entries", [])
        for e in d:
            if e.get("key") and e["key"] not in bib: bib[e["key"]] = e
    return bib


def bib_author(a):
    a = a.replace("{", "").replace("}", "").strip()
    if "," not in a: return "{" + a + "}"  # corporate author (no "Last, First" form): keep as one name
    return a


def norm_venue(v, year):
    """Normalise venue strings to one IEEE-style form: no editorial notes, no duplicated year,
    one canonical name per conference series."""
    v = re.sub(r"\s*\((?:per arXiv[^)]*|arXiv v3[^)]*|widely cited[^)]*|comments:[^)]*|search snippet[^)]*|v1 titled[^)]*|pre-registered[^)]*|cs\.CR|OpenAI)\)", "", v).strip()
    v = re.sub(r"\s*\(Volume \d+: (?:Long|Short) Papers\)", "", v)
    v = re.sub(r"\((\w+) 20\d\d\)", r"(\1)", v)
    v = re.sub(r"\s+20\d\d$", "", v)
    low = v.lower()
    if low.startswith("arxiv"): return "arXiv"
    ordinal = re.search(r"(\d+)(st|nd|rd|th)", v); ordn = ordinal.group(0) + " " if ordinal else ""
    if "iclr" in low or "learning representations" in low: return "Proc. International Conference on Learning Representations (ICLR)"
    if "neurips" in low or "neural information processing" in low:
        return "Advances in Neural Information Processing Systems (NeurIPS)" + (", Datasets and Benchmarks Track" if "datasets and benchmarks" in low else "")
    if low.startswith("findings"):
        conf = "EMNLP" if "emnlp" in low else ("NAACL" if "naacl" in low else "ACL")
        return "Findings of the Association for Computational Linguistics: " + conf
    if "emnlp" in low or "empirical methods" in low:
        suffix = ": System Demonstrations" if "demonstration" in low else (": Industry Track" if "industry" in low else "")
        return "Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)" + suffix
    if "north american chapter" in low or "naacl" in low: return "Proc. Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)"
    if "international joint conference on natural language" in low: return "Proc. 13th International Joint Conference on Natural Language Processing and 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL)"
    if "annual meeting of the association for computational linguistics" in low or low == "proc. acl": return f"Proc. {ordn}Annual Meeting of the Association for Computational Linguistics (ACL)"
    if "workshop" in low and "icml" in low: return "Proc. ICML Workshop on Foundation Models in the Wild"
    if "international conference on machine learning" in low or low.startswith("icml"): return f"Proc. {ordn}International Conference on Machine Learning (ICML)"
    if "theory of computing" in low or "stoc" in low: return "Proc. 56th Annual ACM Symposium on Theory of Computing (STOC)"
    if "computational natural language learning" in low: return "Proc. 27th Conference on Computational Natural Language Learning (CoNLL)"
    if "usenix security" in low: return "Proc. 34th USENIX Security Symposium"
    if "language modeling" in low and "colm" in low: return "Proc. Conference on Language Modeling (COLM)"
    if "sigkdd" in low: return "Proc. 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)"
    if "machine learning and systems" in low: return "Proc. Machine Learning and Systems (MLSys)"
    v = re.sub(r"^Proceedings of the", "Proc.", v)
    return v


def bibtex(e):
    """One BibTeX entry in the shape IEEEtran.bst expects. IEEEtran has no doi field, so the
    link is emitted as url (printed as '[Online]. Available: ...' and made clickable by hyperref)."""
    typ = e.get("type", "misc"); year = str(e.get("year", ""))
    venue = norm_venue(e.get("venue") or e.get("journal") or e.get("booktitle") or "", year)
    arxiv = e.get("arxiv"); doi = e.get("doi")
    f = []
    auth = e.get("authors") or e.get("author") or []
    if isinstance(auth, str): auth = [auth]
    auth = [a for a in auth if a.strip().lower() != "et al."]
    if auth: f.append(("author", " and ".join(bib_author(a) for a in auth)))
    if e.get("title"): f.append(("title", "{" + e["title"] + "}"))
    if venue == "arXiv" or typ == "preprint" and (not venue or venue == "arXiv"):
        typ, venue = "article", f"arXiv preprint arXiv:{arxiv}" if arxiv else "arXiv preprint"
    elif typ == "preprint": typ = "inproceedings"
    typ = {"inproceedings": "inproceedings", "article": "article", "misc": "misc", "book": "book", "techreport": "techreport"}.get(typ, "misc")
    if venue: f.append(("journal" if typ == "article" else ("booktitle" if typ == "inproceedings" else "howpublished"), venue))
    pages = str(e.get("pages", "") or "")
    pages = re.sub(r"^Art\.\s*\d+,\s*", "", pages); pages = re.sub(r"^Art\.\s*", "", pages)
    for k, val in (("year", year), ("volume", e.get("volume")), ("number", e.get("number")), ("pages", pages), ("publisher", e.get("publisher")), ("institution", e.get("institution")), ("note", e.get("note"))):
        if val: f.append((k, str(val)))
    link = f"https://doi.org/{doi}" if doi else (f"https://arxiv.org/abs/{arxiv}" if arxiv else e.get("url"))
    if link: f.append(("url", link))
    sym = lambda v: "".join(SYM.get(ch, ch) for ch in v)  # non-ASCII symbols in titles/venues (BibTeX is not UTF-8 aware)
    body = ",\n".join(f"  {k} = {{{sym(v)}}}" for k, v in f)
    return f"@{typ}{{{e['key']},\n{body}\n}}\n"


# ------------------------------------------------------------------ label map for cross-references
def build_label_map(front_body_apps):
    lab = {"tab": {}, "fig": {}, "sec": {}, "ch": {}}
    for m in re.finditer(r'<span class="tabnum">(Table|Figure) ([A-Z0-9]+\.[0-9]+):</span>', front_body_apps):
        kind = "tab" if m.group(1) == "Table" else "fig"; lab[kind][m.group(2)] = f"{kind}:{m.group(2).replace('.', '-')}"
    for m in re.finditer(r'<h([23]) id="([^"]+)">\s*([A-Z0-9]+(?:\.[0-9]+)+)&nbsp;', front_body_apps):
        lab["sec"][m.group(3)] = f"sec:{m.group(2)}"
    for m in re.finditer(r'<div class="chapter" id="([^"]+)">\s*<div class="chapnum">(Chapter|Appendix) ([A-Z0-9]+)</div>', front_body_apps):
        lab["ch"][(m.group(2), m.group(3))] = f"ch:{m.group(1)}"
    return lab


NUMRE = r"[A-Z0-9]+(?:\.[0-9]+)*"
XREF = re.compile(r"\b(Table|Figure|Section|Chapter|Appendix)(?:s)?(?:&nbsp;|\u00a0| |~)(" + NUMRE + r")((?: (?:and|to) " + NUMRE + r")*)")


def xref_text(t, lab):
    def one(kind, num):
        if kind == "Table" and num in lab["tab"]: return f"\\ref{{{lab['tab'][num]}}}"
        if kind == "Figure" and num in lab["fig"]: return f"\\ref{{{lab['fig'][num]}}}"
        if kind == "Section" and num in lab["sec"]: return f"\\ref{{{lab['sec'][num]}}}"
        if kind in ("Chapter", "Appendix") and (kind, num) in lab["ch"]: return f"\\ref{{{lab['ch'][(kind, num)]}}}"
        return None
    def rep(m):
        kind, num, tail = m.group(1), m.group(2), m.group(3)
        first = one(kind, num)
        if first is None: return m.group(0)
        plural = m.group(0)[len(kind)] == "s"
        out = kind + ("s" if plural else "") + "~" + first
        for w, n in re.findall(r" (and|to) (" + NUMRE + r")", tail):
            r = one(kind, n); out += f" {w} " + (r if r else n)
        return out
    return XREF.sub(rep, t)


CITE = re.compile(r"\[@([A-Za-z0-9_:\-]+(?:\s*[,;]\s*@?[A-Za-z0-9_:\-]+)*)\]")


# ------------------------------------------------------------------ HTML -> LaTeX
class Conv(HTMLParser):
    def __init__(self, lab, bibkeys):
        super().__init__(convert_charrefs=True)
        self.lab, self.bibkeys = lab, bibkeys
        self.out = []; self.stack = []; self.in_pre = False; self.list_stack = []
        self.table = None; self.row = None; self.cell = None; self.caption = None; self.in_caption = False
        self.fig = None; self.skip = 0; self.p_noindent = False; self.chap = None
        self.unknown = set(); self.pending_label = None

    # ---- text handling
    def emit(self, s): (self.cell if self.cell is not None else (self.caption if self.in_caption else (self.fig["buf"] if self.fig is not None and self.fig.get("buf") is not None else self.out))).append(s)

    def handle_data(self, data):
        if self.skip: return
        if self.in_pre: self.emit(data); return
        if getattr(self, "mono_raw", None) is not None: self.mono_raw.append(data); return
        if not data.strip() and "\n" in data and not self.stack: return
        parts = CITE.split(data)
        for i, part in enumerate(parts):
            if i % 2 == 1:
                keys = [k.strip().lstrip("@") for k in re.split(r"[,;]\s*", part) if k.strip()]
                for k in keys:
                    if k not in self.bibkeys: self.unknown.add(k)
                self.emit("\\cite{" + ",".join(keys) + "}")
            else:
                t = esc(part)
                t = xref_text(t, self.lab)
                self.emit(t)

    # ---- tags
    def handle_starttag(self, tag, attrs):
        a = dict(attrs); cls = a.get("class", ""); cs = cls.split()
        if self.skip: self.skip += 1 if tag == "div" else 0; return
        if tag == "div":
            if "chapter" in cs:
                self.chap = {"id": a.get("id"), "num": None, "title": None}; self.stack.append("chapter")
            elif "chapnum" in cs: self.stack.append("chapnum"); self.cell = []
            elif "chaptitle" in cs: self.stack.append("chaptitle"); self.cell = []
            elif "figure" in cs:
                self.fig = {"id": a.get("id"), "img": None, "width": "0.9", "buf": None, "cap": None}; self.stack.append("figure")
            elif "figbox" in cs:
                if self.fig is not None: self.fig["buf"] = []
                self.stack.append("figbox")
            elif "kw" in cs: self.stack.append("kw")
            else: self.stack.append("div")
        elif tag == "h2" or tag == "h3":
            self.stack.append(tag); self.cell = []; self.pending_label = a.get("id")
        elif tag == "p":
            if "figcap" in cs:
                self.in_caption = True; self.caption = []; self.stack.append("figcap")
            else:
                self.stack.append("p"); self.emit("\n\\noindent " if "noindent" in cs and not self.list_stack else "\n")
        elif tag == "b": self.emit("\\textbf{")
        elif tag == "i": self.emit("\\textit{")
        elif tag == "span":
            if "mono" in cs:
                self.mono_raw = [] if not self.in_caption else None; self.emit("\\texttt{") if self.in_caption else None; self.stack.append("mono")
            elif "tabnum" in cs: self.skip_span = True; self.stack.append("tabnum"); self.saved = self._redirect()
            else: self.emit("")
        elif tag == "blockquote": self.emit("\n\\begin{quote}\n")
        elif tag == "pre":
            self.in_pre = True; self.emit("\n\\begin{Verbatim}[fontsize=\\small,breaklines=true,breaksymbolleft={}]\n")
        elif tag == "ul": self.list_stack.append("itemize"); self.emit("\n\\begin{itemize}\n")
        elif tag == "ol": self.list_stack.append("enumerate"); self.emit("\n\\begin{enumerate}\n")
        elif tag == "li": self.emit("\\item ")
        elif tag == "table":
            self.table = {"rows": [], "caption": None, "label": None, "widths": None, "cls": cs}; self.stack.append("table")
        elif tag == "caption": self.in_caption = True; self.caption = []; self.stack.append("caption")
        elif tag == "tr": self.row = []
        elif tag in ("td", "th"):
            self.cell = []; self.cell_meta = {"rowspan": int(a.get("rowspan", 1)), "colspan": int(a.get("colspan", 1)), "th": tag == "th", "width": a.get("style", "")}
        elif tag == "img":
            if self.fig is not None:
                self.fig["img"] = a.get("src"); m = re.search(r"width:(\d+)%", a.get("style", "")); self.fig["width"] = f"{int(m.group(1)) / 100:.2f}" if m else "0.9"
        elif tag == "br": self.emit("\\\\ " if self.cell is not None else "\\newline ")
        elif tag == "a": pass
        elif tag == "thead" or tag == "tbody": pass

    def _redirect(self):
        # skip tabnum span content ("Table 5.1:") since LaTeX numbers captions itself
        cur = self.cell; self.cell = []; return cur

    def handle_endtag(self, tag):
        if self.skip:
            if tag == "div": self.skip -= 1
            return
        if tag == "span" and self.stack and self.stack[-1] == "tabnum":
            self.stack.pop(); self.cell = self.saved; return
        if tag == "div":
            top = self.stack.pop() if self.stack else None
            if top == "chapnum": self.chap["num"] = "".join(self.cell).strip(); self.cell = None
            elif top == "chaptitle":
                self.chap["title"] = "".join(self.cell).strip(); self.cell = None
                num = self.chap["num"] or ""; lab = f"ch:{self.chap['id']}"
                if num.startswith("Appendix") and not getattr(self, "in_appendix", False): self.out.append("\n\\appendix\n"); self.in_appendix = True
                if self.chap["id"] == "refs": self.out.append("\n\\printbibliography[heading=bibintoc,title={References}]\n"); self.skip = 1
                else: self.out.append(f"\n\\chapter{{{self.chap['title']}}}\\label{{{lab}}}\n")
            elif top == "figure":
                f = self.fig; self.fig = None
                lab = self.lab["fig"].get(self._cap_num(f["cap"] or ""), f"fig:{f['id']}")
                self.out.append("\n\\begin{figure}[htbp]\n\\centering\n")
                if f["img"]: self.out.append(f"\\includegraphics[width={f['width']}\\linewidth]{{{f['img']}}}\n")
                if f["buf"]: self.out.append("\\begin{minipage}{0.95\\linewidth}\n" + "".join(f["buf"]) + "\n\\end{minipage}\n")
                self.out.append(f"\\caption{{{f['cap'] or ''}}}\\label{{{lab}}}\n\\end{{figure}}\n")
            elif top == "chapter": self.chap = None
        elif tag in ("h2", "h3"):
            self.stack.pop(); txt = "".join(self.cell).strip(); self.cell = None
            txt = re.sub(r"^[A-Z0-9]+(?:\.[0-9]+)+~+", "", txt)  # drop the literal number; LaTeX numbers sections
            cmd = "section" if tag == "h2" else "subsection"
            self.out.append(f"\n\\{cmd}{{{txt}}}\\label{{sec:{self.pending_label}}}\n")
        elif tag == "p":
            top = self.stack.pop() if self.stack else None
            if top == "figcap":
                self.in_caption = False; cap = "".join(self.caption).strip(); self.caption = None
                if self.fig is not None: self.fig["cap"] = cap
            else: self.emit("\n")
        elif tag == "b" or tag == "i": self.emit("}")
        elif tag == "span":
            if self.stack and self.stack[-1] == "mono":
                self.stack.pop()
                if getattr(self, "mono_raw", None) is not None:
                    raw = "".join(self.mono_raw).replace("{", "\\{").replace("}", "\\}"); self.mono_raw = None; self.emit("\\path{" + raw + "}")
                else: self.emit("}")
        elif tag == "blockquote": self.emit("\\end{quote}\n")
        elif tag == "pre": self.in_pre = False; self.emit("\n\\end{Verbatim}\n")
        elif tag in ("ul", "ol"): env = self.list_stack.pop(); self.emit(f"\\end{{{env}}}\n")
        elif tag == "li": self.emit("\n")
        elif tag == "caption":
            self.in_caption = False; self.stack.pop(); self.table["caption"] = "".join(self.caption).strip(); self.caption = None
        elif tag in ("td", "th"):
            self.row.append({"text": "".join(self.cell).strip(), **self.cell_meta}); self.cell = None
        elif tag == "tr":
            if self.table is not None and self.row is not None: self.table["rows"].append(self.row); self.row = None
        elif tag == "table":
            self.stack.pop(); self.out.append(self.render_table(self.table)); self.table = None

    def _last_open_mono(self):
        # spans are only used for mono and tabnum; tabnum handled separately
        return True

    def _cap_num(self, cap): return ""

    # ---- tables
    def render_table(self, t):
        rows = t["rows"]
        if not rows: return ""
        ncol = max(sum(c["colspan"] for c in r) for r in rows)
        # column widths from the header's style attributes, else equal
        widths = [None] * ncol
        for i, c in enumerate(rows[0]):
            m = re.search(r"width:(\d+)%", c.get("width", "")); widths[i] = int(m.group(1)) / 100 if m else None
        rest = 1 - sum(w for w in widths if w); nfree = sum(1 for w in widths if w is None)
        widths = [w if w else rest / max(nfree, 1) for w in widths]
        widths = [max(w, 0.16) for w in widths]; tot = sum(widths); widths = [w / tot for w in widths]  # no column narrower than its bold header
        tight = "tight" in t.get("cls", []); scale = (not tight) and ncol >= 7  # wide numeric tables: natural columns, scaled to the text width
        if tight: spec = "l" * ncol; size = "\\footnotesize"  # wide numeric tables: natural columns at 10 pt, as in other BTH theses
        if tight and ncol >= 7: scale = True  # still too wide at 10 pt: scale to the text width
        elif scale: spec = "l" * ncol; size = "\\footnotesize"  # very wide tables (7+ columns): scaled to the text width
        else:
            spec = "".join(f">{{\\raggedright\\arraybackslash}}p{{{w * 0.86:.3f}\\linewidth}}" for w in widths)
            size = "\\footnotesize"
        cap = t["caption"] or ""; num = None
        lab = self.table_label if hasattr(self, "table_label") else None
        lines = ["\n\\begin{table}[htbp]", "\\centering", size, f"\\caption{{{cap}}}"]
        if self.cur_table_label: lines.append(f"\\label{{{self.cur_table_label}}}")
        lines.append(("\\resizebox{\\linewidth}{!}{" if scale else "") + f"\\begin{{tabular}}{{{spec}}}"); lines.append("\\toprule")
        pending = {}  # col -> rows remaining (rowspan)
        for ri, r in enumerate(rows):
            cells = []; ci = 0; k = 0
            while ci < ncol:
                if pending.get(ci, 0) > 0: cells.append(""); pending[ci] -= 1; ci += 1; continue
                if k >= len(r): cells.append(""); ci += 1; continue
                c = r[k]; k += 1; txt = c["text"]
                if c["th"]: txt = f"\\textbf{{{txt}}}"
                if c["rowspan"] > 1:
                    pending[ci] = c["rowspan"] - 1; txt = f"\\multirow{{{c['rowspan']}}}{{{'*' if (tight or scale) else '='}}}{{{txt}}}"
                if c["colspan"] > 1:
                    cells.append(f"\\multicolumn{{{c['colspan']}}}{{p{{{sum(widths[ci:ci + c['colspan']]) * 0.86:.3f}\\linewidth}}}}{{{txt}}}"); ci += c["colspan"]
                else: cells.append(txt); ci += 1
            lines.append(" & ".join(cells) + " \\\\")
            if ri == 0 and r and r[0]["th"]: lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}" + ("}" if scale else ""), "\\end{table}\n"]
        return "\n".join(lines)


def convert(fragment, lab, bibkeys):
    # assign table labels by caption number before parsing (caption text is available in the HTML)
    c = Conv(lab, bibkeys); c.cur_table_label = None
    # pre-scan tables to attach labels in order
    tab_nums = re.findall(r'<table class="data[^"]*">\s*<caption><span class="tabnum">Table ([A-Z0-9]+\.[0-9]+):', fragment)
    fig_nums = re.findall(r'<div class="figure" id="[^"]+">.*?<span class="tabnum">Figure ([A-Z0-9]+\.[0-9]+):', fragment, re.S)
    tab_iter = iter(tab_nums); fig_iter = iter(fig_nums)
    orig_start = c.handle_starttag
    def start(tag, attrs):
        if tag == "table":
            n = next(tab_iter, None); c.cur_table_label = lab["tab"].get(n) if n else None
        if tag == "div" and "figure" in dict(attrs).get("class", "").split():
            n = next(fig_iter, None); c.pending_fig_label = lab["fig"].get(n) if n else None
        orig_start(tag, attrs)
    c.handle_starttag = start
    orig_end = c.handle_endtag
    def end(tag):
        if tag == "div" and c.stack and c.stack[-1] == "figure" and getattr(c, "pending_fig_label", None):
            f = c.fig; c.fig = None; c.stack.pop()
            c.out.append("\n\\begin{figure}[htbp]\n\\centering\n")
            if f["img"]: c.out.append(f"\\includegraphics[width={f['width']}\\linewidth]{{{f['img']}}}\n")
            if f["buf"]: c.out.append("\\begin{minipage}{0.95\\linewidth}\n" + "".join(f["buf"]) + "\n\\end{minipage}\n")
            c.out.append(f"\\caption{{{f['cap'] or ''}}}\\label{{{c.pending_fig_label}}}\n\\end{{figure}}\n"); c.pending_fig_label = None; return
        orig_end(tag)
    c.handle_endtag = end
    c.feed(fragment); c.close()
    return "".join(c.out), c.unknown


# ------------------------------------------------------------------ front matter (BTH template)
TMPL = HERE.parent.parent / "DP_thesis_tmpl"


def front_matter(front_html, lab, bibkeys):
    g = lambda pat: (re.search(pat, front_html, re.S) or [None, ""])[1]
    title = html.unescape(g(r'<div class="tp-title">(.*?)</div>')); sub = html.unescape(g(r'<div class="tp-subtitle">(.*?)</div>'))
    top = html.unescape(re.sub(r"<span.*?</span>", "", g(r'<div class="tp-top">(.*?)</div>'), flags=re.S)).replace("<br>", "|").strip()
    degree, date = [x.strip() for x in top.split("|")[:2]]; month, year = date.rsplit(" ", 1)
    faculty = html.unescape(g(r'Faculty of ([A-Za-z ]+?), Blekinge'))
    weeks = g(r'equivalent to (\d+) weeks')
    authors = [a.split("|") for a in html.unescape(g(r'<div class="tp-author">(.*?)</div>')).split("<br>") if a.strip() and a.strip() != "&nbsp;"]
    a1 = authors[0] if authors else ["Firstname Lastname", "...@student.bth.se"]; a2 = authors[1] if len(authors) > 1 else ["", ""]
    sup = html.unescape(re.sub("<[^>]+>", "\n", g(r'University supervisor:<br>(.*?)</p>'))).strip().split("\n")
    sup_name = sup[0].strip(); sup_dept = re.sub(r"^Department of ", "", sup[-1].strip()) if len(sup) > 1 else "Computer Science"
    data = f"""% THESIS DATA (from chapters/00_front.html; edit there and re-run to_latex.py)
\\newcommand{{\\thesisDegree}}{{{esc(degree)}}}
\\newcommand{{\\thesisMonth}}{{{esc(month)}}}
\\newcommand{{\\thesisYear}}{{{esc(year)}}}
\\newcommand{{\\faculty}}{{{esc(faculty)}}}
\\newcommand{{\\thesisWeeks}}{{{weeks or '20'}}}
\\newcommand{{\\thesisTitle}}{{{esc(title)}}}
\\newcommand{{\\thesisSubtitle}}{{{esc(sub)}}}
\\newcommand{{\\authorFirst}}{{{esc(a1[0].strip())}}}
\\newcommand{{\\authorFirstMail}}{{{esc(a1[1].strip() if len(a1) > 1 else '')}}}
\\newcommand{{\\authorSecond}}{{{esc(a2[0].strip())}}}
\\newcommand{{\\authorSecondMail}}{{{esc(a2[1].strip() if len(a2) > 1 else '')}}}
\\newcommand{{\\super}}{{{esc(sup_name)}}}
\\newcommand{{\\superAffiliation}}{{{esc(sup_dept)}}}
"""
    sections = re.findall(r'<div class="frontsection[^"]*" id="([^"]+)">\s*<div class="fmheading">(.*?)</div>(.*?)</div>\s*(?=<!--|$)', front_html, re.S)
    body = {}
    for fid, head, sec in sections:
        if fid == "abbr":
            rows = re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>", sec, re.S)
            t = "\\begin{longtable}{@{}p{0.16\\linewidth}p{0.8\\linewidth}@{}}\n"
            for k, v in rows:
                kt, _ = convert(f"<p>{k}</p>", lab, bibkeys); vt, _ = convert(f"<p>{v}</p>", lab, bibkeys)
                t += f"{kt.strip()} & {vt.strip()} \\\\\n"
            t += "\\end{longtable}\n"
        else:
            if fid == "abstract": sec = sec.replace("<p>", '<p class="noindent">')
            t, _ = convert(sec, lab, bibkeys)
        if fid == "abstract":
            t = t.replace("\n\\noindent \\textbf{Keywords:}", "\n\\vspace{\\baselineskip}\n\\noindent\n\\textbf{Keywords:}")
        body[fid] = t
    tex = [data, TEMPLATE_FRONT]
    tex.append("% ABSTRACT IN ENGLISH\n% -------------------\n\\abstract\n" + body.get("abstract", "") + "\n\\cleardoublepage\n% -------------------\n\n")
    tex.append("% ABSTRACT IN SWEDISH: only needed for civilingenjor theses; not included.\n\n")
    tex.append("% DATA AVAILABILITY STATEMENT\n% ---------------------------\n\\das\n" + body.get("das", "") + "\n\\cleardoublepage\n% -------------------\n\n")
    tex.append("% ACKNOWLEDGMENTS\n% -------------------\n\\acknowledgments\n" + body.get("ack", "") + "\n\\cleardoublepage\n% -------------------\n\n")
    for fid, head in (("genai", "Declaration of Generative AI Usage"), ("abbr", "Abbreviations")):
        if fid in body:
            tex.append(f"\\chapter*{{{head}}}\\addcontentsline{{toc}}{{chapter}}{{{head}}}\\markboth{{{head}}}{{{head}}}\n{body[fid]}\n\\cleardoublepage\n\n")
    tex.append("""% TABLE OF CONTENTS PAGES
\\setcounter{secnumdepth}{3}
\\tableofcontents
\\listoffigures
\\listoftables

\\cleardoublepage
\\pagestyle{headings}
\\pagenumbering{arabic}

""")
    return "".join(tex)


# The BTH template's front page and inner page, verbatim (bth-thesis.tex v4.2, "please do not change").
TEMPLATE_FRONT = r"""
% DOCUMENT BEGINS HERE
\begin{document}

\pagestyle{plain}
\pagenumbering{roman}

% THESIS FRONT PAGE (please do not change)
% ----------------------------------------
{\pagestyle{empty}
\changepage{3cm}{1cm}{-0.5cm}{-0.5cm}{}{-1.5cm}{}{}{}
\noindent
\begin{tabular}{@{}p{0.75\textwidth} p{0.25\textwidth}}
\thesisDegree & \hfill\multirow{3}{*}{\bthcsnotextlogo{3cm}} \\
\thesisMonth \ \thesisYear & \\
\end{tabular}

\center
\vspace {7.5cm}
{\Huge\textbf{\thesisTitle}}

\ifx\thesisSubtitle\empty\else
\vspace {0.5cm}
{\Large\textbf{\thesisSubtitle}}
\fi

\vspace{2cm}
{\Large\textbf{\authorFirst}}

\vspace{0.3cm}
{\Large\textbf{\authorSecond}}

\vspace*{\fill}

\noindent\makebox[\linewidth]{\rule{\textwidth}{1pt}}
Faculty of \faculty, Blekinge Institute of Technology, 371 79 Karlskrona, Sweden

\clearpage
} % Back to \pagestyle{plain}
% ----------------------------------------


% THESIS INNER PAGE (please do not change)
% ----------------------------------------
{\pagestyle{empty}
\changepage{3cm}{1cm}{-0.5cm}{-0.5cm}{}{-1.5cm}{}{}{}

{\small
\noindent
This thesis is submitted to the Faculty of \faculty\ at Blekinge Institute
of Technology in partial fulfillment of the requirements for the degree of
\thesisDegree. The thesis is equivalent to \thesisWeeks\ weeks of full-time studies.

\vspace{1cm}

\noindent
The authors declare that they are the sole authors of this thesis and that they have
not used any sources other than those listed in the bibliography and identified as references.
They further declare that they have not submitted this thesis at any other institution to
obtain a degree.
}

\vspace{10cm}

\noindent
\textbf{Contact Information:} \\
Author(s): \\
\authorFirst \\
E-mail: \authorFirstMail
% Second author block only when a second author is given (template: delete when absent)
\ifx\authorSecond\empty\else \\
\\
\authorSecond \\
E-mail: \authorSecondMail
\fi

\vspace{2cm}

\noindent
University advisor: \\
\super \\
Department of \superAffiliation

\vspace*{\fill}

\noindent
\begin{tabular}{@{}p{0.5\textwidth} l c l}
Faculty of \faculty              & Internet & : & www.bth.se \\
Blekinge Institute of Technology & Phone    & : & +46 455 38 50 00 \\
SE--371 79 Karlskrona, Sweden    & Fax      & : & +46 455 38 50 57 \\
\end{tabular}
\clearpage
} % Back to \pagestyle{plain}
% ----------------------------------------

\setcounter{page}{1}

%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR TEXTS START HERE
%%%%%%%%%%%%%%%%%%%%%%%%

"""

TEMPLATE_END = r"""

% All references are in a separate file: refs.bib (generated from the verified bibliography)
\bibliography{refs}
\bibliographystyle{IEEEtranS}

"""

TEMPLATE_LAST = r"""

% DO NOT CHANGE BELOW
% This part makes sure that the last page is even with BTH-logo.
% -------------------
\cleardoublepage
\thispagestyle{empty}
\vspace*{\fill}
\clearpage{\thispagestyle{empty}}
\changepage{3cm}{1cm}{-0.5cm}{-0.5cm}{}{-1.5cm}{}{}{}
\vspace*{\fill}
\center

{\bthcsnotextlogo{3cm}}
\\
\noindent\makebox[\linewidth]{\rule{\textwidth}{1pt}}
Faculty of \faculty, Blekinge Institute of Technology, 371 79 Karlskrona, Sweden
% -------------------

\end{document}
"""

PREAMBLE = r"""% Generated by thesis/v3/to_latex.py from the thesis v3 chapter sources.
% Document class and preamble: BTH thesis template v4.2 (June 17, 2026), DP_thesis_tmpl/bth-thesis.tex.
% Build: pdflatex thesis && bibtex thesis && pdflatex thesis && pdflatex thesis   (or: make)
\documentclass[a4paper,twoside]{bth}
% BTH THESIS TEMPLATE
%--------------------
% Template version 4.2 -- June 17, 2026
%--------------------

"""

PACKAGES = r"""
% PACKAGES AND COMMANDS START
%----------------------------
% please do not delete or change anything before the END of this section
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{mathenv}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{textcomp}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{pifont}
\usepackage{changepage}
\usepackage{listings}
\usepackage{nameref}
\usepackage{hyperref}
\usepackage{xspace}
\usepackage{xtab}
\usepackage{enumitem}
%\usepackage[sort&compress,numbers,square,comma]{natbib} % natbib interferes with the style; use \cite instead (see below)
\usepackage{cite} % works only with numeric citations; comment out when using author-year styles
\usepackage[color=blue!10,textsize=footnotesize,textwidth=25mm]{todonotes}
\DeclareGraphicsExtensions{.pdf}

\newtheorem{lem}{\textsc{Lemma}}[chapter]
\newtheorem{thm}{\textsc{Theorem}}[chapter]
\newtheorem{prop}{\textsc{Proposition}}[chapter]
\newtheorem{post}{Postulate}[chapter]
\newtheorem{corr}{\textsc{Corollary}}[chapter]
\newtheorem{defs}{\textsc{Definition}}[chapter]
\newtheorem{cons}{\textsc{Constraint}}[chapter]
\newtheorem{ex}{\textbf{Example}}[chapter]
\newtheorem{qu}{\textbf{Question}}[chapter]
% -------------------------
% PACKAGES AND COMMANDS END

% Additions for this thesis (after the template's protected section)
\DeclareGraphicsExtensions{.pdf,.png}
% mathenv (template) loads mdwtab, which replaces the array package and lacks \arraybackslash;
% mdwtab resets \\ itself after every >{...} preamble, so an empty definition is correct.
\providecommand{\arraybackslash}{\let\\\tabularnewline}
\usepackage{fvextra}
% Body font: Computer Modern (cm-super Type 1), as in the template and other BTH theses.
\hypersetup{colorlinks=true,linkcolor=[rgb]{0.10,0.24,0.49},citecolor=[rgb]{0.10,0.24,0.49},urlcolor=[rgb]{0.10,0.24,0.49}}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3.5pt}
\sloppy
\Urlmuskip=0mu plus 1mu
\setcounter{tocdepth}{2}
"""


def main():
    bib = load_bib(); bibkeys = set(bib)
    front = (CH / "00_front.html").read_text()
    body = "".join(Path(f).read_text() + "\n" for f in sorted(CH.glob("ch*.html")))
    apps = "".join(Path(f).read_text() + "\n" for f in sorted(CH.glob("ap*.html")))
    lab = build_label_map(front + body + apps)
    body_tex, unk1 = convert(body, lab, bibkeys)
    apps_tex, unk2 = convert(apps, lab, bibkeys)
    front_tex = front_matter(front, lab, bibkeys)
    tex = PREAMBLE + front_tex.split(TEMPLATE_FRONT)[0] + PACKAGES + TEMPLATE_FRONT + front_tex.split(TEMPLATE_FRONT)[1] + body_tex + TEMPLATE_END + apps_tex + TEMPLATE_LAST
    (OUT / "thesis.tex").write_text(tex)
    used = set(re.findall(r"\\cite\{([^}]+)\}", tex)); used = {k for grp in used for k in grp.split(",")}
    (OUT / "refs.bib").write_text("".join(bibtex(bib[k]) for k in sorted(used) if k in bib))
    for img in set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)):
        src = HERE / img
        if src.exists(): shutil.copy(src.resolve(), OUT / img)
    for f in ("bth.cls", "changepage.sty", "bthnotext.pdf"):
        shutil.copy(TMPL / f, OUT / f)
    (OUT / "Makefile").write_text("""# Build the thesis PDF from the LaTeX sources (BTH template: pdflatex + bibtex, style IEEEtranS).
all:
\t-pdflatex -interaction=nonstopmode thesis.tex
\tbibtex thesis
\t-pdflatex -interaction=nonstopmode thesis.tex
\tpdflatex -interaction=nonstopmode thesis.tex
clean:
\trm -f thesis.aux thesis.bbl thesis.blg thesis.lof thesis.lot thesis.out thesis.toc thesis.log
""")
    report = {"unknown_keys": sorted(unk1 | unk2), "citations": len(used), "figures": tex.count("\\begin{figure}"), "tables": tex.count("\\begin{table}"),
              "chapters": tex.count("\\chapter{"), "words": len(re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", tex).split())}
    print(json.dumps(report, indent=1)); print("wrote", OUT / "thesis.tex")


if __name__ == "__main__":
    main()
