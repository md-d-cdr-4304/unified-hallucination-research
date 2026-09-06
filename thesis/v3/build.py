"""Thesis v3 builder: chapters/*.html fragments -> BTH-style PDF.
- merges bib/*.json, replaces [@key] citations with IEEE numbers in order of first use,
  renders the reference list with DOI / arXiv / URL origins
- auto-generates Contents, List of Figures, List of Tables from headings/captions
- three-document merge (title pages | roman front matter | arabic body) as in build_thesis.py
- writes build_report.json (unknown keys, missing figures, DATA-NEEDED/CITE-NEEDED markers, page counts)
Usage: python build.py [--out THESIS_v2.pdf] [--no-pdf]
"""
import re, json, sys, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
CH = HERE / "chapters"; BIB = HERE / "bib"; FIGS = HERE / "figures"
CSS = (HERE.parent / "part1_front.html").read_text().split("</head>")[0].split("<style>")[1].split("</style>")[0]
CSS = CSS.replace('.toc div a::after { content: leader(" . ") target-counter(attr(href), page); }\n.toc div.roman a::after { content: leader(" . ") target-counter(attr(href), page, lower-roman); }',
                  '.toc div[data-pg]::after { content: leader(" . ") attr(data-pg); }')
CSS += """
.figure { margin: 0.9em 0; page-break-inside: avoid; text-align: center; }
.figure img { max-width: 100%; }
.figcap { text-align: justify; }
table.data { page-break-inside: auto; }
table.data tr { page-break-inside: avoid; }
table.data.keep { page-break-inside: avoid; }
table.data.tight td, table.data.tight th { white-space: nowrap; font-size: 9.2pt; padding-left: 3pt; padding-right: 3pt; }
.chapter.first { page-break-before: auto; }
table.lst { width: 100%; border-collapse: collapse; margin-top: 0.6em; font-size: 10.5pt; }
table.lst td { padding: 3pt 4pt; vertical-align: top; border-bottom: 0.4pt solid #d0d0d0; }
table.lst td.lnum { width: 16%; white-space: nowrap; }
table.lst td.lpg { width: 8%; text-align: right; }
table.lst a { color: #000; text-decoration: none; }
blockquote { margin: 0.5em 0 0.5em 2.2em; }
blockquote p { text-indent: 0; }
.abbr td { padding: 1pt 8pt 1pt 0; vertical-align: top; }
.refs p { font-size: 9.8pt; }
h2, h3 { page-break-after: avoid; break-after: avoid; }
p { orphans: 3; widows: 3; }
h2 + p, h3 + p { orphans: 6; }
.kw { page-break-inside: avoid; }
table.data caption { page-break-after: avoid; break-after: avoid; }
a.cite { color: #1a3d7c; text-decoration: none; }
a.ref { color: #1a3d7c; text-decoration: none; word-break: normal; overflow-wrap: anywhere; }
"""

def load_bib():
    bib = {}
    for f in sorted(BIB.glob("lit_*.json")):
        try: entries = json.loads(Path(f).read_text())
        except Exception as e: print("BIB PARSE ERROR", f, e); continue
        for e in entries:
            k = e.get("key")
            if not k: continue
            if k in bib: continue  # first file wins
            bib[k] = e
    return bib

def ieee(e, n):
    au = e.get("authors") or []
    if len(au) > 6: a = ", ".join(au[:6]) + ", et al."
    elif len(au) > 1: a = ", ".join(au[:-1]) + ", and " + au[-1]
    else: a = au[0] if au else ""
    t = e.get("title", "").rstrip(".")
    venue = e.get("venue", ""); year = e.get("year", ""); vol = e.get("volume", ""); pages = e.get("pages", "")
    typ = e.get("type", "")
    s = f"[{n}] {a}, “{t},” "
    if typ == "article": s += f"<i>{venue}</i>" + (f", vol. {vol}" if vol else "") + (f", pp. {pages}" if pages else "") + f", {year}."
    elif typ == "inproceedings": s += f"in <i>{venue}</i>" + (f", pp. {pages}" if pages else "") + f", {year}."
    elif typ == "preprint": s += (f"<i>{venue}</i>, " if venue and venue.lower() != "arxiv" else "") + f"{year}."
    else: s += (f"<i>{venue}</i>, " if venue else "") + f"{year}."
    origin = []
    if e.get("doi"): origin.append(f'doi: <a class="ref" href="https://doi.org/{e["doi"]}">{e["doi"]}</a>')
    if e.get("arxiv"): origin.append(f'arXiv: <a class="ref" href="https://arxiv.org/abs/{e["arxiv"]}">{e["arxiv"]}</a>')
    if e.get("url") and not (e.get("doi") or e.get("arxiv")): origin.append(f'[Online]. Available: <a class="ref" href="{e["url"]}">{e["url"]}</a>')
    if origin: s += " " + "; ".join(origin) + "."
    return s

def main():
    argv = sys.argv[1:]; out = HERE / (argv[argv.index("--out") + 1] if "--out" in argv else "html_build/THESIS_v3_HTML.pdf")  # the BTH-template PDF is built from latex/ (to_latex.py + make)
    bib = load_bib(); report = {"bib_entries": len(bib), "unknown_keys": [], "missing_figures": [], "markers": [], "duplicate_ids": []}
    front = (CH / "00_front.html").read_text() if (CH / "00_front.html").exists() else "<div class='frontsection abstract' id='abstract'><div class='fmheading'>Abstract</div><p>[[front matter missing]]</p></div>"
    chapter_files = sorted(CH.glob("ch*.html")); app_files = sorted(CH.glob("ap*.html"))
    body = "".join(Path(f).read_text() + "\n" for f in chapter_files)
    apps = "".join(Path(f).read_text() + "\n" for f in app_files)
    # ---- citations ----
    order = []
    def cite(m):
        keys = [k.strip().lstrip("@") for k in re.split(r"[,;]\s*", m.group(1)) if k.strip()]
        nums = []
        for k in keys:
            if k not in bib:
                report["unknown_keys"].append(k); nums.append(f"?{k}"); continue
            if k not in order: order.append(k)
            n = order.index(k) + 1
            nums.append(f'<a class="cite" href="#ref-{n}">{n}</a>')
        return "[" + ", ".join(nums) + "]"
    pat = re.compile(r"\[@([A-Za-z0-9_:\-]+(?:\s*[,;]\s*@?[A-Za-z0-9_:\-]+)*)\]")
    front, body, apps = (pat.sub(cite, x) for x in (front, body, apps))
    refs = '<div class="chapter refs" id="refs">\n  <div class="chaptitle">References</div>\n' + "\n".join(f'  <p id="ref-{i + 1}">{ieee(bib[k], i + 1)}</p>' for i, k in enumerate(order)) + "\n</div>\n"
    # ---- table ids ----
    tcount = [0]
    def tab_id(m):
        tcount[0] += 1; cap = m.group(2); num = re.search(r"Table\s+([A-Z0-9]+\.[0-9]+)", cap)
        tid = "tab" + (num.group(1).replace(".", "-") if num else str(tcount[0]))
        return f'<table class="data" id="{tid}"{m.group(1)}><caption>{cap}</caption>'
    tpat = re.compile(r'<table class="data"([^>]*)>\s*<caption>(.*?)</caption>', re.S)
    body, apps = (tpat.sub(tab_id, x) for x in (body, apps))
    # ---- keep each heading with the block that follows it (paragraph, table, figure, or code); a short first paragraph pulls the next block in ----
    BLOCK = r'(?:<table class="data[^"]*">.*?</table>|<div class="figure" id="[^"]+">.*?</p></div>|<pre>.*?</pre>|<p[^>]*>.*?</p>|<ul>.*?</ul>|<ol>.*?</ol>)'
    def wrap(level, x):
        pat = re.compile(r'(<h%s id="[^"]+">.*?</h%s>)\s*(%s)(?:\s*(%s))?' % (level, level, BLOCK, BLOCK), re.S)
        def f(m):
            first, second = m.group(2), m.group(3) or ""
            if first.startswith("<table") and first.count("<tr") <= 12 and 'class="data keep"' not in first:
                first = first.replace('<table class="data"', '<table class="data keep"', 1)
            keep = first + ("\n" + second if second and first.startswith("<p") and len(re.sub("<[^>]+>", "", first)) < 260 else "")
            rest = m.group(0)[len(m.group(1)):].lstrip()[len(keep.replace("\n" + second, "").rstrip()):] if False else None
            tail = "" if keep.endswith(second) and second else ("\n" + second if second else "")
            return f'<div class="kw">{m.group(1)}\n{keep}</div>{tail}'
        return pat.sub(f, x)
    body, apps = (wrap(2, wrap(3, x)) for x in (body, apps))
    # an h2 directly followed by an h3 group moves with it
    kw2 = re.compile(r'(<div class="kw"><h2 id="[^"]+">(?:(?!<div class="kw">).)*?</div>|<h2 id="[^"]+">(?:(?!<div class="kw">).)*?</h2>)\s*(<div class="kw"><h3(?:(?!<div class="kw">).)*?</div>)', re.S)
    def merge(m):
        return f'<div class="kw">{m.group(1)}\n{m.group(2)}</div>' if len(re.sub("<[^>]+>", "", m.group(1))) < 500 else m.group(0)
    body, apps = (kw2.sub(merge, x) for x in (body, apps))
    # ---- headings for TOC ----
    full = body + refs + apps
    toc = []
    for m in re.finditer(r'<div class="chapter(?: refs)?" id="([^"]+)">\s*(?:<div class="chapnum">([^<]*)</div>\s*)?<div class="chaptitle">(.*?)</div>', full, re.S):
        cid, num, title = m.group(1), m.group(2) or "", re.sub("<[^>]+>", "", m.group(3)).strip()
        label = (num.replace("Chapter ", "").replace("Appendix ", "") + "&nbsp;&nbsp;" if num else "") + title
        toc.append(("l0", cid, label, m.start()))
    for m in re.finditer(r'<h([23]) id="([^"]+)">(.*?)</h\1>', full, re.S):
        toc.append(("l1" if m.group(1) == "2" else "l2", m.group(2), re.sub("<[^>]+>", "", m.group(3)).strip(), m.start()))
    toc.sort(key=lambda t: t[3])
    toc_html = '<div class="toc" id="toc">\n  <div class="fmheading">Contents</div>\n'
    for fid, lab in [("abstract", "Abstract"), ("das", "Data Availability Statement"), ("ack", "Acknowledgements"), ("abbr", "Abbreviations"), ("lof", "List of Figures"), ("lot", "List of Tables")]:
        if f'id="{fid}"' in front or fid in ("lof", "lot"): toc_html += f'  <div class="l0 roman"><a href="#{fid}">{lab}</a></div>\n'
    for lvl, cid, label, _ in toc:
        if lvl == "l2": continue  # keep TOC to two levels
        toc_html += f'  <div class="{lvl}"><a href="#{cid}">{label}</a></div>\n'
    toc_html += "</div>\n"
    # ---- LoF / LoT ----
    figs = re.findall(r'<div class="figure" id="([^"]+)">.*?<span class="tabnum">(Figure [A-Z0-9]+\.[0-9]+):</span>\s*(.*?)</p>', full, re.S)
    tabs = re.findall(r'<table class="data" id="([^"]+)"[^>]*>\s*<caption><span class="tabnum">(Table [A-Z0-9]+\.[0-9]+):</span>\s*(.*?)</caption>', full, re.S)
    def short(c): c = re.sub("<[^>]+>", "", c).strip(); return (c.split(". ")[0] + ".") if len(c) > 160 else c
    def lst(fid, title, rows):
        h = f'<div class="toc {fid}" id="{fid}">\n  <div class="fmheading">{title}</div>\n  <table class="lst">\n'
        for i, n, c in rows:
            h += f'    <tr><td class="lnum"><a href="#{i}">{n}</a></td><td class="lcap"><a href="#{i}">{short(c)}</a></td><td class="lpg" data-anchor="{i}"></td></tr>\n'
        return h + "  </table>\n</div>\n"
    lof = lst("lof", "List of Figures", figs); lot = lst("lot", "List of Tables", tabs)
    # ---- checks ----
    for m in re.finditer(r'<img src="([^"]+)"', full):
        if not (HERE / m.group(1)).exists(): report["missing_figures"].append(m.group(1))
    report["markers"] = re.findall(r"\[\[(DATA-NEEDED|CITE-NEEDED)[^\]]*\]\]", front + full)
    ids = re.findall(r' id="([^"]+)"', full); report["duplicate_ids"] = sorted({i for i in ids if ids.count(i) > 1})
    report["n_figures"] = len(figs); report["n_tables"] = len(tabs); report["n_citations_used"] = len(order)
    report["words_body"] = len(re.sub("<[^>]+>", " ", body).split()); report["words_total"] = len(re.sub("<[^>]+>", " ", full).split())
    # ---- assemble ----
    head = f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Thesis</title><style>{CSS}</style></head><body>\n"
    FOOT_ROMAN = '@bottom-center { content: counter(page, lower-roman); font-family: "Latin Modern Roman"; font-size: 10pt; }'
    def with_footer(f): return head.replace(FOOT_ROMAN, f, 1)
    i_abs = front.index("<!-- ABSTRACT -->") if "<!-- ABSTRACT -->" in front else 0
    body_a = front[:i_abs]; body_b = front[i_abs:] + lof + lot + toc_html
    body_b = body_b.replace('<div class="frontsection abstract"', '<div class="abstract"', 1)
    body_c = re.sub(r'<div class="chapter" id="ch1">', '<div class="chapter first" id="ch1">', body + refs + apps, count=1)
    (HERE / "_full.html").write_text(head + body_a + body_b + body_c + "</body></html>")
    if "--no-pdf" in argv:
        (HERE / "build_report.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1)); return
    from weasyprint import HTML
    from pypdf import PdfWriter
    doc_a = with_footer("@bottom-center { content: none; }") + body_a + "</body></html>"
    doc_b = with_footer(FOOT_ROMAN) + body_b + "</body></html>"
    doc_c = with_footer('@bottom-center { content: counter(page); font-family: "Latin Modern Roman"; font-size: 10pt; }') + body_c + "</body></html>"
    rb = HTML(string=doc_b, base_url=str(HERE)).render(); rc = HTML(string=doc_c, base_url=str(HERE)).render()
    pb, pc = {}, {}
    for i, pg in enumerate(rb.pages):
        for n in pg.anchors: pb.setdefault(n, i)
    for i, pg in enumerate(rc.pages):
        for n in pg.anchors: pc.setdefault(n, i)
    ROM = ["i","ii","iii","iv","v","vi","vii","viii","ix","x","xi","xii","xiii","xiv","xv","xvi","xvii","xviii","xix","xx","xxi","xxii","xxiii","xxiv","xxv","xxvi","xxvii","xxviii","xxix","xxx"]
    def disp(n):
        if n in pc: return str(pc[n] + 1)
        if n in pb: return ROM[pb[n]] if pb[n] < len(ROM) else str(pb[n] + 1)
        return "?"
    doc_b2 = re.sub(r'<div class="(l\d(?: roman)?)"><a href="#([^"]+)">', lambda m: f'<div class="{m.group(1)}" data-pg="{disp(m.group(2))}"><a href="#{m.group(2)}">', doc_b)
    doc_b2 = re.sub(r'<td class="lpg" data-anchor="([^"]+)"></td>', lambda m: f'<td class="lpg">{disp(m.group(1))}</td>', doc_b2)
    w = PdfWriter(); n_pages = []; rendered = []
    for h in (doc_a, doc_b2, doc_c):
        tmp = HERE / "_tmp.pdf"; r = HTML(string=h, base_url=str(HERE)).render(); r.write_pdf(tmp); w.append(str(tmp)); n_pages.append(len(r.pages)); rendered.append(r)
    # Links from the front matter (abstract citations, contents, lists) into the body are dropped by WeasyPrint
    # because the target lives in another document. Re-create them as annotations in the merged PDF.
    from pypdf.annotations import Link
    off_b, off_c = n_pages[0], n_pages[0] + n_pages[1]; n_fixed = 0
    for i, pg in enumerate(rendered[1].pages):
        H = pg.height
        for link in pg.links:
            if link[0] != "internal" or link[1] not in pc: continue
            x1, y1, x2, y2 = link[2]
            rect = (x1 * 0.75, (H - y2) * 0.75, x2 * 0.75, (H - y1) * 0.75)
            w.add_annotation(page_number=off_b + i, annotation=Link(rect=rect, target_page_index=off_c + pc[link[1]])); n_fixed += 1
    report["cross_links_added"] = n_fixed
    with open(out, "wb") as f: w.write(f)
    (HERE / "_tmp.pdf").unlink(missing_ok=True)
    report["pages"] = {"title": len(HTML(string=doc_a, base_url=str(HERE)).render().pages), "roman": len(rb.pages), "arabic": len(rc.pages)}
    report["pages"]["total"] = sum(report["pages"].values())
    (HERE / "build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "markers"}, indent=1)); print("markers:", len(report["markers"])); print("wrote", out)

if __name__ == "__main__":
    main()
