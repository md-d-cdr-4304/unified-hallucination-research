# Bibliography

`lit_*.json` — a list of entries per file, merged by `build.py` and
`to_latex.py` in file-name order; the first file wins on a duplicate `key`.

Entry fields: `key`, `authors` (list), `title`, venue, `year`, and an origin
(`doi`, `arxiv` or `url`), which is printed in the reference list and made
clickable by hyperref in the LaTeX build.

Citation keys used in the chapters but missing here are reported as
`unknown_keys` in `build_report.json`; that list must be empty at submission.
