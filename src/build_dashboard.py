"""Build the dashboard.

outputs/barca_signings_ledger.html  page body only (published as a claude.ai artifact, which adds its own skeleton)
site/index.html                     complete stand-alone page, served by Vercel (see vercel.json)
"""
import json
from .config import ROOT, OUTPUTS
from .export_dashboard import export

SITE = ROOT / "site"

SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex, nofollow">
<style>html{{color-scheme:light}} body{{margin:0}} img{{max-width:100%}} [hidden]{{display:none!important}}</style>
{head}
</head>
<body>
{body}
</body>
</html>
"""


def build():
    data = export()
    body = (ROOT / "dashboard" / "template.html").read_text().replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    (OUTPUTS / "barca_signings_ledger.html").write_text(body)
    SITE.mkdir(exist_ok=True)
    # title, meta, font links and styles go in <head>; the rest is the page
    cut = body.index('<div class="wrap">')
    (SITE / "index.html").write_text(SKELETON.format(head=body[:cut].strip(), body=body[cut:]))
    print("wrote", OUTPUTS / "barca_signings_ledger.html", "and", SITE / "index.html")


if __name__ == "__main__":
    build()
