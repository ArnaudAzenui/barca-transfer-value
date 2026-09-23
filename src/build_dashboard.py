"""Inline the data into dashboard/template.html -> outputs/barca_signings_ledger.html"""
from .config import ROOT, OUTPUTS
from .export_dashboard import export
import json

if __name__ == "__main__":
    data = export()
    html = (ROOT / "dashboard" / "template.html").read_text().replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    (OUTPUTS / "barca_signings_ledger.html").write_text(html)
    print("wrote", OUTPUTS / "barca_signings_ledger.html")
