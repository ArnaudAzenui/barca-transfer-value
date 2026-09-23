# Barça signings: were they worth it?

Every player FC Barcelona **paid a transfer fee** for from 2010/11 to 2025/26 (59 signings), judged on
what he did **at Barça** (La Liga + Champions League + Europa League, relative to players **in the same role**)
and on the **money Barça got back** when he left, with all money in **today's football euros** by default.
It also judges each **decision at the moment of signing**, and benchmarks Barça against **Real Madrid and Atlético**.

## Quick start

```bash
cd ~/projects/barca-transfer-value
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put transfermarkt-datasets.zip in data/raw/ (or it will be downloaded automatically)
python -m src.pipeline          # builds everything (3 money bases) -> outputs/
python -m src.build_dashboard   # outputs/barca_signings_ledger.html
python -m src.make_notebook     # regenerates the notebook; then open it in Jupyter
```

Data: <https://github.com/dcaribou/transfermarkt-datasets> (zip: `https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip`, snapshot to July 2026).

## Data you need to supply

The repository contains the code, the hand-curated tables and the outputs, not the raw data:

* **transfermarkt-datasets.zip** (~240 MB, too big for GitHub): download it from the link above and put it in
  `data/raw/` or the project root. The pipeline unpacks it on first run.
* **Capology tables** (`data/curated/capology_*.csv`) are git-ignored by default, because they're copied from
  capology.com. To rebuild them, save the club *Payrolls* pages and any season *Salaries* pages from capology.com
  into `data/raw/capology/`, then re-extract them. Without them, wages fall back to `data/curated/wages_reported.csv`.

## Hosting the dashboard (Vercel)

`python -m src.build_dashboard` also writes a stand-alone `site/index.html`. `vercel.json` tells Vercel to serve the
`site/` folder as-is: no build step, no install. Import this GitHub repo at https://vercel.com/new and every push to
`main` redeploys automatically. Pages send `noindex` so search engines skip them. Vercel production URLs are public
unless you turn on Deployment Protection, so check what the dashboard shows (e.g. wage estimates) before sharing the link.

## Method

| Step | File | What it does |
|---|---|---|
| Money bases | `src/inflation.py` | **Football index** (headline): mean value of the top 200 top-5-league players (2025/26 = 1.0; €1 in 2010 ≈ €3.16 today). **CPI**: euro-area HICP (`data/curated/euro_hicp.csv`, approximate). **Actual €**: no adjustment. Everything is run in all three. |
| Signings | `src/signings.py` | Paid moves into Barça (club id 131), exits, loans, sale fees. Early deals missing from the dataset come from `data/curated/missing_signings.csv`. |
| League performance | `src/performance.py` | For every league in the data: goals/90, assists/90, minutes share, team points on/off, goals conceded while playing, each z-scored within league, role group and season (club-seasons with <50% line-up data are flagged), combined with role weights (`ROLE_WEIGHTS` in `src/config.py`). |
| European performance | `src/europe.py` | Champions League + Europa League goals/90, assists/90, minutes share, z-scored against that competition's players in the same role; Europa League discounted by 0.6. |
| Trophies | `src/trophies.py` | Winners of La Liga (standings), Champions League, Europa League, Copa del Rey, Supercopa, UEFA Super Cup and Club World Cup (finals). Trophy points = Σ weight × involvement (minutes + goal/assist share), with weights CL 3 > La Liga 2 > Copa/EL 1 > CWC 0.75 > Supercopas 0.5 (`TROPHY_WEIGHTS` in `src/config.py`). |
| Market model | `src/market.py` | OLS on ~7,600 La Liga player-seasons: `log(value) ~ role × league metrics + role × European goals/assists + European minutes + CL/EL stage + trophy points + age + age² + team strength` (R² ≈ 0.67; each trophy point ≈ +16% value). Gives each Barça season a performance-implied value (PIV). |
| Decision at signing | `src/value.py` (`add_pre_signing`) | A second market model fitted on ~92,000 player-seasons in 14 leagues (league fixed effects) values the player's last two league seasons *before* joining; cross-checked against his Transfermarkt value that day. Overpaid/bargain only when both views agree. Combined with the outcome into: smart buy paid off / sound buy went wrong / overpaid as warned / ... |
| Rivals | `src/pipeline.py` (`evaluate_club`, `club_summary`, `premium_tests`) | Same method for Real Madrid (418) and Atlético (13); missing early deals in `data/curated/missing_signings_rivals.csv`; club-level totals, balance per €1, "Barça premium" regression. |
| Wages | `src/wages.py` | Capology actual player salaries where available (Barça 2017/18, 2018/19, 2020/21: `capology_salaries_barca_history.csv`; Real Madrid and Atlético 2026/27: `capology_salaries_2026_27.csv`). Otherwise each club-season's Capology payroll (`capology_payrolls.csv`, 2013/14 on) is split by market value and age (curve fitted on 117 matched salaries), scaled by the player's own actual/estimated ratio when known. Leave-one-season-out check: log correlation 0.74–0.85, median error 31–41% per player. Raw pages are in `data/raw/capology/`. |
| Costs | `src/costs.py` | Fee, wages (above; old hand estimates kept in `wages_reported.csv` for comparison), sale and loan fees, current value for players still at the club. |
| Verdicts | `src/value.py`, `src/pipeline.py` | **Deal balance (headline)**: performance value (each season = its fair fee, priced at age 27, ÷ 5) + money back − fee, with an 80% range. **Fee test**: fee vs fair fee from performance alone. **Total-cost test**: fee + wages − money back vs value delivered, leave-one-out across Barça's signings. |

## Outputs

* `outputs/barca_signings_verdicts.csv`: one row per signing with every number and verdict (football-€ basis)
* `outputs/verdicts_by_money_basis.csv`: fee and verdict in football €, CPI € and actual €
* `outputs/signings_barca_real_atletico.csv`: every paid signing by the three clubs with decision and outcome
* `outputs/barca_signings_ledger.html`: interactive dashboard (views: deal balance / fee test / total cost; money switch)
* `outputs/inflation_index.csv`, `outputs/model_summary.json`
* `notebooks/barca_value_for_money.ipynb`: full walkthrough, robustness checks and limitations

## Things to try next

* Replace the rough wage table with better-sourced figures.
* Add sell-on clauses (a % of later sales Barça received) as a small curated table.
* Starts vs substitute appearances, decisive goals, performance against strong teams (all derivable from the data).
* Swap OLS for gradient boosting in `src/market.py` and compare the verdicts.
