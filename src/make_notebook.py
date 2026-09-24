"""Generate notebooks/barca_value_for_money.ipynb (python -m src.make_notebook)."""
import nbformat as nbf
from .config import ROOT

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Were Barça's signings worth it? (2010/11 – 2025/26)

**Question:** for every player FC Barcelona paid a transfer fee for since 2010, was the fee justified by what he did *at Barça*, given his role?

**Approach, in one picture**

| Step | What | Why |
|---|---|---|
| 1 | **Football-inflation index** | €40m in 2010 ≠ €40m in 2025. All money is converted to 2025/26 football euros. |
| 2 | **Role-aware season scores** | Each La Liga season is scored against same-position peers (goals and assists matter for forwards, availability and goals conceded for defenders). |
| 3 | **Market model** (linear regression, ~7,600 La Liga player-seasons) | Learns what the market pays for a season of output → the *performance-implied value* (PIV) of each Barça season. |
| 4 | **Deal balance (headline)** | Performance value + money back (sale, loan fees, current value) − fee, with an 80% range. |
| 5 | **Fee test** | Fee vs fair fee (PIV × typical fee premium), from performance alone. |
| 6 | **Total-cost test** | Fee + wages − money back vs total value delivered, benchmarked across Barça's own signings (leave-one-out). |

Performance covers **La Liga plus the Champions League and Europa League**, with trophy points for competitions won. Two further questions: **was each fee reasonable at the moment of signing** (section 7c), and **how does Barça's record compare with Real Madrid's and Atlético's** (section 7d). Everything is run in three money bases (football index, consumer prices, actual euros) so you can see what the inflation adjustment does.

Data: [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (snapshot to July 2026) + small hand-curated tables in `data/curated/`.
""")
code("""import sys, json
sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from src.pipeline import run
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e1e0d9', 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
BLUE, RED, GRAY, INK = '#2a78d6', '#e34948', '#898781', '#0b0b0b'
def short(name):
    last = name.split()[-1]
    return name if last in ('Suárez', 'Vidal', 'Jong', 'Torres') else last

d, summary, market_model, market_rows, ix, scored, results = run(verbose=False, use_cache=True)
from src.pipeline import MODELS
market_all = MODELS['football']
allclubs = pd.read_pickle('../data/processed/results_all_clubs.pkl')
summary""")

md("""## 1. Football inflation

Three ways to put fees from different years on the same scale:

* **Football index (headline):** the average Transfermarkt value of the 200 most valuable top-5-league players each season, with 2025/26 = 1.0. It roughly tripled between 2010 and 2025.
* **Consumer prices:** euro-area HICP (`data/curated/euro_hicp.csv`, approximate). It rose only about 40% over the same period.
* **Actual euros:** no adjustment.

The dashed line is a fee-based index (mean of the top-100 fees). It tracks the football index from 2014, but the open dataset is missing many pre-2013 transfers, so it is biased low in the early years.""")
code("""fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(ix.season, ix['football_index'], color=BLUE, lw=2, marker='o', ms=4, label='Football index (headline)')
ax.plot(ix.season, ix['cpi_index'], color='#eb6834', lw=2, marker='o', ms=4, label='Consumer prices (HICP)')
ax.plot(ix.season, ix['fee_index'], color=GRAY, lw=1.5, ls='--', label='Top-100 fee index (check)')
ax.axhline(1, color='#c3c2b7', lw=1)
ax.set_ylabel('Price level (2025/26 = 1.0)'); ax.set_title('Football price inflation, top-5 leagues', loc='left')
ax.legend(frameon=False); plt.show()
i = ix.set_index('season')
for y in (2010, 2013, 2017):
    print(f"€1 in {y}: {1/i.loc[y,'football_index']:.2f} football-€ today, {1/i.loc[y,'cpi_index']:.2f} CPI-€ today")""")

md("""## 2. The signings

59 paid signings. 45 come straight from the dataset; 15 (mostly 2010–2017 players whose later careers left the tracked leagues) come from `data/curated/missing_signings.csv`, including one multi-stage deal (Emerson Royal) merged into a single row. League appearance data starts in **2012/13**. For Villa, Adriano, Mascherano, Afellay, Alexis and Fàbregas, the 2010/11 and 2011/12 seasons use hand-checked StatMuse stats (`data/curated/season_stats_statmuse.csv`), scored against 2012/13 players in the same role, with La Liga and Champions League titles credited from Barça's team totals (`team_seasons_statmuse.csv`). Every signing's La Liga and Champions League seasons were checked against StatMuse: three 2020/21 seasons missing from the dataset (Alba, ter Stegen, de Jong) were added and three goal counts corrected (`stat_corrections.csv`); see `src/patches.py`.""")
code("""d[['player','group','signing_season','from_club','fee_eur','fee_adj','seasons','seasons_covered','exit_to','ongoing','source']] \\
  .assign(fee_eur=lambda x: x.fee_eur/1e6).round(1).sort_values('signing_season')""")

md("""## 3. Role-aware season scores

For each La Liga season, each metric is turned into a z-score **within the same role group and season**, using peers with 450+ minutes:

* `g90`, `a90`: goals and assists per 90
* `mins`: share of the team's league minutes (availability plus the coach's trust)
* `onoff`: team points per game with him (60+ min) minus without
* `cs`: goals the team concedes per game when he plays 60+ (fewer is better)

Per-90 z-scores shrink toward 0 when a player has few minutes. Weights by role are set in `src/config.py` → `ROLE_WEIGHTS`. Change them and re-run.""")
code("""from src.config import ROLE_WEIGHTS
pd.DataFrame(ROLE_WEIGHTS).T""")
code("""b = scored[scored.club_id == 131]
b[b.season == 2015].sort_values('score', ascending=False)[['name','group','minutes','goals','assists','z_g90','z_a90','z_mins','z_onoff','z_cs','score']].round(2).head(15)""")

md("""## 4. The market model: what is a season of output worth?

A **linear regression** on every La Liga player-season 2012/13–2025/26:

`log(end-of-season market value, 2025/26 €) ~ role + role × (g90, a90, mins, onoff, cs z-scores) + age + age² + team points per game`
`    + role × (European g90, a90 z-scores) + European minutes z + in Champions League + in Europa League`

This is your original idea (predict transfer value from stats) at league scale. Using logs means coefficients act as percentages: `+0.5` ≈ ×1.65 value.""")
code("""print(f"n = {int(market_model.nobs)},  R² = {market_model.rsquared:.3f},  residual SD (log) = {np.sqrt(market_model.scale):.3f}")
market_model.params.round(3).to_frame('coef')""")
code("""fig, ax = plt.subplots(figsize=(5.5, 5.5))
pred = np.exp(market_model.fittedvalues)/1e6; act = market_rows['mv_adj']/1e6
ax.scatter(pred, act, s=6, alpha=.25, color=GRAY, label='All La Liga player-seasons')
bb = market_rows.club_id == 131
ax.scatter(pred[bb], act[bb], s=14, color=BLUE, label='Barça player-seasons')
ax.set_xscale('log'); ax.set_yscale('log'); lim = [0.05, 400]
ax.plot(lim, lim, color=INK, lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Predicted value (€m, 2025/26)'); ax.set_ylabel('Actual Transfermarkt value (€m, 2025/26)')
ax.set_title('Market model fit', loc='left'); ax.legend(frameon=False, loc='upper left'); plt.show()""")

md("""## 4b. European matches

Champions League and Europa League appearances are scored like the league: goals/90, assists/90 and share of team minutes, z-scored against **everyone in that competition in the same role** (peers need 270+ minutes). Europa League output is discounted by 0.6. Goalkeepers are judged on minutes only.

What the market model learns (football-€ basis):""")
code("""eu = [k for k in market_model.params.index if k.startswith('eu_') or k in ('in_cl', 'in_el')]
pd.DataFrame({'coef': market_model.params[eu], 'x value': np.exp(market_model.params[eu]), 'p': market_model.pvalues[eu]}).round(3)""")
md("""How to read this: being at a **Champions League club** multiplies a player's value by about 2.6 compared with a non-European club at the same league output (the Europa League by about 1.7). Holding that constant, **individual European goals and assists add little**; European minutes add about 10% per standard deviation. For Barça signings, this mostly means every season is priced at Champions League level, while European output separates players only slightly. Adding Europe raised the model's R² from 0.64 to 0.67.""")
code("""b = scored[scored.club_id == 131]
b[b.season == 2014].sort_values('eu_minutes', ascending=False)[['name','group','minutes','goals','eu_comps','eu_minutes','eu_goals','eu_assists','score','eu_score']].round(2).head(12)""")

md("""## 4c. Trophy points

Goals, assists and minutes count extra in any competition the club **won**. For each won competition:

`involvement = 0.6 × share of team minutes + 0.4 × share of team goals+assists` (each capped at 1)
`trophy points = Σ weight × involvement`, with weights **Champions League 3, La Liga 2, Copa del Rey / Europa League 1, Club World Cup 0.75, Supercopa / UEFA Super Cup 0.5**

Winners come from the match data: most points for La Liga, and the final for cups (scores include shoot-outs; two-legged Supercopa finals use aggregate, then away goals). The market model then learns what one point is worth in euros.""")
code("""from src.trophies import winners
from src.data import load
w = winners(); names = load('clubs').set_index('club_id')['name']
w[w.club_id == 131].assign(competition=lambda x: x.competition_id).groupby('season').competition.apply(', '.join).to_frame('Barça trophies')""")
code("""b = market_model.params['trophy_pts']; se = market_model.bse['trophy_pts']
print(f"Each trophy point multiplies a season's value by {np.exp(b):.3f} (±{1.96*se:.3f}, p={market_model.pvalues['trophy_pts']:.1e})")
print(f"A leading role in a treble (~3.5 points): ×{np.exp(3.5*b):.2f};  a regular in a title-only season (~1.3 points): ×{np.exp(1.3*b):.2f}")
scored[scored.club_id == 131].sort_values('trophy_pts', ascending=False)[['season','name','trophy_pts','trophies']].head(10)""")

md(f"""## 5. Test 1: the fee test

**Fair fee = PIV × market premium**, where PIV is the model's value for each of the player's Barça seasons (geometric mean) and the premium is the median fee ÷ Transfermarkt value for €10m+ transfers into top-5-league clubs (≈1.25).

* **Overpaid**: fee above the 80% prediction interval of the fair fee
* **Bargain**: fee below it
* **Fair price**: inside it

The interval is wide (the model's residual SD is about 0.82 in logs), so a verdict of overpaid or bargain is a strong statement.""")
code("""t = d.dropna(subset=['fair_fee_mean_eur']).copy()
for c in ['fair_fee_mean_eur','fair_fee_lo_eur','fair_fee_hi_eur']: t[c] = t[c]/1e6
col = t.market_verdict.map({'Overpaid': RED, 'Bargain': BLUE, 'Fair price': GRAY})
fig, ax = plt.subplots(figsize=(8, 7))
xs = np.logspace(0, 2.6, 50); k = np.exp(1.2816*np.sqrt(market_model.scale))
ax.fill_between(xs, xs/k, xs*k, color='#f0efec', label='80% interval: fair')
ax.plot(xs, xs, color=INK, lw=1, label='Fee = fair fee')
ax.scatter(t.fair_fee_mean_eur, t.fee_adj, c=col, s=40, edgecolor='white', lw=1, zorder=3)
for _, r in t.iterrows():
    if r.market_verdict != 'Fair price' or r.fee_adj > 90:
        ax.annotate(short(r.player), (r.fair_fee_mean_eur, r.fee_adj), fontsize=8, xytext=(4, 2), textcoords='offset points', color='#52514e')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(3, 400); ax.set_ylim(1, 400)
ax.set_xlabel('Fair fee from Barça performance (€m, 2025/26)'); ax.set_ylabel('Fee paid (€m, 2025/26)')
ax.set_title('Fee paid vs fee justified by performance', loc='left')
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color=c, label=l) for l, c in [('Overpaid', RED), ('Fair price', GRAY), ('Bargain', BLUE)]]
ax.legend(handles=h + ax.get_legend_handles_labels()[0], frameon=False, loc='upper left'); plt.show()""")
code("""t[['player','signing_season','fee_adj','fair_fee_mean_eur','fair_fee_lo_eur','fair_fee_hi_eur','fee_to_fair','market_verdict']] \\
  .sort_values('fee_to_fair', ascending=False).round(2)""")

md("""## 5b. Deal balance (the headline)

**Balance = performance value + money back − fee**

* **Performance value:** each Barça season is credited with 1/5 of that season's fair fee, priced as if the player were 27. Age is left out here because a season's output is worth the same to Barça whatever his birth year. Age still matters through the money back: young players keep resale value, old ones don't.
* **Money back:** sale fee + loan fees received + current market value if he's still at the club.
* **Zero point:** paying exactly the fair fee, playing at that level for five seasons and leaving for free gives a balance of 0.
* **Verdict:** profit or loss only when the whole 80% range sits on one side of zero.""")
code("""t = d.sort_values('balance_adj')
col = t.balance_verdict.map({'Loss': RED, 'Profit': BLUE, 'Break-even': GRAY})
fig, ax = plt.subplots(figsize=(8, 13))
y = np.arange(len(t))
ax.hlines(y, t.balance_lo.clip(-250, 400), t.balance_hi.clip(-250, 400), color=col, alpha=.45, lw=2)
ax.scatter(t.balance_adj, y, c=col, s=30, zorder=3, edgecolor='white')
ax.axvline(0, color=INK, lw=1); ax.set_yticks(y); ax.set_yticklabels([short(n) for n in t.player], fontsize=8)
ax.set_xlim(-250, 400); ax.set_xlabel('Deal balance (€m, 2025/26 football money)')
ax.set_title('Performance value + money back − fee, with 80% range', loc='left'); ax.grid(axis='y', visible=False); plt.show()""")
code("""d[['player','signing_season','fee_adj','perf_value_adj','recovered_adj','balance_adj','balance_lo','balance_hi','balance_verdict','verdict']] \\
  .sort_values('balance_adj', ascending=False).round(1)""")
md("""**How sensitive is this to the 5-season assumption?** Shorter payback periods (3 seasons) credit each season more; longer ones (7) credit it less.""")
code("""from src.value import add_deal_balance
from src.market import market_premium
prem = market_premium(); rows = []
for n in (3, 5, 7):
    x = add_deal_balance(d, market_model, prem, use_seasons=n)
    rows.append(x.set_index('player')['balance_verdict'].rename(f'{n} seasons'))
sens = pd.concat(rows, axis=1)
print({c: sens[c].value_counts().to_dict() for c in sens})
sens[(sens.nunique(axis=1) > 1)]""")

md("""## 6. Test 2: total cost of ownership, with wages

**Net total cost** = fee + wages over the Barça seasons − (sale fee + loan fees received + current market value if he's still at the club), all in 2025/26 €.
**Value delivered** = sum of the performance-implied values across his Barça seasons (in €m-seasons).

Regression across Barça's own signings: `net total cost ~ value delivered`. Each player is judged by a fit **without him** (leave-one-out), with an 80% prediction interval.

**Wages** come from Capology (`src/wages.py`). Actual player salaries are used where Capology lists them: Barça 2017/18, 2018/19 and 2020/21, and Real Madrid and Atlético 2026/27. Everywhere else, each club-season's gross fixed payroll is split across the squad in proportion to what a player of that market value and age typically earns, scaled by the player's own actual/estimated ratio when one is known. Seasons before 2013/14 reuse the 2013/14 payroll. This test is a **supporting check**, not the headline.""")
code("""fit2 = smf.ols('net_total_cost_adj ~ value_delivered_m', data=d).fit()
print(fit2.summary().tables[1]); print('R² =', round(fit2.rsquared, 3))
col = d.net_total_cost_adj_verdict.map({'Overpaid': RED, 'Bargain': BLUE, 'Fair price': GRAY})
fig, ax = plt.subplots(figsize=(8, 5.5))
xs = np.linspace(0, d.value_delivered_m.max()*1.05, 50)
pr = fit2.get_prediction(pd.DataFrame({'value_delivered_m': xs})).summary_frame(alpha=.2)
ax.fill_between(xs, pr.obs_ci_lower, pr.obs_ci_upper, color='#f0efec'); ax.plot(xs, pr['mean'], color=INK, lw=1)
ax.scatter(d.value_delivered_m, d.net_total_cost_adj, c=col, s=40, edgecolor='white', lw=1, zorder=3)
for _, r in d.iterrows():
    if r.net_total_cost_adj_verdict != 'Fair price' or r.value_delivered_m > 300:
        ax.annotate(short(r.player), (r.value_delivered_m, r.net_total_cost_adj), fontsize=8, xytext=(4, 2), textcoords='offset points', color='#52514e')
ax.axhline(0, color='#c3c2b7', lw=1)
ax.set_xlabel('Value delivered (sum of season PIVs, €m, 2025/26)'); ax.set_ylabel('Net total cost incl. wages (€m, 2025/26)')
ax.set_title("Total cost vs value delivered, Barça's own benchmark", loc='left'); plt.show()""")

md("""## 7. Headline verdicts

| Label | Meaning |
|---|---|
| Good deal / Bad deal | The deal balance's 80% range is entirely above / below zero |
| Good deal (resale) | Positive only because of the sale or current value; the football alone didn't cover the fee |
| … (all tests agree) | The fee test and the total-cost test point the same way |
| Leaning good / bad deal | The balance is inconclusive, but both other tests agree |
| Fair deal | Everything else |""")
code("""cols = ['player','signing_season','fee_adj','balance_adj','balance_verdict','market_verdict','net_total_cost_adj_verdict','verdict','partial_data']
d[cols].sort_values('balance_adj', ascending=False).round(1)""")
code("""d.verdict.value_counts()""")

md("""## 7b. What the inflation adjustment does

The whole analysis, including the market model, is re-run in each money basis. Two things stand out:

1. **The football index fits best.** The total-cost model explains far more of the variation in football euros than in actual euros or CPI euros. Without it, early deals look cheap and later ones expensive for reasons that have nothing to do with the players.
2. **Actual euros flatter early deals.** A 2013 fee is compared with seasons valued at 2015–2017 prices, so it looks like a bargain.""")
code("""pd.DataFrame({b: {'market R²': s['market_model']['r2'], 'total-cost R²': s['club_model']['r2']} for b, s in
              json.load(open('../outputs/model_summary.json'))['by_basis'].items()}).T""")
code("""vb = pd.read_csv('../outputs/verdicts_by_money_basis.csv')
print({b: vb[f'verdict_{b}'].value_counts().to_dict() for b in ('football','cpi','nominal')})
vb[(vb.verdict_football != vb.verdict_nominal) | (vb.verdict_football != vb.verdict_cpi)].round(1)""")

md("""## 7c. Decision vs outcome: was the fee reasonable *when he signed*?

Every verdict above judges how a deal *turned out*. This section judges the decision with what was knowable at the time:

* **Stats view:** a second market model, fitted on ~92,000 player-seasons across 14 leagues (with league fixed effects, so a +1 z-score in the Eredivisie is worth less than in La Liga), is applied to the player's **last two league seasons before joining**. Fair fee = that value × 1.25, with an 80% interval.
* **Market view:** his Transfermarkt valuation on the signing date × 1.25. The "normal" range is the 10th–90th percentile of what clubs actually pay over valuations.
* **Decision:** overpaid or bargain only when **both views agree**. Stats miss potential (young prospects look expensive) and valuations can run hot, so disagreements count as fair. If only one view exists (e.g. players from Brazil, whose league data starts in 2024), that view decides.""")
code("""print(f"All-league model: n = {int(market_all.nobs):,}, R² = {market_all.rsquared:.3f}")
cols = ['player','signing_season','fee_adj','pre_fair_eur','pre_ratio','pre_verdict','tm_ratio','tm_verdict','decision','decision_note','verdict','decision_outcome']
x = d[cols].copy(); x['pre_fair_eur'] = x.pre_fair_eur/1e6
x.sort_values('pre_ratio', ascending=False).round(2)""")
code("""pd.crosstab(d.decision, d.verdict.str.replace(r' \\(.*\\)', '', regex=True))""")
code("""col = d.decision.map({'Overpaid': RED, 'Bargain': BLUE}).fillna(GRAY)
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.scatter(d.decision_ratio, d.balance_adj.clip(-200, 350), c=col, s=40, edgecolor='white', zorder=3)
ax.axvline(1, color=INK, lw=1); ax.axhline(0, color=INK, lw=1); ax.set_xscale('log')
for _, r in d.iterrows():
    if abs(r.balance_adj) > 60 or r.decision != 'Fair price' or r.decision_ratio > 4:
        ax.annotate(short(r.player), (r.decision_ratio, min(max(r.balance_adj, -200), 350)), fontsize=8, xytext=(4, 2), textcoords='offset points', color='#52514e')
ax.set_xlabel('Fee ÷ fair fee at signing (log scale)'); ax.set_ylabel('Deal balance (€m)')
ax.set_title('Decision (at signing) vs outcome', loc='left'); plt.show()
d.decision_outcome.value_counts()""")
md("""**Reading it:** only Dembélé was clearly overpaid on both views *and* turned out badly. Most of Barça's bad deals (Coutinho, Griezmann, Pjanić, Arda Turan, Vermaelen, Neto) were **defensible at the time**: the fee sat inside the normal range for the player's record and valuation. They went wrong afterwards through injuries, fit, or decline. De Jong, Trincão, Dest and André Gomes look expensive against their stats but not against their market values; stats undervalue young prospects.""")

md("""## 7d. Barça vs Real Madrid vs Atlético

The same pipeline, run on every paid signing by the two Madrid clubs (football-€ basis; wages imputed). Deals missing from the dataset were filled in by hand for them too (`data/curated/missing_signings_rivals.csv`: Bale, Hazard, Özil, Mandžukić, Jackson Martínez, Diego Costa and others). Atlético's 2014/15 line-ups are missing from the dataset, so two spells from that season have no performance data and are excluded.""")
code("""S = json.load(open('../outputs/model_summary.json'))['headline']
pd.DataFrame(S['clubs']).round(3)""")
code("""pd.DataFrame(S['clubs_2013']).round(3)   # 2013/14 on: the most complete data""")
code("""fig, ax = plt.subplots(figsize=(9, 4))
clubs = ['Barcelona', 'Real Madrid', 'Atlético Madrid']
for i, c in enumerate(clubs):
    x = allclubs[(allclubs.club == c) & allclubs.balance_adj.notna()]
    jit = (np.arange(len(x)) * 37 % 17 - 8) / 40
    ax.scatter(x.balance_adj.clip(-150, 350), i + jit, s=18, alpha=.6, color=[BLUE, GRAY, RED][i])
    ax.plot([x.balance_adj.median()] * 2, [i - .35, i + .35], color=INK, lw=3)
ax.set_yticks(range(3)); ax.set_yticklabels(clubs); ax.axvline(0, color=INK, lw=1)
ax.set_xlabel('Deal balance per signing (€m, football €)'); ax.set_title('Every paid signing, three clubs (black = median)', loc='left'); plt.show()""")
code("""for c in clubs:
    x = allclubs[(allclubs.club == c) & allclubs.balance_adj.notna()].sort_values('balance_adj')
    print(c, '| worst:', ', '.join(f"{p} ({b:+.0f})" for p, b in zip(x.player.head(4), x.balance_adj.head(4))),
          '| best:', ', '.join(f"{p} ({b:+.0f})" for p, b in zip(x.player.tail(4)[::-1], x.balance_adj.tail(4)[::-1])))""")
md("""### Is there a "Barça premium"?

For €10m+ deals by the three clubs: regress log(fee ÷ fair fee) on club, signing year (and age, for the stats view), with Atlético as the reference.""")
code("""pd.DataFrame({k: {c: v[c] for c in ('Barcelona', 'Real Madrid')} for k, v in S['premium_tests'].items()})""")
md("""Against players' **stats before joining**, Barça paid about **+32%** more than Atlético for a comparable record (p ≈ 0.08). Real Madrid paid no premium. Against **Transfermarkt valuations** the gap is small and not significant (+8%). Barça's fees look normal next to the market's valuations, but high next to the players' actual output, which fits a club that pays for reputation and potential.""")

md("""### Wages: Capology payroll shares vs my earlier estimates""")
code("""from src.wages import fit_wage_curve, validate
f = fit_wage_curve(scored)
print(f"Wage curve (n={int(f.nobs)} Capology salaries): within a squad, wage ∝ value^{f.params['lmv']:.2f} × e^({f.params['age']:.3f}·age)")
validate(scored)   # leave-one-season-out: estimate a Barça season's salaries without its Capology figures""")
md("""The payroll split gets the *ranking* of salaries broadly right (log correlation 0.74 to 0.85), but individual estimates are off by 30 to 40% on a typical player. That's why actual Capology figures replace the estimate wherever they exist (Barça 2017/18, 2018/19, 2020/21), and a player's other seasons are scaled by his own actual/estimated ratio.""")
code("""x = d[['player','seasons','wages_old_adj','wages_adj','wage_source']].copy()
print('correlation with the old hand estimates:', round(x[['wages_old_adj','wages_adj']].corr().iloc[0,1], 2))
x.assign(diff=x.wages_adj - x.wages_old_adj).sort_values('diff').round(1)""")

md("""## 8. Robustness: how much do the wage figures matter?

Scale every wage by 0.5× and 1.5× and re-run Test 2.""")
code("""from src.value import club_verdicts
rows = []
for k in [0.5, 1.0, 1.5]:
    x = d.copy(); x['net_total_cost_adj'] = x.net_fee_cost_adj + k*x.wages_adj
    x = club_verdicts(x, 'net_total_cost_adj')
    rows.append(x.set_index('player')['net_total_cost_adj_verdict'].rename(f'wages ×{k}'))
sens = pd.concat(rows, axis=1)
sens[(sens != 'Fair price').any(axis=1)]""")

md("""## 9. Limitations (read before quoting results)

* **Box-score stats miss a lot.** Build-up play, pressing and progressive passing aren't in the data, so deep midfielders (Frenkie de Jong, Pjanić) and ball-playing defenders are probably under-rated.
* **2010–12 performance is missing** (league and Europe); those seasons are credited at each player's covered average.
* **Copa del Rey and Supercopa count only through trophy points.** Cup minutes are mostly rotation and hard to benchmark as performance, but winning them earns credit.
* **Trophy weights (3 / 2 / 1 / 0.75 / 0.5) are a preference**, not estimated. The market model estimates only the euro value of one point.
* **Sell-on clauses** (a % of a player's next transfer) aren't in the data, so any extra money Barça received that way isn't counted.
* **The 5-season payback** in the deal balance is an assumption; section 5b shows how the verdicts shift with 3 or 7.
* **Decision at signing** uses at most the last two league seasons in the 14 covered leagues (Brazilian and most non-European leagues only from 2024), so players from Brazil are judged on market value alone.
* **Rivals** have imputed wages and hand-filled early deals. Atlético's 2014/15 line-ups are missing from the dataset, and the dataset lacks transfer histories for many players who retired before ~2022, which is why hand-filled tables were needed for all three clubs.
* **Positions are each player's current Transfermarkt position**, not necessarily his role at Barça.
* **Wages are partly estimated**: actual Capology salaries cover 3 Barça seasons; other seasons split the real payroll totals by value and age (typical error ±30–40% per player, see section 6). Capology's own figures are estimates too. Treat Test 2 as supporting evidence.
* **Ongoing players** (Pedri, Raphinha, Koundé, Olmo, …) are credited with their current market value as money recoverable. Their verdicts can still change.
* **Injuries count against a player.** Minutes share is part of the score, so Umtiti's or Dembélé's injury years lower their value. That's intentional: an injured signing doesn't justify his fee.
* The **market premium** (≈1.25) and the **80% interval** are choices; change them in `src/value.py` and `src/market.py`.
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "barca_value_for_money.ipynb"
nbf.write(nb, out)
print("wrote", out)
