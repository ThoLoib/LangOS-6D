# Stage 5 — Proxy-Greifstudie: Ergebnisse

Lauf 2026-09-11 auf tessa-pc (RTX 4090), Branch `lenny-stage5-prep` @ `e937a3c8`,
Protokoll unverändert aus `docs/STAGE5_PROTOCOL.md` (fixiert 2026-09-10). 10 Objekte
(Rang 1–10) × 6 Instanzen × Bedingungen {`gt_pose`, `gt`, `proxy`} = 180 Trials, keine
`fp_error`, keine fehlenden CADs, keine Anwendbarkeits-Ausnahmen (jedes eigene CAD lieferte
Kandidaten im Greifer). Reibung Finger×Objekt = 1.0 (multiplikativ, nachgemessen);
YCB-V-Massen aus der YCB-Objektliste (Calli et al. 2015), T-LESS/LM-O 0.2 kg. `random`
wurde nicht gelaufen (DECISIONS.md 2026-09-11). Quelle aller Zahlen:
`_s5_out/proxy_grasp/{trials.csv, REPORT.md}` (Kopie: `final_results/stage5/`).

## 1 · Erfolgsraten

`success@k` = Trial gelingt in ≤ 5 ausgeführten Versuchen (Kopfzahl); `success@1` = erster
Versuch; „Versuche“ = Rate je ausgeführtem Versuch. Ohne Konfidenzintervalle
(Vereinbarung 2026-09-03) — interpretiert wird nur die gepaarte Differenz, nie die absolute
Höhe (Protokoll: „Was die Studie nicht kann“).

**Gepoolt (10 Objekte, 60 Instanzen)**

| Bedingung | success@k | success@1 | Versuche | Kandidaten err. (Med.) | blocked | D_sym Med. |
|---|---|---|---|---|---|---|
| `gt_pose` (Decke Sampler+Executor) | 24/60 (40 %) | 15/60 | 24/119 (20 %) | 11.0 | 77 | — |
| `gt` (eigenes CAD + FoundationPose) | **31/60 (52 %)** | 16/60 | 31/123 (25 %) | 11.5 | 59 | 1.7 mm |
| `proxy` (Pipeline: fixierter Proxy + FP) | 19/60 (32 %) | 13/60 | 19/142 (13 %) | **6.0** | 47 | 6.2 mm |

**Je Datensatz (success@k, `gt` → `proxy`)**

| Datensatz | Objekte | `gt_pose` | `gt` | `proxy` | Δ(gt→proxy) |
|---|---|---|---|---|---|
| YCB-V | 3 | 2/18 (11 %) | 5/18 (28 %) | **8/18 (44 %)** | **+17 Pp.** |
| T-LESS | 6 | 17/36 (47 %) | 20/36 (56 %) | 8/36 (22 %) | **−33 Pp.** |
| LM-O | 1 | 5/6 (83 %) | 6/6 (100 %) | 3/6 (50 %) | −50 Pp. |
| **gepoolt** | 10 | 40 % | **52 %** | 32 % | **−20 Pp.** |

## 2 · Gepaarte Vergleiche (dieselbe Instanz unter beiden Bedingungen)

| A → B | n | A | B | Δ (B−A) | Bilanz nur A : nur B : beide : keine |
|---|---|---|---|---|---|
| `gt` → `proxy` (gepoolt) | 60 | 52 % | 32 % | **−20 Pp.** | **18 : 6 : 13 : 23** |
| `gt` → `proxy` (YCB-V) | 18 | 28 % | 44 % | +17 Pp. | 2 : 5 : 3 : 8 |
| `gt` → `proxy` (T-LESS) | 36 | 56 % | 22 % | −33 Pp. | 13 : 1 : 7 : 15 |
| `gt` → `proxy` (LM-O) | 6 | 100 % | 50 % | −50 Pp. | 3 : 0 : 3 : 0 |
| `gt_pose` → `gt` (gepoolt) | 60 | 40 % | 52 % | +12 Pp. | **0 : 7 : 24 : 29** |

**Kopfzahl der Studie: Das Proxy-CAD kostet gepoolt 20 Prozentpunkte Greif-Erfolg**
(52 % → 32 %), aber der Verlust ist **nicht gleichverteilt**: Auf T-LESS und LM-O verliert
das Proxy deutlich, auf **YCB-V gewinnt es** (+17 Pp.) — die formnahen Haushaltsproxys
(Becher→Herz-Tasse, Crackerbox→Kaffeekuchen-Box) liefern dort *mehr* ausführbare Griffe als
das eigene CAD (mug 5/6 gegen 4/6, cracker 3/6 gegen 1/6). Eine gute Formübereinstimmung
kann das Original schlagen; ein Industrieteil-Proxy mit anderer Detailgeometrie
(T-LESS↔ITODD) kann es nicht.

## 3 · Je Objekt

| Rang | Objekt | Proxy (fixiert) | `gt_pose` | `gt` | `proxy` | D_sym gt / proxy |
|---|---|---|---|---|---|---|
| 1 | ycbv 14 mug | cup-red_heart | 2/6 | 4/6 | **5/6** | 2.1 / 4.2 |
| 2 | tless 30 | GSO-Ramekin | 5/6 | 6/6 | **0/6** | 1.2 / 4.8 |
| 3 | tless 19 | itodd obj_000018 | 2/6 | 2/6 | 1/6 | 1.7 / 6.2 |
| 4 | ycbv 2 cracker box | Coffeecake-Box | 0/6 | 1/6 | **3/6** | 2.0 / 5.0 |
| 5 | ycbv 17 scissors | Diamond-Schere | 0/6 | 0/6 | 0/6 | 2.0 / 3.4 |
| 6 | tless 22 | itodd obj_000026 | 4/6 | 4/6 | 2/6 | 1.5 / 8.3 |
| 7 | lmo 9 duck | CHICKEN_RACER | 5/6 | 6/6 | 3/6 | 3.8 / 14.8 |
| 8 | tless 10 | itodd obj_000018 | 6/6 | 6/6 | 3/6 | 1.3 / 9.3 |
| 9 | tless 28 | itodd obj_000018 | 0/6 | 0/6 | 0/6 | 1.5 / 6.8 |
| 10 | tless 18 | itodd obj_000013 | 0/6 | 2/6 | 2/6 | 1.4 / 6.8 |

Drei Objekte scheitern in **allen** Bedingungen (scissors, tless 28; tless 18 nahezu):
das ist die Sampler/Executor-Decke dieser Szenen (flach auf dem Tisch bzw. verstellte
Anfahrkorridore), keine Proxy-Eigenschaft — sichtbar daran, dass auch `gt_pose` dort 0/6 ist.
`tless 30` ist der markanteste Proxy-Ausfall (gt 6/6 → proxy 0/6): der Ramekin-Proxy setzt
die Griffe an den auskragenden Rand, der beim echten Objekt (Zylinder) ins Leere greift —
D_sym 4.8 mm ist dafür blind, die Bbox-Kompatibilität (dimDev 0.14) auch.

## 4 · Wo der Verlust entsteht

- **Kandidatenseite (wie vorhergesagt):** erreichbare Posen halbieren sich im Median
  (11.5 → 6.0); `unreachable` steigt von 4 auf 10 Trials.
- **Aber auch die Ausführung:** Rate je ausgeführtem Versuch fällt von 25 % (31/123) auf
  13 % (19/142) — 12 Pp., knapp außerhalb der vorhergesagten ±10 Pp. Der Proxy kostet
  also **beides**: Kandidaten *und* Griffqualität (Fehlgriff-Klasse `grasp_failed`
  16 → 24). Der tless-30-Fall zeigt den Mechanismus: geplant wird am Proxy-Detail
  (Rand), gegriffen am echten Objekt.

## 5 · Die sechs Vorhersagen (STAGE5_PROTOCOL.md, vor dem Lauf)

| # | Vorhersage | Ergebnis |
|---|---|---|
| 1 | `gt_pose` ≥ 80 % | **Verfehlt:** 40 %. Nur LM-O (83 %) erreicht die Marke. Die Decke von Sampler + Executor in Clutter-Szenen wurde deutlich überschätzt; Ausfallbild wie vorhergesagt (Korridore/Greiferbreite: 21 grasp_failed, 11 all_blocked; cracker 0/6, scissors 0/6). |
| 2 | `gt` höchstens 10 Pp. unter `gt_pose` | **Erfüllt — aber anders als gedacht:** `gt` liegt 12 Pp. **über** `gt_pose` (Bilanz 0:7:24:29). Der Pose-Verlust ist wie erwartet klein (D_sym 1.7 mm), aber die Prämisse „gt_pose ist die Obergrenze“ kippt (→ §6). |
| 3 | `proxy` 15–30 Pp. unter `gt`; Verlust über fehlende Kandidaten, Versuchsrate ±10 Pp. | **Halb erfüllt:** Δ −20 Pp. liegt im Band ✓; Kandidaten brechen ein ✓ (reach 11.5→6.0); aber die Versuchsrate fällt um 12 Pp. (25→13 %) — der Verlust läuft *auch* über schlechtere Griffe, nicht nur über weniger. |
| 4 | T-LESS am nächsten an `gt`, YCB-V dazwischen, LM-O am weitesten | **Verfehlt (bis auf LM-O):** Reihenfolge invertiert — YCB-V **+17 Pp.** (Proxy schlägt gt), T-LESS −33 Pp., LM-O −50 Pp. ✓ als schlechtester. Die kleine dimDev der ITODD-Proxys sagt Bbox-Kompatibilität, nicht Griff-Kompatibilität: Detailgeometrie (Rippen, Ränder, Bohrungen) entscheidet. |
| 5 | `random` ≤ 15 % | **Nicht prüfbar:** Bedingung bewusst nicht gelaufen (DECISIONS.md 2026-09-11). |
| 6 | D_sym: `gt` ≈ 2 mm, `proxy` 4–12 mm | **Erfüllt:** 1.7 mm (identisch zum Stage-3-gt-Median) und 6.2 mm gepoolt; einzig LM-O-proxy (14.8 mm) liegt über dem Band. |

## 6 · Der unerwartete Befund: `gt` schlägt die eigene „Decke“

Planen auf der **FoundationPose-Pose** gewinnt gegen Planen auf der wahren Sim-Pose auf
7 Instanzen und verliert auf **keiner** (0:7:24:29) — konsistent über alle drei Datensätze,
am stärksten auf YCB-V (11 % gegen 28 %, 39 gegen 22 blockierte Anfahrten). Das ist
systematisch, nicht Rauschen, und war nicht vorhergesagt. Die Ursache ist mit diesen Daten
nicht identifizierbar; kandidierende Erklärungen: die FP-Pose stammt aus dem *echten*
Tiefenbild und kann die real aufliegende Kontaktlage besser treffen als die
BOP-GT-Annotation, aus der die Sim-Pose gesetzt wird; oder der mm-Versatz verschiebt
Anfahrten aus grenzwertig kollidierenden Korridoren. Für die Kernfrage der Studie ist das
folgenlos — `gt` und `proxy` nutzen beide FP und bleiben exakt vergleichbar; `gt_pose`
bleibt als Mechanik-Kontrolle berichtet, taugt aber nicht als Obergrenze.

## 7 · Replikation: zweiter Rechner, andere Reibung

Laptop-Teillauf 2026-09-10/11 (88+ Trials, effektive Reibung 2.4 statt 1.0; GPU-Abbruch vor
Vollendung), gleiche Instanzen, gepaart:

| Δ(gt→proxy) | tessa-pc (µ=1.0) | Laptop (µ=2.4) | Vorzeichen |
|---|---|---|---|
| gepoolt | −20.0 Pp. (n=60) | −15.9 Pp. (n=44) | ✓ gleich |
| YCB-V | **+16.7** (n=18) | **+11.1** (n=18) | ✓ gleich (Proxy gewinnt auf beiden!) |
| T-LESS | −33.3 (n=36) | −45.0 (n=20) | ✓ gleich |
| LM-O | −50.0 (n=6) | ±0.0 (n=6) | ✗ (n=6-Zelle, 1:1-Bilanz) |

Der Befund hängt **nicht an der Reibung**: Gepoolt und in beiden großen Zellen repliziert
das Vorzeichen — einschließlich des positiven YCB-V-Vorzeichens — über Rechner und einen
Faktor 2.4 in der Reibung. Nur die kleinste Zelle (LM-O, 6 Instanzen) ist instabil.

## 8 · Einordnung

1. **Für die These:** Das von OSCAR+ gefundene Proxy-CAD trägt eine Handlung in etwa
   6 von 10 Fällen, in denen auch das eigene CAD trüge (13+6 von 31 gt-Erfolgen) — und der
   Verlust konzentriert sich dort, wo der Proxy zwar die Hülle, nicht aber die
   Detailgeometrie trifft. D_sym und Bbox-Kompatibilität sind dafür notwendige, keine
   hinreichenden Prädiktoren (tless 30: 4.8 mm, dimDev 0.14, 0/6).
2. **Positivbefund:** Formnahe Proxys können das Original **übertreffen** (YCB-V +17 Pp.,
   repliziert auf dem Laptop) — mehr greifbare Fläche/Symmetrie liefert mehr ausführbare
   Kandidaten. Retrieval-Qualität und Greif-Nutzen sind auch hier nicht dasselbe
   (Echo von Stage 3c).
3. **Absolute Raten sind Eigenschaften dieses Simulators** (Sampler-Decke 40 %,
   statisches Clutter, GT-Masken) und werden nicht als Hardware-Aussage gelesen —
   Rahmung wie im Protokoll fixiert.

*Reproduktion:* `python3 grasping/experiment_proxy_grasp.py --check` → `--plan` → (Lauf)
→ `--conditions gt_pose,gt,proxy` → `--report --conditions gt_pose,gt,proxy`. Instanzen
und Proxys sind in `grasping/proxy_grasp_instances.json` eingefroren; Abweichungen der
Laufumgebung stehen in `docs/AI_LOG.md` (2026-09-11).
