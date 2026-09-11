# Objektauswahl fürs GT-vs-Proxy-Grasping — datengetrieben

## TOP 20 — Gesamtranking (2026-09-10)

Rangfolge: erst empirisch Bestätigtes und Alle-Gates-Kandidaten, dann nach
(dimDev, Pose-Median); ab Rang 11 mit wachsenden Vorbehalten. Empfehlung:
**Rang 1–10 für die Erfolgsrate**, 11–20 nur zuziehen, wenn die Fallzahl es
braucht. Details/Szenenlisten in den Abschnitten darunter.

| # | Objekt | minDim | Proxy (fixiert) | dimDev | pMed | n (Anteil) | Einordnung |
|---|---|---|---|---|---|---|---|
| 1 | ycbv 14 mug | 81¹ | cup-red_heart | 0.07 | 4.4 | 78 (52 %) | empirisch bester Fall; p95 8 mm |
| 2 | tless 30 | 51 | GSO-Ramekin | 0.14 | 5.0 | 47 (33 %) | bester Neuzugang der Gates-Analyse |
| 3 | tless 19 | 47 | itodd obj_000018 | 0.13 | 6.6 | 78 (41 %) | |
| 4 | ycbv 2 cracker box | 72 | Coffeecake-Box | 0.09 | 5.0 | 25 (11 %) | Karton→Karton |
| 5 | ycbv 17 scissors | 16 | Diamond-Schere | 0.20 | 3.4 | 16 (21 %) | Schere→Schere; bester pMed |
| 6 | tless 22 | 44 | itodd obj_000026 | 0.46² | 8.2 | 121 (63 %) | empirisch bestätigt (4/5→3/5) |
| 7 | lmo 9 duck | 77³ | CHICKEN_RACER | 0.14 | 11.7 | 52 (29 %) | 3 mm Greifer-Luft |
| 8 | tless 10 | 42 | itodd obj_000018 | 0.16 | 8.1 | 32 (22 %) | |
| 9 | tless 28 | 48 | itodd obj_000018 | 0.26 | 6.8 | 59 (30 %) | |
| 10 | tless 18 | 64 | itodd obj_000013 | 0.26 | 5.8 | 42 (29 %) | |
| 11 | tless 9 | 63 | itodd obj_000018 | 0.29 | 8.7 | 109 (44 %) | dev an der Grenze, dafür größtes n |
| 12 | tless 25 | 61 | itodd obj_000018 | 0.27 | 5.0 | 14 (15 %) | |
| 13 | ycbv 3 sugar box | 50 | Nesquik-Dose | 0.26 | 7.0 | 9 (2 %) | dritter YCB-V-Fall; Anteil klein → strikt Szenenliste |
| 14 | tless 4 | 40 | Germanium-Flasche | 0.10 | 5.5 | 26 (4 %) | Anteil klein → strikt Szenenliste |
| 15 | tless 21 | 43 | Android-Figur | 0.10 | 7.6 | 10 (5 %) | Proxy semantisch absurd, geometrisch passend |
| 16 | tless 5 | 54 | itodd obj_000018 | 0.25 | 4.9 | 17 (9 %) | |
| 17 | tless 7 | 62 | itodd obj_000013 | 0.24 | 9.9 | 42 (17 %) | |
| 18 | tless 24 | 43 | Sanitizer-Flasche | 0.26 | 8.3 | 33 (17 %) | |
| 19 | lmo 10 eggbox | 69 | MINI_ROLLER | 0.27 | 10.4 | 9 (5 %) | n=9 |
| 20 | lmo 1 ape | 76 | Mario-Figur | 0.28 | 12.9 | 9 (5 %) | n=9; Spielzeug→Spielzeug |

¹ Bbox-minDim reißt G1 formal, empirisch gegriffen (Rand/Henkel).
² dimDev über der Schwelle, empirisch funktionierend; formnähere Alternative:
Android-Figur (0.10, aber nur 6 %). ³ 77 mm gegen 80 mm Öffnung.

Szenen für Rang 16–18 (Proxy-fixiert, beste zuerst):
- tless 5: s03/im5+41+109+117+179, s02/im463, s11/im421+497
- tless 7: s02/im3+145, s12/im100+190+298+336, s06/im367, s18/im466
- tless 24: s10/im5+20+71+80+168, s08/im84+155, s19/im291

Datensatz-Mix der Top 20: 4× YCB-V, 13× T-LESS, 3× LM-O — T-LESS dominiert,
weil die ITODD-Proxys Industrieteile strukturell gut ersetzen. Wer Balance
will: Rang 1–10 enthält alle vier YCB-V-/LM-O-tauglichen Spitzenfälle.

Ausgeschlossen bleiben (Begründungen unten): ycbv 1 chef can (Greifer),
tless 20 (Sim-FP), lmo 11 glue (Bbox täuscht, pMed 64 mm), ycbv 5 mustard und
tless 16 (Mechanismus-Exhibits).

Erstellt 2026-09-09 aus den Stage-3-Läufen (`gt/combined_gt.json`,
`3b_cross/*/records.json`) + `models_info.json` + Proxy-Mesh-Abmessungen.
Gates aus dem Versuchs-Feedback: **G1** greifbar (kleinste Bbox-Abmessung
≤ 78 mm, Panda öffnet 80) · **G2** FP-tauglich (GT-Lauf-Median ≤ 5 mm) ·
**G4** formkompatibler Proxy (sortierte Bbox des häufigsten Proxys weicht in
keiner Dimension > 20 % ab — die Nesquik-Achsen- und die Halbhöhen-Falle).

## Empfohlenes Set für die Erfolgsrate

| Objekt | minDim | GT-Med | Proxy-Med / p95 | häufigster Proxy (Anteil) | dimDev | Szenen, in denen dieser Proxy gewählt wird (beste zuerst) |
|---|---|---|---|---|---|---|
| **ycbv 14 mug** *(behalten)* | 81 mm¹ | 2.2 | 4.4 / 8.2 | cup-red_heart (52 %) | 0.07 | s55: im1038/1048/1120/1136/1164/1176/1347/1520 |
| **tless 19** *(neu)* | 47 | 1.3 | 14.6 / 68.5 | itodd obj_000018 (41 %) | 0.13 | s13/im272, s10/im103+176+252+287+482, s08/im441 |
| **tless 22** *(behalten)* | 44 | 1.6 | 8.6 / 34.2 | itodd obj_000026 (63 %) | 0.46² | s14: im2/7/65/132/147/263/346, s10/im431 |
| **tless 30** *(neu)* | 51 | 1.5 | 7.5 / 127.8 | GSO Porzellan-Ramekin (33 %) | 0.14 | s19/im128+180+219+291, s01/im127+197+212, s15/im149 |
| **lmo 9 duck** *(neu)* | 77³ | 3.3 | 20.3 / 60.3 | GSO CHICKEN_RACER (29 %) | 0.14 | s02: im47/434/503/837/844/968/1087/1123 |
| **ycbv 3 sugar box** *(optional 6.)* | 50 | 2.2 | 15.4 / 30.4 | Ultra_JarroDophilus (56 %) | 0.38² | s54: im1531/1595/1597/1605/1663, s51/im546 |

¹ Bbox-minDim 81 mm reißt G1 formal — empirisch gegriffen (Rand/Henkel); behalten.
² dimDev über der 0.20-Schwelle, aber empirisch funktionierend (tless 22: 4/5→3/5)
bzw. bester verfügbarer dritter YCB-V-Kandidat.
³ 77 mm gegen 80 mm Öffnung = 3 mm Luft — Griff quer zur kleinsten Achse wählen.

## Raus — und warum

- **ycbv 1 chef can**: 102 mm > Greifer, strukturell ungreifbar (bestätigt).
  Bitter, denn **retrieval-seitig ist es das beste Objekt der ganzen Tabelle**
  (Proxy-Median 2.1 mm, dimDev 0.01, Proxy-Anteil 93 % — der Kaffeedosen-Proxy
  ist quasi ein Zwilling). Mit einem 140-mm-Greifer (z. B. Robotiq 2F-140) wäre
  es der Demo-Fall schlechthin — als Fußnote erwähnenswert.
- **tless 20**: scheitert in eurer Sim schon mit GT-CAD. Achtung, ehrlicher
  Befund: in *unserem* BOP-Setting ist sein GT-Median unauffällig (1.5 mm) —
  der FP-Ausfall ist also spezifisch für die gerenderte Sim-Szene, kein
  generelles FP-Versagen an diesem Objekt. So oder so: raus aus der Rate.

## Als Mechanismus-Exhibits behalten (nicht in die Rate)

- **ycbv 5 mustard / Nesquik** (dimDev 0.25, Achslage) und **tless 16 /
  obj_000027** (dimDev 0.55, halbe Höhe): beide zeigen exakt den Befund
  „das Proxy kostet Kandidaten, nicht Griffqualität“ — die Bbox-Abweichung
  sagt den Ausfall quantitativ voraus. Als Negativ-Paar berichten.

## Praxis-Hinweise

1. **Szenen nach Proxy wählen** (Spalte rechts): der häufigste Proxy stellt je
   nach Objekt nur 29–63 % der Instanzen. Wer Szenen zufällig zieht, misst bei
   n=5 zur Hälfte einen anderen Proxy — die Listen oben fixieren das.
2. Die Nenner-Falle aus dem Feedback gilt weiter: Objekte-gegriffen berichten,
   nicht Griffe-pro-ausgeführtem-Versuch.
3. Zur offenen Frage „Fallzahl erhöhen oder aufschreiben?": mit diesem Set
   beides in dieser Reihenfolge — erst mehr Szenen pro Objekt (die Listen geben
   je 8 her), denn 5 Versuche/Objekt tragen keine Rate; der Mechanismus-Teil
   (blockierte Kandidaten, dimDev als Prädiktor) ist schon jetzt berichtbar.

## Erweiterung: mehr Objekte über den besten häufigen Proxy (2026-09-09)

Statt nur den *häufigsten* Proxy zu bewerten, wird je Objekt der **beste unter
allen Proxys mit ≥ 8 Instanzen** gewählt — die Szenenliste fixiert ihn dann.
pMed/n/Anteil beziehen sich auf genau diesen Proxy.

**Tier A — uneingeschränkt zusätzlich empfohlen:**

| Objekt | Proxy | dimDev | pMed | n (Anteil) | Szenen |
|---|---|---|---|---|---|
| **ycbv 2 cracker box** | Carnation-Coffeecake-Box | 0.09 | 5.0 | 25 (11 %) | s50: im1658/1669/1695/1711/1733/1756/1778/1874 |
| **ycbv 17 scissors** | Diamond-Visions-Schere (Schere→Schere!) | 0.20 | 3.4 | 16 (21 %) | s51: im35/138/240/515/637/649/672/675 |
| **tless 10** | itodd obj_000018 | 0.16 | 8.1 | 32 (22 %) | s05/im317+332+436+475, s11/im405+412+486, s16/im448 |
| **tless 18** | itodd obj_000013 | 0.26 | 5.8 | 42 (29 %) | s19/im255+263+345+356+411, s03/im292+328+335 |
| **tless 28** | itodd obj_000018 | 0.26 | 6.8 | 59 (30 %) | s13/im2+7+69, s15/im12+18+77+149+164 |

**Tier B — brauchbar mit Vorbehalt** (kleiner Anteil oder dev an der Grenze):

| Objekt | Proxy | dimDev | pMed | Vorbehalt |
|---|---|---|---|---|
| tless 9 | obj_000018 | 0.29 | 8.7 | dev an der Grenze; dafür n=109 (44 %) |
| tless 25 | obj_000018 | 0.27 | 5.0 | Anteil 15 % |
| tless 4 | Germanium-GE132-Flasche | 0.10 | 5.5 | Anteil nur 4 % — strikt an die Szenenliste halten (s20/im3+58, s09/im31+70+91+161, s05/im64+70) |
| tless 21 | Android-Figur (Chrom) | 0.10 | 7.6 | Proxy semantisch absurd, geometrisch passend — fürs Greifen zählt Geometrie |
| lmo 10 eggbox | MINI_ROLLER | 0.27 | 10.4 | n=9 |
| lmo 1 ape | Mario-Figur | 0.28 | 12.9 | n=9; Spielzeug→Spielzeug |

**Trotz guter Einzelwerte NICHT aufnehmen:** lmo 11 glue (Bbox passt mit 0.05,
aber pMed 64.5 mm — dünnes Objekt, Pose instabil; die Bbox allein täuscht),
tless 20 (Sim-FP-Ausfall, s. oben), tless 1/2 (Proxy-Anteil 1–2 %).

**Nebenbefunde:** ycbv 5 mustard bleibt Mechanismus-Fall — auch der beste
häufige Proxy ist Nesquik (0.25). tless 22 hätte mit der Android-Figur (0.10)
einen noch formnäheren, aber seltenen Proxy als Alternative zu obj_000026.

Damit stehen insgesamt **bis zu 11 rate-taugliche Objekte** (5–6 aus dem
Kern-Set + 5 Tier A) plus Tier B nach Bedarf — genug, um die Fallzahl-Kritik
zu adressieren, ohne die Gates aufzuweichen.

Reproduktion: Analyse-Skripte im Chat-Verlauf (grasp_pick.py, grasp_pick2.py) —
lesen nur vorhandene Ergebnisdateien, rechnen nichts neu.
