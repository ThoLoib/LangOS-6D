# Stage 1 — SHREC'18 Retrieval: vollständige Ergebnisse

*Alle 38 Arme des Stage-1-Grids, nach Experimentliste geordnet, mit Signifikanz und
Kategorien-Analyse. Ergebnisordner: `object_retrieval/results_shrec18_v2_stage1_42v_k5`
(auch auf Drive). Lauf abgeschlossen 2026-08-27. n = 2101 Queries / 3308 CADs.*

Begleitdokumente: `EVALUATION_STORY_AND_PLAN.md` (Konzept), `PIPELINE_IMPLEMENTATION.md`
(Methode), `EXPERIMENTS_IMPLEMENTATION.md` (Code), `CONFIG_COMPARISON.md` (Konfig-Vergleich).

---

## 0. Konfiguration (BASE)

| Komponente | Wert |
|---|---|
| **S_text** | CLIP ViT-B/32, Query-Bild vs. 42 Per-View-Beschreibungen, **max** über Views |
| **S_view** | DINOv2-base, **mean**-Patch-Pooling, **42 Views**, top-k-softmax **k=5**, τ=0.5 |
| **S_shape** | ULIP-2 coloured (1280-d, 10k Punkte), **pc-Modus**, Partial-Views, **alle 42**, top-k-softmax **k=5** |
| **Fusion** | Weighted Sum, Min–Max-Norm, **w = (0.3, 0.4, 0.3)**, volle Datenbank |
| **Geometrie** | aus (Geometrie-Arme: dGeDi → RANSAC, Shortlist **K = 50**) |
| **Determinismus** | `PYTHONHASHSEED=0` |

---

## 0.1 Die Metriken — was genau gemessen wird

### Relevanzstufen (Grundlage aller Metriken)
SHREC'18 bewertet **abgestuft**, nicht binär:

| Stufe | Bedeutung |
|---|---|
| **2** | Treffer hat dieselbe **Kategorie *und* Subkategorie** wie die Query |
| **1** | nur dieselbe **Kategorie** |
| **0** | irrelevant |

20 Kategorien; 2683 der 3308 CADs tragen zusätzlich eine Subkategorie. **Wichtig: „richtig"
heißt Klassenzugehörigkeit, nicht Instanz.** Ein Treffer gilt als korrekt, wenn er *irgendein*
CAD derselben (Sub-)Kategorie ist — nicht, wenn er exakt dasselbe Objekt ist.

### Tabelle A — die offiziellen SHREC'18-Metriken
Berechnet mit dem **unveränderten offiziellen `metrics.py`**, damit die Zahlen
leaderboard-vergleichbar sind. Alle Metriken laufen über die **Top-f** der Rangliste, wobei
**f = Größe der Query-Kategorie** ist (`x = rel[:f]`).

| Metrik | Was sie misst |
|---|---|
| **nDCG** | Qualität der ganzen Rangliste mit **abgestuften** Gewinnen (Subkategorie zählt doppelt), diskontiert nach Rang. Die Headline-Metrik. |
| **precision** | Anteil relevanter Treffer in den Top-f |
| **recall** | relevante Treffer in Top-f, geteilt durch f |
| **F1** | harmonisches Mittel aus beiden |
| **AP** | Average Precision — belohnt, wenn Relevantes *früh* kommt |
| **NNT1 / NNT2** | First / Second Tier — Anteil der Relevanten in den Top-f bzw. Top-2f |

> ⚠️ **Warum precision = recall = F1 = NNT1 = NNT2 identisch sind.** Weil die Liste **vor** der
> Metrikberechnung auf Top-f gekürzt wird und f zugleich die Zahl der Relevanten ist, gilt
> |zurückgegeben| = |relevant| → precision = recall, und damit auch F1. NNT2 kann über eine nur
> f Elemente lange Liste nicht bis 2f schauen und fällt auf NNT1 zurück. Das ist **kein Fehler
> unsererseits**, sondern das Verhalten des offiziellen Skripts („bug-for-bug identical to the
> published leaderboard") — aber es bedeutet: **diese fünf Spalten tragen genau eine
> Information.** Berichtenswert sind daher **nDCG** und **AP**.

### Tabelle B — Tiefen-/Top-1-Familie
Diese Metriken werden bei der **Geometrie-Tiefe K** abgeschnitten (K = 50 in der BASE), weil
die offiziellen Metriken bei f abschneiden und dadurch für die Geometrie-Arme blind wären.

| Metrik | Was sie misst |
|---|---|
| **NN_cat** | Ist der **Top-1**-Treffer in der richtigen **Kategorie**? |
| **NN_sub** = **hit@1** | Ist der **Top-1**-Treffer in der richtigen **Subkategorie**? ← *die in diesem Dokument durchgehend berichtete Top-1-Metrik* |
| **MRR** | Kehrwert des Rangs des ersten Kategorie-Treffers, gemittelt (1.0 = immer Rang 1) |
| **MRR_sub** | dasselbe auf Subkategorie-Ebene |
| **mAP_K** | Average Precision innerhalb der Tiefe K, normiert auf min(K, R) |
| **nDCG_K** | nDCG innerhalb der Tiefe K, mit **korrigiertem** DCG (s. u.) |
| **hit_sub@N** | Ist in den Top-N mindestens ein Subkategorie-Treffer? (N = 1, 5, 10, 20, 50, 100) |

### Warum hit@1 die Subkategorie nutzt
Die Kategoriezuordnung („ist es ein Stuhl?") löst der Sprachkanal weitgehend allein. Die offene
Frage ist, ob das System die richtige **Variante** innerhalb der Kategorie findet — und genau
das entscheidet, ob das gelieferte CAD zur Pose taugt. Deshalb ist **NN_sub** die Bezugsgröße
für die Geometrie-Arme und die Brücke zu Stage 3. Für die BASE: NN_cat 0.392 vs. **NN_sub 0.340**.

### Zwei nDCG, die nie dieselbe Spalte teilen
- **`nDCG` (Tabelle A)** nutzt das offizielle `metrics.dcg` — inklusive eines bekannten
  Off-by-one (Rang 1 wird doppelt gezählt, das letzte Element fällt weg). **Unverändert
  gelassen** für Leaderboard-Vergleichbarkeit.
- **`nDCG_K` (Tabelle B)** nutzt eine **korrigierte** DCG-Formel.

Es sind also **verschiedene Größen** und werden nie gemischt.

---

## 1. Auf einen Blick

| Konfiguration | nDCG | hit@1 | Bemerkung |
|---|---|---|---|
| CLIP-Text allein | 0.4218 | 0.130 | schwächster Einzelkanal |
| ULIP-2 Shape allein (pc) | 0.5353 | 0.328 | cross-Modus fällt auf 0.4809 |
| DINOv2 View allein | 0.5506 | 0.334 | stärkster Einzelkanal |
| Text + View (OSCAR-Kanäle) | 0.5519 | 0.312 | die OSCAR-Kanalmenge |
| **Volle Fusion (BASE)** | **0.5868** | **0.341** | **+0.035 durch S_shape** |
| **+ Geometrie (dGeDi+RANSAC, K=50)** | **0.6405** | **0.472** | **+0.054 nDCG, +0.130 hit@1** |
| OSCAR-Kaskade (τ=0.37 → DINO) | 0.4561 | 0.235 | Baseline; volle Fusion **+0.131** |

---

## 1.1 Vergleich mit dem offiziellen SHREC'18-Track

Offizielle Ergebnisse aus Pham et al., *SHREC'18: RGB-D Object-to-CAD Retrieval*,
3DOR 2018, **Tabelle 3** (DOI 10.2312/3dor.20181052). Zwei Auswertungsstrategien:
*standard* und *weighted* (jede Query invers zur Häufigkeit ihrer GT-Kategorie gewichtet,
um die Klassenunwucht zu adressieren).

**Offizielle Track-Beiträge** (Tabelle 3 des Papers, Test-Split) — berichtet werden der
**beste Beitrag** und der **beste punktbasierte** Ansatz, letzterer als direktes Pendant zu
unserem Shape-Kanal. Die übrigen sieben Runs (weitere Tran-, Li- und Khoi-Varianten) liegen
dazwischen und tragen nichts Zusätzliches bei.

| Team | Run | Ansatz | Precision | Recall | mAP | **NDCG** | **w-NDCG** |
|---|---|---|---|---|---|---|---|
| **Tran** | view-ring-bow-2 | view-based, bester Beitrag | 0.820 | 0.820 | 0.820 | **0.801** | 0.742 |
| **Khoi** | pointnet | **punktbasiert**, bester seiner Klasse | 0.706 | 0.706 | 0.706 | **0.665** | 0.647 |

**OSCAR+ auf demselben Test-Split (649 Queries)** — mit dem unveränderten offiziellen
`metrics.py` gerechnet, alle sieben Metriken:

| Arm | n | **nDCG** | precision | recall | F1 | **AP** | NNT1 | NNT2 |
|---|---|---|---|---|---|---|---|---|
| nur CLIP-Text (S_text) | 649 | 0.4102 | 0.1371 | 0.1371 | 0.1371 | 0.0429 | 0.1371 | 0.1371 |
| nur ULIP-2 (S_shape) | 649 | 0.5437 | 0.2383 | 0.2383 | 0.2383 | 0.1428 | 0.2383 | 0.2383 |
| nur DINOv2 (S_view) | 649 | 0.5524 | 0.2572 | 0.2572 | 0.2572 | 0.1508 | 0.2572 | 0.2572 |
| **OSCAR+ (BASE)** | 649 | **0.5945** | 0.2819 | 0.2819 | 0.2819 | **0.1646** | 0.2819 | 0.2819 |
| **OSCAR+ (+Geometrie)** | 649 | **0.6434** | 0.2826 | 0.2826 | 0.2826 | **0.1719** | 0.2826 | 0.2826 |

*Die w-Spalten (weighted) berechnen wir nicht — der Track gewichtet dort jede Query invers zur
Häufigkeit ihrer GT-Kategorie; unsere Auswertung nutzt durchgehend die Standard-Strategie.
Auch hier fallen `precision = recall = F1 = NNT1 = NNT2` zusammen (§0.1) — in den
Track-Beiträgen ebenso, was die gemeinsame Metrikdefinition bestätigt.*

### Was übereinstimmt ✓
- **Datensatz identisch:** 2101 RGB-D-Queries, 3308 CADs, 20 Kategorien, 43 Subkategorien.
- **Metrikdefinition identisch:** NDCG mit abgestufter Relevanz (2 = Kategorie *und*
  Subkategorie, 1 = nur Kategorie, 0 = falsch); alle Metriken über die **ersten K** Treffer,
  **K = Größe der GT-Kategorie**. Genau das rechnet unser `score_official` (`x = rel[:f]`).
- **Damit ist auch die Precision-=-Recall-Beobachtung (§0.1) im Paper selbst angelegt** — sie
  ist eine Eigenschaft der Track-Auswertung, kein Artefakt unserer Implementierung.

### Der Test-Split ist rekonstruierbar ✓
Das Paper: *„we instead provide example ranked lists for every RGB-D queries in the training
set"* — die Trainings-Queries haben also Beispiel-Rankinglisten, die Test-Queries nicht. Genau
diese Listen liegen in `eval/datasets/shrec18/shrec18_full/results/`:

| | Anzahl | Anteil |
|---|---|---|
| Queries mit Rankingliste → **Training** | 1452 | 69,1 % |
| Queries ohne → **Test** | **649** | **30,9 %** |

Das reproduziert das im Paper genannte **70/30**-Verhältnis. Zwei unabhängige Belege, dass es
wirklich die Trainings-Listen sind und nicht die GT: (a) die Dateien enthalten nur **5 Einträge
mit Score 0.000**, also *partielle* Listen — das Paper sagt *„not exhaustive, and only cover a
subset of the ground truth pairings"*; (b) unsere GT stammt aus einer **anderen Quelle**
(`rgbd.csv` + `cad.csv`, alle 2101/3308 mit echten Labels).

**Damit können wir auf exakt demselben Query-Set berichten.** Die Werte oben sind auf den
**649 Test-Queries** gerechnet — rein offline aus den Per-Query-Records, ohne Neulauf:

| Arm | Test-Split (649) | alle 2101 | Δ |
|---|---|---|---|
| OSCAR+ (BASE) | **0.5945** | 0.5868 | +0.008 |
| OSCAR+ (+Geometrie) | **0.6434** | 0.6405 | +0.003 |
| nur DINOv2 | 0.5524 | 0.5506 | +0.002 |
| nur ULIP-2 | 0.5437 | 0.5353 | +0.008 |
| nur CLIP-Text | 0.4102 | 0.4218 | −0.012 |

Die Unterschiede sind durchweg klein (≤ 0.012) — der Test-Split ist also **weder leichter noch
schwerer** als der Gesamtdatensatz. Das ist ein nützlicher Nebenbefund: unsere übrigen
Stage-1-Zahlen auf allen 2101 Queries sind dadurch nicht verzerrt.

### Zwei Gründe, warum die Zahlen dennoch **nicht** direkt vergleichbar sind ⚠️
1. **Überwacht vs. trainingsfrei.** Tabelle 2 des Papers: *„all of them are based on supervised
   deep learning"*. Alle Teilnehmer nutzten die Trainings-Rankinglisten, um die 20 Kategorien zu
   rekonstruieren, und trainierten darauf Klassifikatoren. **OSCAR+ sieht keine
   Kategorielabels und trainiert nichts** — alle Encoder sind eingefroren.
2. **Andere Aufgabenformulierung.** Das Paper schreibt explizit: *„the retrieval problem can be
   reformulated as a classification problem. We do this by simply return objects with the same
   predicted labels."* Da die GT **alle** CADs derselben Kategorie umfasst, liefert eine korrekte
   20-Wege-Klassifikation bei K = |Kategorie| nahezu perfekte Precision/Recall. Das erklärt die
   hohen Werte und ist eine **grundlegend andere Aufgabe** als offenes Retrieval.

### Wie man es berichten sollte
Nicht als „OSCAR+ ist schlechter als der Track", sondern als **Einordnung zweier
Aufgabenstellungen**: die Track-Teilnehmer lösen eine überwachte 20-Wege-Klassifikation auf
einem 30-%-Testsplit; OSCAR+ löst offenes, trainingsfreies Retrieval über den vollen
Datensatz. Query-Set und Metrikdefinition sind jetzt **identisch**; was bleibt, ist der
Unterschied in Aufgabenstellung und Supervision (Punkte 1 und 2) — und der lässt sich nicht
wegrechnen, nur benennen.

*Nebenbefund:* der beste punktbasierte Ansatz (PointNet, NDCG 0.665) liegt **0.136 unter** dem
besten view-basierten (0.801) — dieselbe Rangfolge der Modalitäten wie in unserem Block A, wo
DINOv2 (0.5524) den Shape-Kanal (0.5437) schlägt, dort allerdings mit **deutlich kleinerem
Abstand** (0.009 statt 0.136). Die Lücke zwischen Bild und Geometrie ist mit modernen
Foundation-Encodern also erheblich geschrumpft.

---

## 2. Block A — Kanal-Design (isoliert, ein Kanal, keine Fusion)

Jeder Design-Ablation wird **isoliert** gefahren, damit nur die geänderte Variable wirkt;
der fusionierte Wert steht daneben, wo er eine andere Geschichte erzählt.

### A1 · Appearance-Encoder: DINOv2 vs. SigLIP ◆
| Arm | nDCG | mAP | hit@1 | MRR |
|---|---|---|---|---|
| **DINOv2** | **0.5506** | 0.1548 | **0.334** | 0.477 |
| SigLIP | 0.5165 | 0.0861 | 0.264 | 0.458 |

**DINOv2 gewinnt klar** — signifikant (+0.034 nDCG, p<0.0001; hit@1 +0.070). SigLIP wurde
fair mit seinem nativen MAP-Head gepoolt (der frühere Wert 0.5245 nutzte einen degenerierten
Patch-0-Token).

### A2 · Anzahl Render-Views (Appearance) ◇
| Views | nDCG | mAP | hit@1 |
|---|---|---|---|
| 8 | 0.5302 | 0.1317 | 0.303 |
| 16 | 0.5481 | 0.1563 | 0.326 |
| 32 | 0.5426 | 0.1475 | 0.321 |
| **42** | **0.5506** | 0.1548 | **0.334** |

**Sättigung ab 16 Views** (V16 = 99,5 % von V42); der V32-Einbruch liegt im Rauschen.

### A3 · Shape-Encoder: ULIP-2 vs. Uni3D (pc) ◆
| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| ULIP-2 | 0.5353 | 0.1386 | **0.328** |
| Uni3D | 0.5337 | 0.1514 | 0.309 |

**Unentschieden** (nDCG praktisch gleich). Auf **hit@1 ist ULIP-2
besser** (+0.018, p=0.038). → **ULIP-2 behalten** — es hat zusätzlich den Cross-Modus, den
Uni3D nicht besitzt (Uni3D ist pc-only).

### A4 · Shape-Referenz: Partial-Views vs. Full-Mesh ◆
| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| **Partial-Views (BASE)** | **0.5353** | 0.1386 | **0.328** |
| Full-Mesh | 0.4858 | 0.1376 | 0.279 |

**Der stärkste Einzeleffekt in Block A: +0.0495 nDCG (p<0.0001), +0.049 hit@1.** Eine
partielle Referenz ist geometrisch vergleichbar mit der partiellen Query — das Full-Mesh
sieht die Rückseite, die der Sensor nie sieht.

### A5 · Query-Modus: pc vs. cross ◇
| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| **pc** (Query-Punktwolke) | **0.5353** | 0.1386 | **0.328** |
| cross (Query-Bild) | 0.4809 | 0.0926 | 0.264 |

**−0.054 nDCG ohne Tiefe.** Das ist die **Brücke zu Stage 2**: MI3DOR hat keine Tiefe, muss
also cross fahren — dieser Arm beziffert exakt, was das kostet.

### A6 · Query-Farben: XYZ+RGB vs. XYZ-only ◇
| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| XYZ+RGB (BASE) | 0.5353 | 0.1386 | 0.328 |
| **XYZ-only** | **0.5422** | 0.1557 | **0.360** |

**Farbe schadet leicht — aber systematisch.** XYZ-only gewinnt in **1152 von 1999**
nicht-gleichen Queries, auf hit@1 sogar deutlich (+0.032).
⚠️ **Konfundiert:** der XYZ-Arm tauscht den ganzen ULIP-Turm mit (ViT-B/512-d/8k Punkte statt
ViT-g/1280-d/10k) — es gibt keinen ViT-g-XYZ-Checkpoint. Also **keine saubere
Farb-Ablation**; so berichten.

### A7 · Anzahl Shape-Gallery-Views ◇
| Views | nDCG | mAP | hit@1 |
|---|---|---|---|
| 8 | 0.5119 | 0.1218 | 0.291 |
| 16 | 0.5227 | 0.1340 | 0.308 |
| 32 | 0.5300 | 0.1314 | 0.317 |
| **42 (BASE)** | **0.5353** | 0.1386 | **0.328** |

**Monoton — mehr Shape-Views helfen durchgehend**, anders als Appearance (A2), das ab 16
sättigt.

---

## 3. Block B — Fusion

### B1 · Fusionsstrategie: Weighted-Sum vs. RRF ◆
| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| **Weighted Sum (BASE)** | **0.5868** | 0.1666 | **0.341** |
| Reciprocal Rank Fusion | 0.5744 | 0.1379 | 0.318 |

**Weighted Sum gewinnt signifikant** (+0.0124, p<0.0001; Bilanz 1320:718). RRF ist der
Standard (Cormack k=60), aber seine Konstante ist auf TREC-Listenlängen kalibriert; als
negatives Ergebnis berichtet, nicht nachtuniert.

### B2 · Gewichts-Sensitivität (Heatmap) ◇
*Hinweis: als einzige Ablation noch bei **16v/k8** gerechnet — als Sensitivitätsaussage
(„BASE liegt nahe am Optimum") von der Shape-Config unabhängig, daher nicht nachgefahren.*
- **pc-Modus:** Optimum (0.2, 0.4, 0.4) = 0.5916 vs. BASE 0.5889 → **+0.003, Rauschen.**
  Die BASE-Gewichte sind also nicht getunt, aber auch nicht schlecht gewählt.
- **cross-Modus** (Stage-2-Brücke): Optimum verschiebt sich auf (0.3, **0.6**, 0.1) = 0.5567,
  und BASE (0.5453) fällt **unter View-only (0.5506)** — ohne Tiefe muss der Shape-Kanal
  heruntergewichtet werden. Die pc-Gewichte übertragen sich **nicht**.

### B3 · Kanalbeitrag + OSCAR-Baseline ◆
| Konfiguration | nDCG | hit@1 |
|---|---|---|
| S_text | 0.4218 | 0.130 |
| S_shape | 0.5353 | 0.328 |
| S_view | 0.5506 | 0.334 |
| S_text + S_view (**OSCAR-Kanäle**) | 0.5519 | 0.312 |
| **volle Fusion** | **0.5868** | **0.341** |
| OSCAR-Kaskade (τ=0.37 → DINO argmax) | 0.4561 | 0.235 |

**Der Kernbefund von OSCAR+:** S_shape zu OSCARs Text+View hinzuzufügen bringt **+0.035 nDCG**
und **+0.029 hit@1**. Gegen die *echte* OSCAR-Kaskade beträgt der Vorsprung **+0.131 nDCG**.

**Zur OSCAR-Kaskade.** Sie prunt per CLIP-Text-Schwellwert τ=0.37 auf eine Shortlist und
arg-maxt darin über DINOv2 (kein Shape). Auf SHREC'18 greift der Schwellwert praktisch nie:
er prunt bei **98,3 % der Queries auf leer** und fällt dann auf Top-20 zurück — die Kaskade ist
faktisch „CLIP-Top-20 → DINO". Der Rückstand ist also **keine Frage der Parametrierung**,
sondern der Architektur: jedes Pruning verwirft Kandidaten, bevor die anderen Kanäle sie
bewerten konnten, während die volle Fusion alle 3308 CADs simultan bewertet.

---

## 4. Block C — Geometrie-Reranking (auf der fusionierten Rangliste)

Alle Varianten ordnen die **Top-K = 50** der Fusion um. `D_trim` = getrimmte einseitige
Oberflächendistanz (oberste 10 % verworfen, robust gegen Partialität).

### C1 · Geometrie-Signal ◆

**Backend: dGeDi-Dienst** (GeDi-Deskriptoren + RANSAC), derselbe wie Stage 3, mit
`use_icp=True`. Alle Varianten ordnen die **Top-K = 50** der Fusion um.

| Arm | Rangkriterium | nDCG | mAP | hit@1 | MRR |
|---|---|---|---|---|---|
| keine (= BASE) | fusionierter Score | 0.5868 | 0.1666 | 0.341 | 0.478 |
| `fitness` | RANSAC-**Inlier-Anteil** | 0.6251 | 0.1680 | 0.439 | 0.606 |
| **`chamfer_ransac`** | **getrimmte Oberflächendistanz nach Ausrichtung** | **0.6405** | 0.1737 | **0.472** | **0.638** |
| `both` (Borda) | Rangfusion Fitness ⊕ Distanz | 0.6362 | 0.1711 | 0.465 | 0.626 |

**Ergebnis:** die **ausgerichtete Distanz** ist das beste Geometriesignal — nDCG
0.5868 → 0.6405 (+0.054), und vor allem **hit@1 0.341 → 0.472 = +38 % relativ**, MRR
0.478 → 0.638. Da die Pose-Stufe nur den Top-1 konsumiert, ist der Top-1-Gewinn die
entscheidende Zahl; reines nDCG unterschätzt den Beitrag um Faktor ~2,5.

Die **Fitness allein** (nur „wie gut lassen sie sich überhaupt ausrichten") bleibt um
0.015 nDCG / 0.033 hit@1 zurück — die Distanz **nach** der Ausrichtung trägt also echte
Zusatzinformation. **Borda** verwässert wieder leicht (0.6362), weil die schwächere
Fitness-Stimme gleichberechtigt eingeht.

> ⚠️ **Zwei Arme des ursprünglichen Designs sind unter dGeDi nicht messbar** und deshalb
> oben **nicht** aufgeführt:
> - **`chamfer_unaligned`** (die „Distanz ohne Ausrichtung"-Kontrolle) — der dGeDi-Pfad
>   schreibt kein `d_unaligned`-Feld; `geometry_score` fällt dann auf den Tiebreak
>   `fitness` zurück, d. h. der Arm **degeneriert still zum Fitness-Arm** (deshalb
>   bitidentische 0.6251). Er misst **nicht**, was sein Name sagt.
> - **`chamfer_icp`** — dGeDi läuft mit `use_icp=True` und führt RANSAC→ICP **intern
>   zusammen** aus; der Code setzt `d_icp = d_ransac`. Es gibt nur **eine** Distanz, ein
>   separater ICP-Effekt ist nicht isolierbar. Folgerichtig heißt `chamfer_ransac` hier
>   genau genommen **RANSAC→ICP**.

### C2 · Shortlist-Tiefe K ◇
Alle Werte bei **42v/k5** (K=20/5 aus dem K=50-Registrierungs-Cache abgeleitet, keine neuen
RANSAC-Läufe — dieselbe Ableitung wie im Original-Sweep).

| Rangkriterium innerhalb der Shortlist | K=50 | K=20 | K=5 |
|---|---|---|---|
| **ausgerichtete Distanz** *(Sieger)* | **0.6405** | 0.6279 | 0.6022 |
| Distanz ⊕ Fitness (Borda) | 0.6362 | 0.6240 | 0.6001 |
| Geometrie ⊕ Fusions-Rang (Borda) | 0.6287 | 0.6153 | 0.5979 |
| nur RANSAC-Fitness | 0.6251 | 0.6171 | 0.5980 |
| Geometrie auf Text+View *(ohne Shape-Kanal)* | 0.5961 | 0.5820 | 0.5623 |
| *hit@1 der ausgerichteten Distanz* | *0.472* | *0.464* | *0.426* |

**Was die fünf Kriterien unterscheiden** (alle nutzen dieselbe dGeDi-Registrierung):
- **ausgerichtete Distanz** — sortiert nach der getrimmten Oberflächendistanz *nach* der
  Ausrichtung. Das beste Kriterium.
- **Distanz ⊕ Fitness** — Rangfusion (Borda) der beiden Geometriesignale; der Fusions-Score
  der Kanäle wird innerhalb der Shortlist verworfen.
- **Geometrie ⊕ Fusions-Rang** — behält zusätzlich den Rang aus der Kanalfusion als dritten
  Stimmgeber. Beantwortet: *trägt der Fusions-Score noch Information, wenn Geometrie da ist?*
- **nur RANSAC-Fitness** — sortiert nur nach dem Inlier-Anteil, ohne Distanz.
- **Geometrie auf Text+View** — Geometrie-Rerank auf einer Fusion **ohne** Shape-Kanal.
  Isoliert, was die Geometrie beiträgt, wenn S_shape fehlt.

**Tiefer ist besser** (+0.038 nDCG und **+0.046 hit@1** über den Bereich) — das richtige Modell
liegt oft jenseits von Rang 5. Der Effekt ist über **alle** Geometrie-Arme konsistent, also eine
Eigenschaft der Shortlist-Tiefe, nicht eines einzelnen Signals. K=50 ist die BASE-Tiefe.

*Robustheit:* die alte Config (16v/k8) ergab 0.6406 / 0.6287 / 0.6041 — praktisch identisch, die
Schlussfolgerung hängt also nicht an der Shape-Config.
**Hinweis:** Stage 3 nutzt **K=5** — bewusst anders (6× mehr Queries, und dort schadet
Geometrie ohnehin; siehe `CONFIG_COMPARISON.md` §7).

### C3 · Shape vs. Geometrie — redundant? ◇
| Konfiguration | nDCG | hit@1 |
|---|---|---|
| Text+View (weder Shape noch Geometrie) | 0.5519 | 0.312 |
| + S_shape in der Fusion (= BASE) | 0.5868 | 0.341 |
| + GeDi-Rerank auf der Text+View-Shortlist (ohne Shape) | 0.5961 | 0.406 |
| + **beides** (Shape in Fusion, dann GeDi) | 0.6251 | 0.439 |
| GeDi ⊕ Basis-Rang (Borda) | 0.6287 | 0.459 |

**Nicht redundant — komplementär.** Beide heben Text+View um jeweils ~+0.04 und **stapeln
sich**. „GeDi-only" heißt dabei: GeDi rerankt die **Text+View**-Shortlist — Geometrie ist kein
eigenständiger Kanal (ein Full-Database-S_GeDi wäre ~830 h pro Zelle, strukturell nicht machbar).

---

## 5. Wie stabil sind die Unterschiede?

Jeder Arm wird über dieselben 2101 Queries ausgewertet, deshalb lässt sich **je Query**
vergleichen, welcher Arm besser war. Die Bilanz dieser Einzelvergleiche sagt mehr als der
Abstand der Mittelwerte allein: ein Vorsprung, der aus wenigen großen Ausschlägen stammt,
sieht im Mittelwert gleich aus wie einer, der auf breiter Front entsteht.

| Vergleich (nDCG) | Δ | gewonnene Queries | Einordnung |
|---|---|---|---|
| Geometrie: keine vs. GeDi+RANSAC | −0.0537 | 599 : **1264** | Geometrie gewinnt breit |
| Partial vs. Full-Mesh (isoliert) | +0.0495 | **1015** : 974 | knappe Bilanz, großer Abstand |
| DINOv2 vs. SigLIP (isoliert) | +0.0341 | **1213** : 811 | DINOv2 gewinnt breit |
| Weighted-Sum vs. RRF | +0.0124 | **1320** : 718 | Weighted-Sum gewinnt breit |
| XYZ+RGB vs. XYZ-only (isoliert) | −0.0068 | 847 : **1152** | Farbe schadet leicht, aber stetig |
| ULIP-2 vs. Uni3D (fusioniert) | −0.0045 | 1009 : 1027 | **Gleichstand** |
| ULIP-2 vs. Uni3D (isoliert) | +0.0017 | ausgeglichen | **Gleichstand** |
| Config 16v/k8 → 42v/k5 | +0.0021 | 938 : 1064 | **Nulleffekt** |

Zwei Zeilen verdienen Beachtung, weil Mittelwert und Bilanz auseinanderlaufen:

**ULIP-2 gegen Uni3D** sieht im Mittelwert nach einem Uni3D-Vorsprung aus (−0.0045),
die Bilanz steht aber mit 1009:1027 praktisch unentschieden — der Abstand entsteht aus
wenigen Ausreißern, nicht aus durchgängig besseren Rankings. Auf hit@1 liegt ULIP-2 sogar
vorn (+0.018). Wir bleiben deshalb bei ULIP-2, das zusätzlich den Cross-Modal-Zweig hat,
den Uni3D nicht besitzt.

**Die Config-Umstellung 16v/k8 → 42v/k5** ist mit 938:1064 ebenfalls ein Nulleffekt. Sie
hat Vergleichbarkeit über die Stages hergestellt, keine besseren Zahlen.

Auf **hit@1** werden alle breiten Vorsprünge größer (Geometrie −0.130, SigLIP +0.070,
Full-Mesh +0.049, RRF +0.024).

**Zwei frühere Aussagen werden dadurch korrigiert:** „Uni3D gewinnt fusioniert" ist **kein
echter Effekt**, und die **Konfig-Korrektur (16v/k8 → 42v/k5) ist ein Nulleffekt** — sie hat
Vergleichbarkeit über die Stufen gebracht, nicht andere Zahlen.

---

## 6. Kategorien-Analyse — was sehen die Embeddings unterschiedlich?

Per-Kategorie-nDCG der isolierten Kanäle (Kategorien mit n ≥ 20; 2101 Queries):

| Kategorie | n | text | view | shape | Fusion | +Geom | bester Einzelkanal |
|---|---|---|---|---|---|---|---|
| chair | 513 | 0.615 | 0.881 | **0.898** | 0.890 | 0.902 | shape |
| display | 192 | **0.669** | 0.603 | 0.628 | 0.698 | 0.748 | text |
| table | 140 | **0.523** | 0.482 | 0.392 | 0.445 | 0.462 | text |
| sofa | 139 | 0.164 | **0.473** | 0.226 | 0.490 | 0.601 | view |
| bin | 133 | **0.438** | 0.371 | 0.337 | 0.486 | 0.496 | text |
| desk | 118 | 0.289 | **0.429** | 0.309 | 0.396 | 0.468 | view |
| storage | 116 | 0.539 | **0.671** | 0.609 | 0.670 | 0.684 | view |
| book | 90 | 0.274 | 0.291 | **0.490** | 0.399 | 0.448 | shape |
| bookshelf | 86 | 0.211 | **0.496** | 0.400 | 0.434 | 0.512 | view |
| box | 78 | 0.167 | **0.224** | 0.034 | 0.197 | 0.271 | view |
| **keyboard** | 65 | 0.149 | 0.090 | **0.716** | 0.422 | **0.805** | shape |
| **bag** | 64 | 0.020 | 0.080 | **0.384** | 0.100 | 0.181 | shape |
| bed | 62 | 0.490 | **0.689** | 0.524 | 0.738 | 0.795 | view |
| machine | 58 | 0.357 | **0.442** | 0.409 | 0.537 | 0.630 | view |
| light | 52 | **0.614** | 0.604 | 0.450 | 0.602 | 0.597 | text |
| pillow | 46 | 0.195 | 0.767 | **0.808** | 0.840 | 0.592 | shape |
| printer | 41 | 0.233 | **0.382** | 0.282 | 0.481 | 0.489 | view |
| oven | 40 | 0.185 | 0.231 | **0.242** | 0.282 | 0.400 | shape |
| **pc** | 39 | 0.215 | 0.206 | **0.359** | 0.320 | **0.603** | shape |
| **cup** | 29 | 0.259 | **0.451** | 0.032 | 0.338 | **0.605** | view |

### Die Spezialisierung ist systematisch
- **S_shape gewinnt** bei geometrisch markanten, visuell unauffälligen Objekten:
  **keyboard** (0.716 vs. 0.090 view!), **bag** (0.384 vs. 0.080), book, pillow, chair, pc.
  Eine Tastatur ist flach-rechteckig mit charakteristischem Profil, aber optisch fast
  merkmalsfrei — genau der Fall, für den der 3D-Kanal existiert.
- **S_view gewinnt** bei texturreichen Möbeln: sofa, bed, bookshelf, storage, desk, printer.
- **S_text gewinnt** bei semantisch trennscharfen Klassen: **display**, table, bin, light —
  Kategorien, deren LLaVA-Beschreibungen distinkt sind („a flat screen monitor…").
- **Totalausfälle sind aufschlussreich:** `cup` mit S_shape = **0.032** (Tassen sind als
  Punktwolke fast nicht unterscheidbar — der Henkel verschwindet in der Partialansicht),
  `bag` mit S_text = **0.020** (Taschen sind sprachlich generisch).

### ⚠️ Der wichtigste Befund: die feste Gewichtung **schadet** in 12 von 20 Kategorien
| Kategorie | bester Einzelkanal | Fusion | Verlust |
|---|---|---|---|
| **keyboard** | shape 0.716 | 0.422 | **−0.294** |
| **bag** | shape 0.384 | 0.100 | **−0.284** |
| cup | view 0.451 | 0.338 | −0.113 |
| book | shape 0.490 | 0.399 | −0.091 |
| table | text 0.523 | 0.445 | −0.078 |
| bookshelf | view 0.496 | 0.434 | −0.062 |
| pc, desk, box, light, chair, storage | — | — | −0.039 … −0.001 |

Wo **ein** Kanal dominiert, zieht die feste Gewichtung (0.3/0.4/0.3) ihn zu den schwachen
Kanälen herunter. Der Gesamt-nDCG-Gewinn der Fusion (+0.035) ist also ein **Mittelwert über
gegenläufige Effekte**: große Gewinne dort, wo die Kanäle sich ergänzen — spürbare Verluste
dort, wo einer allein recht hätte. Das ist das stärkste Argument für **adaptive/gelernte
Gewichte** (oder eine Per-Query-Kanalauswahl) als Ausblick.

### Geometrie repariert genau die Shape-dominierten Fälle
| Kategorie | Fusion → +Geometrie | Δ |
|---|---|---|
| keyboard | 0.422 → **0.805** | **+0.383** |
| pc | 0.320 → 0.603 | +0.283 |
| cup | 0.338 → 0.605 | +0.267 |
| oven | 0.282 → 0.400 | +0.117 |
| sofa | 0.490 → 0.601 | +0.110 |
| … | | |
| light | 0.602 → 0.597 | −0.005 |
| **pillow** | 0.840 → **0.592** | **−0.248** |

Geometrie holt dort am meisten, wo die Fusion den Shape-Kanal verwässert hat (keyboard, pc)
**und** wo der Shape-Kanal versagt hat (cup) — die Ausrichtung liefert Evidenz, die kein
globales Embedding hat. **Ausnahme `pillow`:** hier schadet sie deutlich; weiche,
deformierbare Objekte haben keine stabile Starrkörper-Ausrichtung, RANSAC findet
Scheinkorrespondenzen. Das ist die inhaltliche Grenze der Methode.

---

## 7. Diskussionswürdige Punkte

1. **nDCG unterschätzt den Geometrie-Beitrag um Faktor ~2,5.** Die Geometrie hebt nDCG um
   +0.054 (+9 % rel.), **hit@1 aber um +0.130 (+38 % rel.)**. Wer nur nDCG berichtet, hält den
   Beitrag für moderat — auf der Metrik, die die Pose-Stufe tatsächlich konsumiert, ist er
   2,5-mal so groß. Die Folge ist konkret: die beiden Metriken küren teils **verschiedene
   Sieger**. Text+View führt auf nDCG (0.5519 vs. 0.5506 für View allein), fällt auf hit@1 aber
   deutlich zurück (0.312 vs. 0.334). Welche Metrik man berichtet, entscheidet also, welche
   Designentscheidung man trifft.
2. **Der Text-Kanal verbessert die Liste, verschlechtert aber den Top-1.** View allein: nDCG
   0.5506 / hit@1 **0.334**. Text+View: nDCG 0.5519 (**+0.001**) / hit@1 **0.312** (**−0.022**).
   CLIP-Text ist ein *breites* Kategoriesignal — es zieht relevante Objekte in die Liste, drängt
   aber falsche auf Rang 1. Für ein reines Retrieval-Ranking ist das ein Gewinn, für eine
   Pose-Pipeline ein Verlust.
3. **Ohne den Shape-Kanal sind einzelne Klassen praktisch nicht retrievierbar.** `keyboard`
   0.716 (shape) vs. 0.090 (view) vs. 0.149 (text); `bag` 0.384 vs. 0.080 vs. 0.020. Der
   durchschnittliche Fusionsgewinn (+0.035) verdeckt, dass S_shape für flache, texturarme
   Objekte der **einzige funktionierende Kanal** ist. Das ist das stärkere Argument für OSCAR+
   als der Mittelwert.
4. **Feste Gewichte sind ein Kompromiss mit Kosten** — nachweisbar negativ in 12 von 20
   Kategorien (§6), bis zu −0.294 (`keyboard`). Der Fusionsgewinn ist ein Mittelwert über
   gegenläufige Effekte; adaptive oder per-Query gewählte Gewichte sind der naheliegende
   Ausblick.
5. **Der Tiefenverlust trifft den Top-1 doppelt so hart wie die Liste.** pc → cross: nDCG
   −10,2 %, hit@1 **−19,5 %**. Für Stage 2 (monokular) und die Pose-Stufe ist der Verzicht auf
   Tiefe also teurer, als die nDCG-Zahl allein vermuten lässt.
6. **Appearance sättigt, Shape nicht.** DINOv2 ist ab 16 Views flach (V16 = 99,5 % von V42),
   ULIP-2 steigt monoton bis 42 (+0.023 von V8). Renderings werden schnell redundant, während
   Partialansichten immer neue Oberflächenregionen abdecken — **unterschiedliche View-Budgets
   pro Kanal** wären effizienter als ein gemeinsamer Wert.
7. **Grenzen der Geometrie:** −0.248 bei `pillow` (deformierbar, keine stabile
   Starrkörper-Ausrichtung). Zusammen mit Stage 3 (Geometrie schadet der Pose) ergibt das:
   Geometrie ist ein **Retrieval**-Werkzeug für starre, scan-saubere Objekte.
8. **Farbe im Query bringt nichts** (eher leicht negativ) — aber konfundiert mit dem
   Encoder-Wechsel; als offene Frage kennzeichnen, nicht als Ergebnis.

## 8. Anhang — alle 38 Arme

| Arm | Gruppe | nDCG | mAP | P/NN1 |
|---|---|---|---|---|
| E2_chamfer_ransac | C1 | 0.6405 | 0.1737 | 0.2802 |
| E2_chamfer_icp *(= _ransac, nicht abtrennbar)* | C1 | 0.6405 | 0.1737 | 0.2802 |
| E2_chamfer_unaligned *(degeneriert zu fitness)* | C1 | 0.6251 | 0.1680 | 0.2754 |
| E2_both | C1 | 0.6362 | 0.1711 | 0.2786 |
| O1e_gedi_with_base | C3 | 0.6287 | 0.1726 | 0.2802 |
| E2_fitness | C1 | 0.6251 | 0.1680 | 0.2754 |
| O1c_gedi_post_fusion | C3 | 0.5961 | 0.1382 | 0.2437 |
| E7_uni3d | A3 | 0.5913 | 0.1725 | 0.2862 |
| O5_xyz_only | A6 | 0.5880 | 0.1713 | 0.2802 |
| **E1c_full_fusion (BASE)** | B3 | **0.5868** | 0.1666 | 0.2790 |
| A7f_full_fusion_shape_V42 | A7 | 0.5868 | 0.1666 | 0.2790 |
| O4_V16 | A2 | 0.5820 | 0.1661 | 0.2777 |
| O4_V32 | A2 | 0.5800 | 0.1615 | 0.2740 |
| E6_rrf | B1 | 0.5744 | 0.1379 | 0.2577 |
| O4_V8 | A2 | 0.5714 | 0.1486 | 0.2607 |
| E4_siglip | A1 | 0.5659 | 0.1227 | 0.2388 |
| O2_visual_first *(gestrichen)* | — | 0.5570 | 0.1558 | 0.2579 |
| E1b_text_view | B3 | 0.5519 | 0.1360 | 0.2455 |
| E1_view_only / A2_V42 | A1/A2 | 0.5506 | 0.1548 | 0.2579 |
| A2_view_only_V16 | A2 | 0.5481 | 0.1563 | 0.2571 |
| A2_view_only_V32 | A2 | 0.5426 | 0.1475 | 0.2496 |
| O5_xyz_shape_only | A6 | 0.5422 | 0.1557 | 0.2409 |
| E1_shape_only / A7_V42 | A4/A7 | 0.5353 | 0.1386 | 0.2355 |
| E7_uni3d_shape_only | A3 | 0.5337 | 0.1514 | 0.2486 |
| A2_view_only_V8 | A2 | 0.5302 | 0.1317 | 0.2303 |
| A7_shape_only_V32 | A7 | 0.5300 | 0.1314 | 0.2301 |
| A7_shape_only_V16 | A7 | 0.5227 | 0.1340 | 0.2332 |
| O2_clip_threshold_cal *(gestrichen)* | — | 0.5186 | 0.0877 | 0.1808 |
| E4_siglip_only | A1 | 0.5165 | 0.0861 | 0.1944 |
| A7_shape_only_V8 | A7 | 0.5119 | 0.1218 | 0.2220 |
| E2b_fullmesh_shape_only | A4 | 0.4858 | 0.1376 | 0.2257 |
| E7_ulip2_cross_shape_only | A5 | 0.4809 | 0.0926 | 0.1914 |
| E1d_clip_pruned | B3 | 0.4566 | 0.0500 | 0.1424 |
| O2_clip_threshold | B3 | 0.4561 | 0.0499 | 0.1424 |
| E1_oscar_cascade | B3 | 0.4561 | 0.0501 | 0.1424 |
| E1a_text_only | B3 | 0.4218 | 0.0470 | 0.1424 |

*Aliase (identisch mit ihrem Basis-Arm) sind zusammengefasst: E2_none = E1c_full_fusion,
O1b = E1c, O2_full_database = E1c, O4_V42 = E1c, O5_xyzrgb = E1c, E2b_partial = E1c.*

## Stärkster Arm ohne Geometrie (Stand 2026-09-03)

Nach dem Farb-Fix für texturierte Meshes ist **`E2b_fullmesh` der stärkste Arm ohne
geometrisches Re-Ranking** — volle Fusion mit einer Full-Mesh-Shape-Referenz:

| Arm | nDCG | NN_sub |
|---|---|---|
| **`E2b_fullmesh`** | **0.5935** | **0.3598** |
| `E7_uni3d` | 0.5913 | 0.3455 |
| `O5_xyz_only` | 0.5880 | 0.3541 |
| `E1c_full_fusion` (BASE) | 0.5868 | 0.3413 |

Je Query gegen BASE über alle 2101 Queries:

| Metrik | Δ zugunsten Full-Mesh | gewonnene Queries |
|---|---|---|
| nDCG | 0.0067 | 904 : **1127** |
| NN_sub | 0.0186 | 143 : **182** |

Der Abstand ist klein, die Bilanz aber in beiden Metriken klar zugunsten von Full-Mesh —
also kein Ausreißereffekt, sondern ein durchgängiger, wenn auch kleiner Vorteil.

Isoliert bleibt es umgekehrt: der partielle Shape-Kanal ist für sich genauer
(0.5353 gegen 0.4956). Der Nutzen des vollständigen Meshes entsteht erst in der Fusion —
seine Fehler korrelieren offenbar weniger mit denen von Text und Erscheinung. Dasselbe
Muster zeigt Stage 3 (`3a_cross_fullmesh_v2` 0.5151 gegen 0.4818).

Welche Kategorien wo gewinnen: `python3 tools/compare_arms_by_category.py --preset fusion`.

## Die vollständige Shape-Matrix (Stand 2026-09-04)

Query-Modus × Gallery-Repräsentation, **isolierter** Shape-Kanal (nDCG / NN_sub):

| | partial | full-mesh |
|---|---|---|
| **pc** | **0.5353 / 0.3275** | 0.4956 / 0.2822 |
| **cross** | 0.4809 / 0.2637 | 0.4569 / 0.2028 |

Isoliert gewinnt **partial in beiden Modi**, und **pc in beiden Repräsentationen** — auf
SHREC'18, wo die Tiefendaten aus sauberen Scans stammen.

Fusioniert kehrt sich nur die Repräsentationsachse um:

| Arm | nDCG | NN_sub |
|---|---|---|
| `E2b_fullmesh` (pc × full-mesh) | **0.5935** | **0.3598** |
| `E1c_full_fusion` (BASE, pc × partial) | 0.5868 | 0.3413 |
| `E7_ulip2_cross_fullmesh` | 0.5511 | 0.3084 |

**Der Vergleich mit Stage 3 ist der eigentliche Befund.** Dort ist cross × full-mesh der
*beste* Arm (R@1 0.5151), hier der *schwächste* der drei. Der Unterschied liegt in der
Qualität der Tiefendaten: SHREC liefert saubere Scans, in denen die Punktwolken-Query
überlegen ist; BOP liefert verrauschte Sensortiefe in unaufgeräumten Szenen, wo das Bild
das verlässlichere Formsignal ist. Die Wahl des Query-Modus ist damit keine Design-
entscheidung, sondern eine Eigenschaft der Aufnahmesituation.
