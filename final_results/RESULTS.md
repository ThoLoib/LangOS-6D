# OSCAR+ — Gesamtergebnisse der Evaluation (Stage 1–4)

Masterarbeit TU Wien · Stand 2026-09-07 · Hardware: RTX 4090 (tessa-pc, Container + Host)

Dieses Dokument fasst alle vier Ergebnis-Artefakte in einer durchgehenden Erzählung zusammen.
Jede Zahl stammt aus einem konkreten Ergebnisordner; die Zuordnung Tabelle → Datei steht im
[README.md](README.md) dieses Ordners. Die interaktiven Fassungen mit allen Diagrammen liegen
unter [artifacts/](artifacts/).

**Konvention für Grafiken:** Blöcke der Form „📊 **Grafik-Empfehlung**“ markieren Stellen, an
denen eine grafische Umsetzung der danebenstehenden Tabelle für die Arbeit sinnvoll ist. Wo die
Grafik im HTML-Artefakt bereits existiert, ist das vermerkt — die Rohdaten je Grafik liegen als
CSV in diesem Ordner. Ein Verzeichnis aller Empfehlungen steht am [Ende des Dokuments](#verzeichnis-der-grafik-empfehlungen).

---

## Inhalt

1. [Das System und der rote Faden](#das-system-und-der-rote-faden)
2. [Stage 1 — SHREC'18: Welche Bausteine funktionieren?](#stage-1--shrec18-welche-bausteine-funktionieren)
3. [Stage 2 — MI3DOR: Überlebt die Fusion den Wegfall der Tiefe?](#stage-2--mi3dor-überlebt-die-fusion-den-wegfall-der-tiefe)
4. [Stage 3 — BOP: Taugt das gefundene CAD für die Pose?](#stage-3--bop-taugt-das-gefundene-cad-für-die-pose)
5. [Stage 4 — Latenz: Was kostet das alles?](#stage-4--latenz-was-kostet-das-alles)
6. [Übergreifende Diskussion](#übergreifende-diskussion)
7. [Verzeichnis der Grafik-Empfehlungen](#verzeichnis-der-grafik-empfehlungen)

---

## Das System und der rote Faden

**OSCAR+ in einem Absatz.** Aus einem RGB(-D)-Bild und einem Sprachprompt wird das Zielobjekt
segmentiert (GroundingDINO + SAM), in eine Punktwolke gehoben und über drei eingefrorene
Encoder-Kanäle gegen eine CAD-Gallery verglichen: **S_text** (CLIP ViT-B/32 gegen
Per-View-Beschreibungen), **S_view** (DINOv2-base über gerenderte Ansichten) und **S_shape**
(ULIP-2, wahlweise Query-Punktwolke = *pc-Modus* oder Query-Bild = *cross-Modus*). Die Kanäle
werden per Weighted Sum w = (0.3, 0.4, 0.3) über die volle Datenbank fusioniert; optional
rerankt ein Geometrie-Schritt (dGeDi-Deskriptoren + RANSAC) die Top-K. Das Top-1-CAD geht an
FoundationPose. Nichts wird trainiert; alle Encoder bleiben eingefroren.

**Die vier Stufen sind eine Fragenkette**, keine vier unabhängigen Benchmarks:

| Stufe | Datensatz | Frage | Bedingungen |
|---|---|---|---|
| **Stage 1** | SHREC'18 (2101 Queries / 3308 CADs) | Welche Encoder, Repräsentationen und Fusionsentscheidungen funktionieren — einzeln und zusammen? | kontrolliert: saubere Scans, Tiefe vorhanden, Klassen-Relevanz |
| **Stage 2** | MI3DOR (10 500 / 3848) | Überlebt die Konfiguration den Wegfall der Tiefe? | monokulare Bilder, cross-Modus erzwungen |
| **Stage 3** | BOP: YCB-V, T-LESS, LM-O (12 284 Instanzen / 1316 CADs) | Findet die Pipeline das *exakte* Objekt, und reicht notfalls ein Ersatzmodell für die Pose? | unaufgeräumte Szenen, Sensortiefe, Instanz-Relevanz, Millimeter |
| **Stage 4** | YCB-V + 59 Ziel-CADs | Was kostet eine Anfrage, was ein neues Objekt? | dieselbe Pipeline, Zeit statt Genauigkeit |

Die Übergaben zwischen den Stufen sind explizit gemessen: Stage-1-Arm **A5** beziffert vorab,
was der cross-Modus kostet, den Stage 2 fahren *muss*. Stage 1 und 2 zusammen zeigen, dass die
Antwort auf „partial oder full-mesh?“ vom Query-Modus und von der Fusion abhängt — Stage 3
prüft dieselbe Achse auf Instanz-Ebene. Das Geometrie-Reranking, das auf SHREC der stärkste
Hebel ist, schadet auf BOP an jeder Stelle; der Grund wird als **bedingte
Top-1-Genauigkeit** operationalisiert und ist vorab prüfbar. Stage 4 liefert die Kostenseite
genau der Entscheidungen, die Stage 1–3 nach Qualität treffen (View-Zahl, Shortlist-Tiefe K,
Repräsentation, Geometrie).

---

## Stage 1 — SHREC'18: Welche Bausteine funktionieren?

**Setup.** 2101 RGB-D-Queries gegen 3308 CADs in 20 Kategorien; 2683 CADs tragen zusätzlich
eine Subkategorie. Relevanz ist **abgestuft**: 2 = gleiche Kategorie *und* Subkategorie,
1 = nur gleiche Kategorie, 0 = irrelevant. **„Richtig“ heißt Klassenzugehörigkeit, nicht
Instanz** — ein Treffer zählt, wenn er *irgendein* CAD derselben (Sub-)Kategorie ist.
Konfiguration (BASE): CLIP ViT-B/32 (max über 42 Per-View-Beschreibungen) · DINOv2-base
(mean-Patch-Pooling, 42 Views, top-k-softmax k=5, τ=0.5) · ULIP-2 coloured 1280-d (pc-Modus,
partielle Referenz-Views, alle 42) · Weighted Sum w = (0.3, 0.4, 0.3) über die volle Datenbank ·
Geometrie-Arme rerangieren die Top-K=50 via dGeDi→RANSAC.

### 1.0 Die Metriken — was genau gemessen wird

**Tabelle A — offizielle SHREC'18-Metriken**, berechnet mit dem unveränderten offiziellen
`metrics.py` (leaderboard-vergleichbar). Alle laufen über die Top-f, wobei f = Größe der
Query-Kategorie:

| Metrik | Was sie misst |
|---|---|
| **nDCG** | Qualität der ganzen Rangliste mit **abgestuften** Gewinnen (Subkategorie doppelt), rangdiskontiert — die Headline-Metrik |
| precision / recall / F1 | Anteil relevanter Treffer in den Top-f |
| **AP** | Average Precision — belohnt, wenn Relevantes *früh* kommt |
| NNT1 / NNT2 | First / Second Tier |

> **Warum precision = recall = F1 = NNT1 = NNT2 identisch sind:** Die Liste wird *vor* der
> Metrikberechnung auf Top-f gekürzt, und f ist zugleich die Zahl der Relevanten →
> |zurückgegeben| = |relevant| → precision = recall, damit auch F1; NNT2 kann über eine nur f
> Elemente lange Liste nicht bis 2f schauen. Das ist kein Fehler unsererseits, sondern das
> Verhalten des offiziellen Skripts — aber es bedeutet: **diese fünf Spalten tragen genau eine
> Information.** Berichtenswert sind **nDCG** und **AP**.

**Tabelle B — Tiefen-/Top-1-Familie**, bei der Geometrie-Tiefe K abgeschnitten (K = 50 in der
BASE), weil die offiziellen Metriken bei f abschneiden und für die Geometrie-Arme blind wären:

| Metrik | Was sie misst |
|---|---|
| NN_cat | Ist der Top-1-Treffer in der richtigen **Kategorie**? |
| **NN_sub = hit@1** | Ist der Top-1-Treffer in der richtigen **Subkategorie**? ← die durchgehend berichtete Top-1-Metrik |
| MRR / MRR_sub | Kehrwert des Rangs des ersten (Sub-)Kategorie-Treffers, gemittelt |
| mAP_K / nDCG_K | AP bzw. nDCG innerhalb Tiefe K (nDCG_K mit **korrigiertem** DCG) |
| hit_sub@N | mindestens ein Subkategorie-Treffer in den Top-N (N = 1, 5, 10, 20, 50, 100) |

> **Warum hit@1 die Subkategorie nutzt:** Die Kategoriezuordnung („ist es ein Stuhl?“) löst der
> Sprachkanal weitgehend allein; die offene Frage ist, ob das System die richtige **Variante**
> findet (BASE: NN_cat 0.392 vs. NN_sub 0.340). Die Subkategorie ist der beste Stellvertreter,
> den Stage 1 für „geometrisch brauchbar“ hat — mehr nicht: wie weit ein Klassen-Treffer
> geometrisch danebenliegen kann, misst erst Stage 3 (Median-D_sym 1.7 mm exaktes CAD gegen
> 18.4 mm Stellvertreter). NN_sub ist für die Pose **notwendig, nicht hinreichend**.
> **Zwei nDCG, die nie dieselbe Spalte teilen:** `nDCG` (Tabelle A) nutzt das offizielle
> `metrics.dcg` inklusive eines bekannten Off-by-one — unverändert für
> Leaderboard-Vergleichbarkeit; `nDCG_K` (Tabelle B) die korrigierte Formel.

### 1.1 Auf einen Blick

| Konfiguration | nDCG | hit@1 | Bemerkung |
|---|---|---|---|
| CLIP-Text allein | 0.4218 | 0.130 | schwächster Einzelkanal |
| ULIP-2 Shape allein (pc) | 0.5353 | 0.328 | cross-Modus fällt auf 0.4809 |
| DINOv2 View allein | 0.5506 | 0.334 | stärkster Einzelkanal |
| Text + View (OSCAR-Kanäle) | 0.5519 | 0.312 | die OSCAR-Kanalmenge |
| **Volle Fusion (BASE)** | **0.5868** | **0.341** | +0.035 durch S_shape |
| Volle Fusion, Full-Mesh-Referenz | 0.5935 | 0.360 | stärkster Arm *ohne* Geometrie |
| + Geometrie auf der BASE (dGeDi+RANSAC, K=50) | 0.6405 | 0.472 | +0.054 nDCG, +0.130 hit@1 |
| **+ Geometrie auf der Full-Mesh-Referenz** | **0.6417** | **0.481** | **stärkster Arm insgesamt** |
| OSCAR-Kaskade (τ=0.37 → DINO) | 0.4561 | 0.235 | volle Fusion **+0.131** |

`nDCG`/`mAP` messen die ganze Rangliste; `hit@1` und `MRR` die Top-1-Güte — das, was die
Pose-Stufe konsumiert. Beide werden durchgängig berichtet, weil sie **unterschiedliche Sieger**
liefern (→ 1.7).

### 1.2 Vergleich mit dem offiziellen SHREC'18-Track

Offizielle Ergebnisse aus Pham et al., *SHREC'18: RGB-D Object-to-CAD Retrieval*, 3DOR 2018,
Tabelle 3 (DOI 10.2312/3dor.20181052). **Mit welcher Konfiguration wir antreten:** beide
OSCAR+-Zeilen sind dieselbe BASE wie überall sonst, nur auf die 649 Test-Queries eingeschränkt
— kein eigens getunter Lauf (Ergebnisordner `results_shrec18_v2_stage1_42v_k5_testsplit`):

| | |
|---|---|
| Kanäle | CLIP `base` · DINOv2 `base`, 42 Views · ULIP-2 `ulip_pc_rgb` |
| Fusion | weighted_sum, w = (0.3, 0.4, 0.3), `scope=full` (alle 3308 CADs) |
| Appearance-Pooling | mean über Patch-Token, top-k-softmax k=5, τ=0.5 |
| Shape-Referenz | Partial-Views, alle 42 gepoolt (pc-Modus) |
| Geometrie-Zeile | `chamfer_ransac`, K=50, `geom_voxel` 0.02, Inlier 1.5 % des Durchmessers |
| Gallery | 3308 CADs — **nicht** auf den Test-Split reduziert; nur das Query-Set ist angeglichen |

Offizielle Track-Beiträge (bester Beitrag und bester punktbasierter Ansatz; die übrigen sieben
Runs liegen dazwischen):

| Team | Run | Ansatz | Precision | Recall | mAP | NDCG | w-NDCG |
|---|---|---|---|---|---|---|---|
| Tran | view-ring-bow-2 | view-based, bester Beitrag | 0.820 | 0.820 | 0.820 | 0.801 | 0.742 |
| Khoi | pointnet | punktbasiert, bester seiner Klasse | 0.706 | 0.706 | 0.706 | 0.665 | 0.647 |

OSCAR+ auf demselben Test-Split (649 Queries), unverändertes offizielles `metrics.py`:

| Arm | n | nDCG | precision | recall | F1 | AP | NNT1 | NNT2 |
|---|---|---|---|---|---|---|---|---|
| nur CLIP-Text (S_text) | 649 | 0.4102 | 0.1371 | 0.1371 | 0.1371 | 0.0429 | 0.1371 | 0.1371 |
| nur ULIP-2 (S_shape) | 649 | 0.5437 | 0.2383 | 0.2383 | 0.2383 | 0.1428 | 0.2383 | 0.2383 |
| nur DINOv2 (S_view) | 649 | 0.5524 | 0.2572 | 0.2572 | 0.2572 | 0.1508 | 0.2572 | 0.2572 |
| **OSCAR+ (BASE)** | 649 | **0.5945** | 0.2819 | 0.2819 | 0.2819 | **0.1646** | 0.2819 | 0.2819 |
| **OSCAR+ (+Geometrie)** | 649 | **0.6434** | 0.2826 | 0.2826 | 0.2826 | **0.1719** | 0.2826 | 0.2826 |

Die w-Spalten (weighted) berechnen wir nicht — unsere Auswertung nutzt durchgehend die
Standard-Strategie. Auch hier fallen precision = recall = F1 = NNT1 = NNT2 zusammen — in den
Track-Beiträgen ebenso, was die gemeinsame Metrikdefinition bestätigt.

**Der Test-Split ist rekonstruierbar.** Das Paper stellt Beispiel-Rankinglisten nur für
Trainings-Queries bereit; genau diese liegen in `shrec18_full/results/`: **1452 mit Liste
(69.1 % → Training), 649 ohne (30.9 % → Test)** — das reproduziert das im Paper genannte
70/30. Zwei unabhängige Belege, dass es die Trainings-Listen sind und nicht die GT: die Dateien
enthalten nur 5 Einträge mit Score 0.000 („not exhaustive“), und unsere GT stammt aus einer
anderen Quelle (`rgbd.csv` + `cad.csv`, alle 2101/3308 mit echten Labels). Die Unterschiede
zwischen Test-Split und Gesamtdatensatz sind durchweg klein (≤ 0.012) — der Test-Split ist
weder leichter noch schwerer; unsere übrigen Stage-1-Zahlen auf allen 2101 Queries sind dadurch
nicht verzerrt.

**Zwei Gründe, warum die Zahlen dennoch nicht direkt vergleichbar sind:**

1. **Überwacht vs. trainingsfrei.** Alle Track-Teilnehmer („all of them are based on
   supervised deep learning“) rekonstruierten aus den Trainings-Rankinglisten die 20 Kategorien
   und trainierten Klassifikatoren. OSCAR+ sieht keine Kategorielabels und trainiert nichts.
2. **Andere Aufgabenformulierung.** Der Track reformuliert Retrieval als Klassifikation
   („return objects with the same predicted labels“); da die GT *alle* CADs derselben Kategorie
   umfasst, liefert eine korrekte 20-Wege-Klassifikation bei K = |Kategorie| nahezu perfekte
   Precision/Recall — eine grundlegend andere Aufgabe als offenes Retrieval.

**Wie man es berichten sollte:** nicht als „OSCAR+ ist schlechter als der Track“, sondern als
Einordnung zweier Aufgabenstellungen. *Nebenbefund:* der beste punktbasierte Ansatz (PointNet,
NDCG 0.665) liegt 0.136 unter dem besten view-basierten (0.801) — dieselbe Rangfolge der
Modalitäten wie in unserem Block A, dort aber mit deutlich kleinerem Abstand (0.009 statt
0.136). Die Lücke zwischen Bild und Geometrie ist mit modernen Foundation-Encodern erheblich
geschrumpft.

### 1.3 Block A — Kanal-Design (isoliert)

Jede Design-Ablation läuft **isoliert** (ein Kanal, keine Fusion), damit nur die geänderte
Variable wirkt.

**A1 · Appearance-Encoder: DINOv2 vs. SigLIP**

| Arm | nDCG | mAP | hit@1 | MRR |
|---|---|---|---|---|
| **DINOv2** | **0.5506** | 0.1548 | 0.334 | 0.477 |
| SigLIP | 0.5165 | 0.0861 | 0.264 | 0.458 |

DINOv2 gewinnt klar (+0.034 nDCG, hit@1 +0.070; Bilanz 1213:811). **Warum der Vergleich fair
ist:** jedes Modell wird durch den Kopf gelesen, mit dem es trainiert wurde — DINOv2 über das
Mittel der Patch-Token, SigLIP über seinen **MAP-Head** (`pooler_output`), denn SigLIP hat
**kein CLS-Token**. Die naheliegende „einheitliche“ Variante (beide gleich poolen) wäre keine
Gleichbehandlung, sondern ein Fehler: `last_hidden_state[:, 0]` ist bei SigLIP schlicht das
erste Patch-Token — ein beliebiger Bildausschnitt. Genau so lief die Ablation zuerst, und
SigLIP sah dadurch künstlich schwach aus; der berichtete Wert ist der aus dem MAP-Kopf, der
Rückstand von 0.034 also SigLIPs echter.

**A2 · Anzahl Render-Views (Appearance)** — Sättigung ab 16 Views (V16 = 99.5 % von V42):

| Views | nDCG | mAP | hit@1 |
|---|---|---|---|
| 8 | 0.5302 | 0.1317 | 0.303 |
| 16 | 0.5481 | 0.1563 | 0.326 |
| 32 | 0.5426 | 0.1475 | 0.321 |
| **42** | **0.5506** | 0.1548 | **0.334** |

**A3 · Shape-Encoder: ULIP-2 vs. Uni3D** — Gleichstand, kein Sieger:

| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| ULIP-2 | 0.5353 | 0.1386 | **0.328** |
| Uni3D | 0.5337 | 0.1514 | 0.309 |

Isoliert trennen die beiden 0.0016 nDCG bei ausgeglichener Bilanz; auf hit@1 liegt ULIP-2 vorn
(+0.018). Den Ausschlag gibt die Reichweite, nicht die Metrik: **ULIP-2 behalten** — es besitzt
zusätzlich den cross-Modus (Uni3D ist pc-only), und genau der trägt Stage 2 und Stage 3.

**A4 · Shape-Referenz: Partial-Views vs. Full-Mesh (isoliert)**

| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| **Partial-Views (BASE)** | **0.5353** | 0.1386 | **0.328** |
| Full-Mesh | 0.4956 | 0.1376 | 0.282 |

Isoliert gewinnt die partielle Referenz (+0.0397 nDCG, +0.046 hit@1): sie ist geometrisch
vergleichbar mit der partiellen Query — das Full-Mesh sieht die Rückseite, die der Sensor nie
sieht. **Fusioniert kehrt sich das um** (pc-Modus): Full-Mesh 0.5935 / 0.3598 gegen 0.5868 /
0.3413, Bilanz 1127:904 — damit ist `E2b_fullmesh` der stärkste Arm ohne Geometrie. Im
cross-Modus gilt das *nicht* (partial 0.5588 gegen 0.5511). Der Befund dahinter: der partielle
Kanal ist für sich genauer, macht aber Fehler, die mit Text und Erscheinung **korrelieren**;
das vollständige Mesh irrt anders und ergänzt die Fusion deshalb besser.

**A4b · Die vollständige Matrix: Modus × Repräsentation** (nDCG / hit@1)

*Isolierter Shape-Kanal* — partial gewinnt in beiden Modi, pc in beiden Repräsentationen:

| Query-Modus | partial | full-mesh |
|---|---|---|
| **pc** | **0.5353 / 0.328** | 0.4956 / 0.282 |
| cross | 0.4809 / 0.264 | 0.4569 / 0.203 |

*Volle Fusion* — die Repräsentationsachse kippt mit dem Query-Modus:

| Query-Modus | partial | full-mesh | Sieger |
|---|---|---|---|
| pc | 0.5868 / 0.341 | **0.5935 / 0.360** | full-mesh |
| cross | **0.5588 / 0.329** | 0.5511 / 0.308 | partial |

Nur wenn die Query eine Punktwolke ist, ergänzt das vollständige Mesh die Fusion besser; kommt
die Query als Bild, bleibt die partielle Referenz vorn.

**Je Kategorie:** über alle vier Zellen bleibt die Kategorienbilanz nahezu unverändert —
partial gewinnt in 12, 13, 10 bzw. 11 von 20 Klassen —, obwohl der Gesamtwert das Vorzeichen
dreht (isoliert pc +0.040 für partial, fusioniert pc −0.007). **Der Umschwung entsteht nicht
dadurch, dass breit andere Klassen gewinnen, sondern durch die Größe weniger Ausschläge.**
Neun Klassen sind über alle vier Zellen vorzeichenstabil:

| immer besser mit … | Kategorien |
|---|---|
| **partieller** Referenz | book · sofa · bookshelf · pc · oven · pillow |
| **vollständigem Mesh** | desk · light · box |

Die übrigen elf wechseln die Seite — `keyboard` am deutlichsten (isoliert pc +0.362 für
partial, fusioniert pc −0.135). Ein einfaches „welche Referenz für welches Objekt“ lässt sich
nicht ableiten. Auffällig immerhin: unter den drei Full-Mesh-Klassen sind mit `desk` und `box`
zwei, deren Shape-Kanal ohnehin am Boden liegt (0.309 bzw. 0.034 isoliert).

> 📊 **Grafik-Empfehlung G1 — Divergierende Kategorien-Balken (4 Panels):** je Kategorie ein
> horizontaler Balken Δ nDCG (links = full-mesh besser, rechts = partial besser), vier Panels
> für isoliert/fusioniert × pc/cross, ergänzt um den Anteil gewonnener Queries. Daten:
> [stage1/category_fullmesh_vs_partial.csv](stage1/category_fullmesh_vs_partial.csv). Bereits
> interaktiv (4 Tabs) im [Stage-1-Artefakt](artifacts/stage1_results.html), §2 A4b.

**Der Gegensatz zu Stage 3 ist der Befund:** auf BOP ist cross × full-mesh der *beste* Arm
(R@1 0.5151), hier der schwächste der vier fusionierten. Naheliegende Erklärung ist die
Tiefenqualität: SHREC sind saubere Scans, in denen die Punktwolken-Query überlegen ist; BOP
liefert verrauschte Sensortiefe, wo das Bild das verlässlichere Formsignal ist. **Für diese
beiden Datensätze** ist der Query-Modus damit keine freie Designentscheidung, sondern folgt
der Aufnahmesituation — belegt an zwei Punkten; ob wirklich die Tiefenqualität die wirksame
Größe ist (Szenenkomplexität, Objektgröße, Gallery-Zusammensetzung unterscheiden sich
ebenfalls), trennen zwei Datensätze nicht. Handlungsregel ja, allgemeines Gesetz nein.

**A5 · Query-Modus: pc vs. cross** — die Brücke zu Stage 2:

| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| **pc (Query-Punktwolke)** | **0.5353** | 0.1386 | **0.328** |
| cross (Query-Bild) | 0.4809 | 0.0926 | 0.264 |

−0.054 nDCG ohne Tiefe. MI3DOR hat keine Tiefe und muss cross fahren — dieser Arm beziffert
exakt, was das kostet.

**A6 · Query-Farben: XYZ+RGB vs. XYZ-only** — konfundiert, nicht als reine Farb-Ablation
berichten:

| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| XYZ+RGB (BASE) | 0.5353 | 0.1386 | 0.328 |
| **XYZ-only** | **0.5422** | 0.1557 | **0.360** |

Farbe schadet leicht, aber systematisch (XYZ-only gewinnt 1152 von 1999 nicht-gleichen
Queries). **Aber:** der XYZ-Arm tauscht den ganzen ULIP-Turm mit (ViT-B / 512-d / 8k Punkte
statt ViT-g / 1280-d / 10k) — es gibt keinen ViT-g-XYZ-Checkpoint, also keine saubere
Farb-Ablation. → bleibt eine **offene Frage** (Diskussionspunkt 8).

**A7 · Anzahl Shape-Gallery-Views** — monoton steigend, anders als Appearance:

| Views | nDCG | mAP | hit@1 |
|---|---|---|---|
| 8 | 0.5119 | 0.1218 | 0.291 |
| 16 | 0.5227 | 0.1340 | 0.308 |
| 32 | 0.5300 | 0.1314 | 0.317 |
| **42 (BASE)** | **0.5353** | **0.1386** | **0.328** |

### 1.4 Block B — Fusion

**B1 · Weighted-Sum vs. Reciprocal Rank Fusion**

| Arm | nDCG | mAP | hit@1 |
|---|---|---|---|
| **Weighted Sum (BASE)** | **0.5868** | **0.1666** | **0.341** |
| Reciprocal Rank Fusion | 0.5744 | 0.1379 | 0.318 |

Weighted Sum gewinnt klar (+0.0124; Bilanz 1320:718). RRFs Konstante (Cormack k=60) ist auf
TREC-Listenlängen kalibriert — als negatives Ergebnis berichtet, nicht nachtuniert.

**B2 · Gewichts-Sensitivität** *(als einzige Ablation noch bei 16v/k8 gerechnet — die Aussage
ist eine Sensitivitätsaussage und von der Shape-Config unabhängig)*

- **pc-Modus:** Optimum (0.2, 0.4, 0.4) = 0.5916 gegen BASE 0.5889 → +0.003, Rauschen. Die
  BASE-Gewichte sind nicht getunt, aber gut gewählt.
- **cross-Modus** (Brücke zu Stage 2): Optimum verschiebt sich auf (0.3, **0.6**, 0.1) =
  0.5567, und BASE (0.5453) fällt **unter View-only (0.5506)** — ohne Tiefe muss der
  Shape-Kanal heruntergewichtet werden. Die pc-Gewichte übertragen sich **nicht**.

**Die Gewichtskarte** (je 66 Simplex-Punkte, Schrittweite 0.1): im pc-Modus ein **breites
Plateau** — eine ganze Region liegt nahe am Optimum, die Gewichtswahl ist unkritisch. Im
cross-Modus zieht sich das Feld zur **View-Kante** zusammen; das Optimum sitzt dort, wo Shape
fast ausgeschaltet ist. Stage 2 reproduziert die Richtung unabhängig — dort wandert das
Gewicht allerdings zu *Text* (0.45, 0.35, 0.20) statt zu View. **Gemeinsam ist beiden
Datensätzen nur „Shape herunter“**; wohin das Gewicht geht, sagt eine SHREC-Karte für MI3DOR
nicht richtig voraus.

> 📊 **Grafik-Empfehlung G2 — Ternäre Gewichtskarten (pc & cross):** zwei Simplex-Karten
> (Ecken = nur Text / nur View / nur Shape), Punktfarbe = nDCG, Marker für BASE und Optimum.
> Daten: [stage1/weightmap_pc.csv](stage1/weightmap_pc.csv) und
> [stage1/weightmap_cross.csv](stage1/weightmap_cross.csv). Bereits als Canvas im
> [Stage-1-Artefakt](artifacts/stage1_results.html), §3 B2.

**B3 · Kanalbeitrag + OSCAR-Baseline**

| Konfiguration | nDCG | hit@1 |
|---|---|---|
| S_text | 0.4218 | 0.130 |
| S_shape | 0.5353 | 0.328 |
| S_view | 0.5506 | 0.334 |
| S_text + S_view (OSCAR-Kanäle) | 0.5519 | 0.312 |
| **volle Fusion** | **0.5868** | **0.341** |
| OSCAR-Kaskade (τ=0.37 → DINO argmax) | 0.4561 | 0.235 |

**Der Kernbefund von OSCAR+:** S_shape zu OSCARs Text+View hinzuzufügen bringt +0.035 nDCG
und +0.029 hit@1; gegen die *echte* OSCAR-Kaskade beträgt der Vorsprung **+0.131 nDCG**.
**Zur Kaskade:** sie prunt per CLIP-Text-Schwellwert τ=0.37 auf eine Shortlist und arg-maxt
darin über DINOv2 (kein Shape). Auf SHREC'18 prunt der Schwellwert bei **98.3 % der Queries
auf leer** und fällt auf Top-20 zurück — die Kaskade ist faktisch „CLIP-Top-20 → DINO“. Der
Rückstand ist **keine Frage der Parametrierung, sondern der Architektur**: jedes Pruning
verwirft Kandidaten, bevor die anderen Kanäle sie bewerten konnten.

### 1.5 Block C — Geometrie-Reranking

Alle Varianten ordnen die **Top-K = 50** der Fusion um. Backend: dGeDi-Dienst
(GeDi-Deskriptoren + RANSAC, ICP-Verfeinerung im selben Aufruf), identisch zu Stage 3.

**C1 · Geometrie-Signal**

| Arm | Rangkriterium | nDCG | hit@1 | MRR |
|---|---|---|---|---|
| keine (= BASE) | fusionierter Score | 0.5868 | 0.341 | 0.478 |
| `fitness` | RANSAC-**Inlier-Anteil** | 0.6251 | 0.439 | 0.606 |
| **`chamfer_ransac`** | **getrimmte Oberflächendistanz nach Ausrichtung** | **0.6405** | **0.472** | **0.638** |
| `both` (Borda) | Rangfusion Fitness ⊕ Distanz | 0.6362 | 0.465 | 0.626 |

Die ausgerichtete Distanz ist das beste Geometriesignal: nDCG +0.054, vor allem aber
**hit@1 0.341 → 0.472 = +38 % relativ** (MRR 0.478 → 0.638). Da die Pose-Stufe nur den Top-1
konsumiert, ist das die entscheidende Zahl — nDCG unterschätzt den Beitrag um Faktor ~2.5.
Borda verwässert leicht, weil die schwächere Fitness-Stimme gleichberechtigt eingeht.

**Was das Backend liefert — und welche Zweige daraus messbar sind:** je Query-Kandidat-Paar
*ein* Registrierungsversuch (GeDi → Korrespondenzen → RANSAC, `use_icp=True`). Daraus fallen
genau **zwei unabhängige Signale** ab; die Arme sind ihre vier möglichen Verwendungen
(`E2_fitness`, `E2_chamfer_ransac`, `E2_both`, `O1e_gedi_with_base`). Weil Ausrichtung und
Distanz aus einem Aufruf stammen, gibt es **keinen abtrennbaren ICP-Schritt** und keine
„Distanz ohne Ausrichtung“ — ein eigener „ICP-Arm“ oder „unaligned-Arm“ wäre ein
byte-identisches Duplikat von `chamfer_ransac` bzw. `fitness` und wird deshalb nicht
berichtet. Die vier Zeilen oben sind vollständig: alles, was dieses Backend an
Rangkriterien hergibt.

**C2 · Shortlist-Tiefe K** (K=20/5 aus dem K=50-Registrierungs-Cache abgeleitet):

| Rangkriterium innerhalb der Shortlist | K=50 | K=20 | K=5 |
|---|---|---|---|
| **ausgerichtete Distanz** *(Sieger)* | **0.6405** | 0.6279 | 0.6022 |
| Distanz ⊕ Fitness (Borda) | 0.6362 | 0.6240 | 0.6001 |
| Geometrie ⊕ Fusions-Rang (Borda) | 0.6287 | 0.6153 | 0.5979 |
| nur RANSAC-Fitness | 0.6251 | 0.6171 | 0.5980 |
| Geometrie auf Text+View *(ohne Shape-Kanal)* | 0.5961 | 0.5820 | 0.5623 |
| *hit@1 der ausgerichteten Distanz* | *0.472* | *0.464* | *0.426* |

Tiefer ist besser — +0.038 nDCG und +0.046 hit@1 über den Bereich, konsistent über **alle**
Geometrie-Arme: eine Eigenschaft der Shortlist-Tiefe, nicht eines Signals. **Was die Tiefe
kostet** (Stage 4 misst den Schritt direkt: ≈ 0.4 s je Kandidat):

| K | nDCG | hit@1 | Geometriezeit/Query | Einordnung |
|---|---|---|---|---|
| 5 | 0.6022 | 0.426 | ≈ 2 s *(gemessen)* | doppelt so teuer wie die restliche Kette |
| 20 | 0.6279 | 0.464 | ≈ 8 s *(hochgerechnet)* | +0.026 nDCG für 4× Zeit |
| 50 | 0.6405 | 0.472 | ≈ 20 s *(hochgerechnet)* | +0.013 weitere nDCG für 2.5× weitere Zeit |

**Der Ertrag flacht ab, die Kosten nicht:** von K=5 auf K=20 kostet ein nDCG-Punkt rund
230 ms, von K=20 auf K=50 bereits 920 ms. K=50 ist die richtige Wahl für eine
Offline-Retrieval-Auswertung und die falsche für eine antwortende Pipeline — dass Stage 3 K=5
nutzt, ist genau diese Abwägung (bei 12 284 statt 2101 Queries, und dort senkt Geometrie die
Genauigkeit ohnehin).

> 📊 **Grafik-Empfehlung G3 — Ertrag vs. Kosten der Shortlist-Tiefe:** Liniendiagramm nDCG
> (und hit@1) über K ∈ {5, 20, 50} mit zweiter Achse Geometriezeit/Query; macht das Abflachen
> des Ertrags bei linear wachsenden Kosten sichtbar. Daten: Tabelle oben (C2). *Noch in keinem
> Artefakt umgesetzt.*

**C3 · Shape vs. Geometrie — redundant?** Die ersten vier Zeilen bilden ein **gepaartes 2×2**
(Shape-Kanal an/aus × Geometrie an/aus), beide Geometrie-Zeilen mit demselben Kriterium
`fitness`, weil der Text+View-Arm nur in dieser Variante existiert:

| Konfiguration | Geometrie-Kriterium | nDCG | hit@1 |
|---|---|---|---|
| Text+View (weder Shape noch Geometrie) | — | 0.5519 | 0.312 |
| + S_shape in der Fusion (= BASE) | — | 0.5868 | 0.341 |
| + GeDi-Rerank auf Text+View-Shortlist (ohne Shape) | `fitness` | 0.5961 | 0.406 |
| + **beides** (Shape in Fusion, dann GeDi) | `fitness` | 0.6251 | 0.439 |
| *darüber hinaus, mit dem besten Kriterium:* | | | |
| beides, bestes Kriterium | `chamfer_ransac` | 0.6405 | 0.472 |
| beides + Fusionsrang (Borda) | `both_borda_base` | 0.6287 | 0.459 |
| **beides, auf der Full-Mesh-Referenz** | `chamfer_ransac` | **0.6417** | **0.481** |

**Nicht redundant — komplementär.** Im gepaarten 2×2 heben Shape und Geometrie Text+View um
jeweils ~+0.04 und **stapeln sich** (0.5519 → 0.6251). Die Zeilen darunter beantworten eine
andere Frage (*wie weit kommt man insgesamt*) und sind keine vierte Zelle des 2×2. Wichtig:
*alle* Geometrie rerankt eine Shortlist; „GeDi-only“ heißt GeDi auf der Text+View-Shortlist,
nicht Geometrie als eigenständiger Kanal — ein Full-Database-S_GeDi läge bei 0.4 s je
Registrierung und 2101 × 3308 Paaren bei **~770 h pro Zelle**.

**C4 · Geometrie auf dem stärksten Arm**

| Grundlage | ohne Geometrie | mit Geometrie | Gewinn |
|---|---|---|---|
| BASE (pc × partial) | 0.5868 / 0.341 | 0.6405 / 0.472 | +0.0537 / +0.130 |
| **`E2b_fullmesh`** | 0.5935 / 0.360 | **0.6417 / 0.481** | +0.0482 / +0.121 |

Auf dem stärkeren Arm fällt der Gewinn 10 % kleiner aus. Deutlicher am Vorsprung selbst: vor
der Geometrie liegt `E2b_fullmesh` um +0.0067 nDCG vorn, danach nur noch um +0.0012 — das
Re-Ranking **frisst 82 % des Vorsprungs**. Geometrisches Re-Ranking **ersetzt** eine
Sortierung, statt sie zu ergänzen; „+0.054 durch Geometrie“ ist keine Eigenschaft des
Verfahrens, sondern des Paares aus Verfahren und Ausgangsranking.

**Woran man „schwaches Ranking“ misst — die aggregierte Zahl ist es nicht.** SHREC (hit@1
0.341) und BOP (R@1 0.482) liegen nicht weit auseinander, und trotzdem hilft Geometrie hier
und schadet dort. Re-Ranking greift nur *innerhalb* der Shortlist, also zählt die **bedingte
Top-1-Genauigkeit** (Top-1 ÷ Recall@K):

| innerhalb der Shortlist | SHREC'18, K=50 | BOP 3a cross, K=5 |
|---|---|---|
| Fusions-Score (Amtsinhaber) | 0.341 / 0.862 = **0.396** | 0.482 / 0.733 = **0.657** |
| Geometrie (Herausforderer) | 0.472 / 0.862 = **0.548** | 0.423 / 0.733 = **0.577** |
| **Differenz** | **+15.2 Pp. → Geometrie gewinnt** | **−8.0 Pp. → Geometrie verliert** |
| Kopfraum (Recall@K − Top-1) | 0.520 | 0.251 |

Das Kriterium ist **vorab prüfbar**, ohne eine einzige Registrierung zu rechnen: der Kopfraum
ist die Obergrenze dessen, was Re-Ranking gewinnen kann, und beide Zahlen liegen aus dem
Retrieval-Lauf schon vor. Auf SHREC lässt die Fusion 0.520 liegen; auf BOP nur 0.251 — und
sortiert dort, wo das Ziel erreichbar ist, bereits zwei von drei Fällen richtig. *Nicht
vergleichbar* ist die absolute Höhe zwischen den Spalten (SHREC zählt Subkategorie-Treffer,
BOP das exakte Zielobjekt); aussagekräftig ist allein die Differenz innerhalb einer Spalte.

### 1.6 Wie stabil sind die Unterschiede?

Jeder Arm läuft über dieselben 2101 Queries; je Query lässt sich vergleichen, welcher Arm
besser war. Die Bilanz sagt mehr als der Abstand der Mittelwerte:

| Vergleich (nDCG) | Δ | gewonnene Queries | Einordnung |
|---|---|---|---|
| Geometrie: keine vs. GeDi+RANSAC | −0.0537 | 599 : **1264** | Geometrie gewinnt breit |
| DINOv2 vs. SigLIP (isoliert) | +0.0341 | **1213** : 811 | DINOv2 gewinnt breit |
| Weighted-Sum vs. RRF | +0.0124 | **1320** : 718 | Weighted-Sum gewinnt breit |
| Full-Mesh vs. Partial (fusioniert) | +0.0067 | **1127** : 904 | klein, aber durchgängig |
| Partial vs. Full-Mesh (isoliert) | +0.0397 | **1015** : 974 | knappe Bilanz, großer Abstand |
| XYZ+RGB vs. XYZ-only (isoliert) | −0.0068 | 847 : **1152** | Farbe schadet leicht, aber stetig |
| ULIP-2 vs. Uni3D (isoliert) | +0.0016 | ausgeglichen | Gleichstand |

**Wo Mittelwert und Bilanz auseinanderlaufen:** Partial gegen Full-Mesh ist der lehrreiche
Fall — isoliert ein großer Abstand bei fast ausgeglichener Bilanz (partial gewinnt *seltener*,
dann aber deutlich), fusioniert dreht sich das Vorzeichen bei klarer Bilanz. Ein Mittelwert
allein hätte beide Male in die Irre geführt.

### 1.7 Kategorien-Analyse

Per-Kategorie-nDCG der drei isolierten Kanäle. **Die Spezialisierung ist systematisch:**
S_shape gewinnt bei geometrisch markanten, visuell unauffälligen Objekten (keyboard 0.716 vs.
0.090 view; bag 0.384 vs. 0.080), S_view bei texturreichen Möbeln, S_text bei semantisch
trennscharfen Klassen. Totalausfälle sind aufschlussreich: `cup` mit S_shape 0.032 (der Henkel
verschwindet in der Partial-Punktwolke), `bag` mit S_text 0.020 (sprachlich generisch).

> 📊 **Grafik-Empfehlung G4 — Kategorien-Balken (3 Kanäle × 20 Klassen):** gruppierte
> horizontale Balken je Kategorie (text/view/shape), Balkenlänge = nDCG, sortiert nach n.
> Daten: [stage1/category_channels.csv](stage1/category_channels.csv). Bereits im
> [Stage-1-Artefakt](artifacts/stage1_results.html), §6.

**Die feste Gewichtung schadet in 12 von 20 Kategorien.** Alle 20 Kategorien, nach Δ sortiert
(Δ = Fusion − bester Einzelkanal, Metrik nDCG, gepaart über dieselben Queries):

| Kategorie | n | text | view | shape | bester Einzelkanal | Fusion | Δ |
|---|---|---|---|---|---|---|---|
| keyboard | 65 | 0.149 | 0.090 | 0.716 | shape 0.716 | 0.422 | **−0.294** |
| bag | 64 | 0.020 | 0.080 | 0.384 | shape 0.384 | 0.100 | **−0.284** |
| cup | 29 | 0.259 | 0.451 | 0.032 | view 0.451 | 0.338 | −0.113 |
| book | 90 | 0.274 | 0.291 | 0.490 | shape 0.490 | 0.399 | −0.091 |
| table | 140 | 0.523 | 0.482 | 0.392 | text 0.523 | 0.445 | −0.078 |
| bookshelf | 86 | 0.211 | 0.496 | 0.400 | view 0.496 | 0.434 | −0.062 |
| pc | 39 | 0.215 | 0.206 | 0.359 | shape 0.359 | 0.320 | −0.039 |
| desk | 118 | 0.289 | 0.429 | 0.309 | view 0.429 | 0.396 | −0.033 |
| box | 78 | 0.167 | 0.224 | 0.034 | view 0.224 | 0.197 | −0.027 |
| light | 52 | 0.614 | 0.604 | 0.450 | text 0.614 | 0.602 | −0.012 |
| chair | 513 | 0.615 | 0.881 | 0.898 | shape 0.898 | 0.890 | −0.008 |
| storage | 116 | 0.539 | 0.671 | 0.609 | view 0.671 | 0.670 | −0.001 |
| sofa | 139 | 0.164 | 0.473 | 0.226 | view 0.473 | 0.490 | +0.017 |
| display | 192 | 0.669 | 0.603 | 0.628 | text 0.669 | 0.698 | +0.029 |
| pillow | 46 | 0.195 | 0.767 | 0.808 | shape 0.808 | 0.840 | +0.032 |
| oven | 40 | 0.185 | 0.231 | 0.242 | shape 0.242 | 0.282 | +0.040 |
| bin | 133 | 0.438 | 0.371 | 0.337 | text 0.438 | 0.486 | +0.048 |
| bed | 62 | 0.490 | 0.689 | 0.524 | view 0.689 | 0.738 | +0.049 |
| machine | 58 | 0.357 | 0.442 | 0.409 | view 0.442 | 0.537 | +0.095 |
| printer | 41 | 0.233 | 0.382 | 0.282 | view 0.382 | 0.481 | +0.099 |

**Die Verteilung ist schief, und das ist der Punkt:** die acht Gewinne liegen zwischen +0.017
und +0.099, die zwölf Verluste reichen bis −0.294 — die zwei größten Verluste (keyboard, bag)
sind allein dreimal so groß wie der größte Gewinn. Beide sind Klassen, in denen der
Shape-Kanal um ein Vielfaches über den anderen liegt. Umgekehrt gewinnt die Fusion dort, wo
die Kanäle *ähnlich stark* sind (machine, printer): unabhängige Fehler mitteln sich heraus,
statt ein starkes Signal zu verdünnen. → stärkstes Argument für adaptive/gelernte Gewichte.

**Geometrie repariert genau die Shape-dominierten Fälle:**

| Kategorie | Fusion → +Geometrie | Δ |
|---|---|---|
| keyboard | 0.422 → 0.805 | **+0.383** |
| pc | 0.320 → 0.603 | +0.283 |
| cup | 0.338 → 0.605 | +0.267 |
| oven | 0.282 → 0.400 | +0.117 |
| sofa | 0.490 → 0.601 | +0.110 |
| light | 0.602 → 0.597 | −0.005 |
| **pillow** | 0.840 → 0.592 | **−0.248** |

Geometrie holt am meisten, wo die Fusion den Shape-Kanal verwässert hat (keyboard, pc) *und*
wo der Shape-Kanal versagt hat (cup) — die Ausrichtung liefert Evidenz, die kein globales
Embedding hat. **Ausnahme `pillow`:** weiche, deformierbare Objekte haben keine stabile
Starrkörper-Ausrichtung; RANSAC findet Scheinkorrespondenzen. Das ist die inhaltliche Grenze
der Methode.

### 1.8 Alle 40 Arme

Direkt aus `metrics_summary.json` je Arm; `mAP` ist das offizielle `AP`, `hit@1` ist `NN_sub`.
Nach nDCG sortiert. *Nicht aufgeführt (gestrichen): die zwei byte-identischen
Geometrie-Aliasse (`chamfer_icp` = `chamfer_ransac`, `chamfer_unaligned` = `fitness`, → 1.5
C1) und das fusionierte Uni3D.*

| Arm | Gruppe | nDCG | mAP | hit@1 | Bemerkung |
|---|---|---|---|---|---|
| `E2b_fullmesh_geo` | A4 | 0.6417 | 0.1773 | 0.481 | stärkster Arm insgesamt |
| `E2_chamfer_ransac` | C1 | 0.6405 | 0.1737 | 0.472 | |
| `E2_both` | C1 | 0.6362 | 0.1711 | 0.465 | |
| `O1e_gedi_with_base` | C3 | 0.6287 | 0.1726 | 0.459 | |
| `E2_fitness` | C1 | 0.6251 | 0.1680 | 0.439 | |
| `O1c_gedi_post_fusion` | C3 | 0.5961 | 0.1382 | 0.406 | |
| `E2b_fullmesh` | A4 | 0.5935 | 0.1716 | 0.360 | stärkster Arm ohne Geometrie |
| `O5_xyz_only` | A6 | 0.5880 | 0.1713 | 0.354 | |
| `A7f_full_fusion_shape_V42` | A7 | 0.5868 | 0.1666 | 0.341 | = BASE |
| `E1c_full_fusion` | B3 | 0.5868 | 0.1666 | 0.341 | BASE |
| `O4_V16` | A2 | 0.5820 | 0.1661 | 0.333 | |
| `O4_V32` | A2 | 0.5800 | 0.1615 | 0.329 | |
| `E6_rrf` | B1 | 0.5744 | 0.1379 | 0.317 | |
| `O4_V8` | A2 | 0.5714 | 0.1486 | 0.325 | |
| `E4_siglip` | A1 | 0.5659 | 0.1227 | 0.305 | |
| `E7_ulip2_cross` | A5 | 0.5588 | 0.1452 | 0.329 | |
| `O2_visual_first` | — | 0.5570 | 0.1558 | 0.341 | gestrichen |
| `E1b_text_view` | B3 | 0.5519 | 0.1360 | 0.312 | |
| `E7_ulip2_cross_fullmesh` | A5 | 0.5511 | 0.1445 | 0.308 | |
| `A2_view_only_V42` | A2 | 0.5506 | 0.1548 | 0.334 | = E1_view_only |
| `E1_view_only` | A1 | 0.5506 | 0.1548 | 0.334 | = A2_view_only_V42 |
| `A2_view_only_V16` | A2 | 0.5481 | 0.1563 | 0.326 | |
| `A2_view_only_V32` | A2 | 0.5426 | 0.1475 | 0.321 | |
| `O5_xyz_shape_only` | A6 | 0.5422 | 0.1557 | 0.360 | |
| `A7_shape_only_V42` | A7 | 0.5353 | 0.1386 | 0.327 | = E1_shape_only |
| `E1_shape_only` | A4 | 0.5353 | 0.1386 | 0.327 | = A7_shape_only_V42 |
| `E7_uni3d_shape_only` | A3 | 0.5337 | 0.1514 | 0.309 | |
| `A2_view_only_V8` | A2 | 0.5302 | 0.1317 | 0.303 | |
| `A7_shape_only_V32` | A7 | 0.5300 | 0.1314 | 0.317 | |
| `A7_shape_only_V16` | A7 | 0.5227 | 0.1340 | 0.308 | |
| `O2_clip_threshold_cal` | — | 0.5186 | 0.0877 | 0.281 | gestrichen |
| `E4_siglip_only` | A1 | 0.5165 | 0.0861 | 0.264 | |
| `A7_shape_only_V8` | A7 | 0.5119 | 0.1218 | 0.291 | |
| `E2b_fullmesh_shape_only` | A4 | 0.4956 | 0.1339 | 0.282 | |
| `E7_ulip2_cross_shape_only` | A5 | 0.4809 | 0.0926 | 0.264 | |
| `E7_ulip2_cross_fullmesh_shape_only` | A5 | 0.4569 | 0.0907 | 0.203 | |
| `E1d_clip_pruned` | B3 | 0.4566 | 0.0500 | 0.231 | |
| `O2_clip_threshold` | — | 0.4561 | 0.0499 | 0.229 | |
| `E1_oscar_cascade` | B3 | 0.4561 | 0.0501 | 0.235 | |
| `E1a_text_only` | B3 | 0.4218 | 0.0470 | 0.130 | |

### 1.9 Diskussion (Stage 1)

1. **nDCG unterschätzt den Geometrie-Beitrag um Faktor ~2.5.** Geometrie hebt nDCG um +0.054
   (+9 % rel.), hit@1 aber um +0.130 (+38 % rel.). Die Folge ist konkret: beide Metriken küren
   teils verschiedene Sieger — Text+View führt auf nDCG (0.5519 vs. 0.5506 für View allein),
   fällt auf hit@1 aber deutlich zurück (0.312 vs. 0.334).
2. **Der Text-Kanal verbessert die Liste, verschlechtert aber den Top-1.** CLIP-Text ist ein
   *breites* Kategoriesignal — es zieht Relevantes in die Liste, drängt aber Falsches auf
   Rang 1. Für ein Retrieval-Ranking ein Gewinn, für eine Pose-Pipeline ein Verlust.
3. **Ohne den Shape-Kanal sind einzelne Klassen praktisch nicht retrievierbar** (keyboard
   0.716 shape gegen 0.090 view; bag 0.384 gegen 0.080). Der durchschnittliche Fusionsgewinn
   (+0.035) verdeckt, dass S_shape für flache, texturarme Objekte der *einzige* funktionierende
   Kanal ist — das stärkere Argument für OSCAR+ als der Mittelwert.
4. **Feste Gewichte sind ein Kompromiss mit Kosten** — negativ in 12 von 20 Kategorien, bis
   −0.294. Adaptive oder per-Query gewählte Gewichte sind der naheliegende Ausblick.
5. **Der Tiefenverlust trifft den Top-1 doppelt so hart wie die Liste** (pc → cross: nDCG
   −10.2 %, hit@1 −19.5 %). Für Stage 2 und die Pose-Stufe ist der Verzicht auf Tiefe teurer,
   als die nDCG-Zahl vermuten lässt.
6. **Appearance sättigt, Shape nicht.** DINOv2 ab 16 Views flach; ULIP-2 steigt monoton bis
   42. Renderings werden schnell redundant, Partialansichten decken immer neue
   Oberflächenregionen ab — unterschiedliche View-Budgets pro Kanal wären effizienter.
7. **Grenzen der Geometrie:** −0.248 bei `pillow` (deformierbar). Zusammen mit Stage 3:
   Geometrie ist ein *Retrieval*-Werkzeug für starre, scan-saubere Objekte.
8. **Farbe im Query bringt nichts** (−0.0068 nDCG, Bilanz 847:1152) — aber der Vergleich ist
   **konfundiert und bleibt eine offene Frage**: XYZ-only nutzt ein anderes Release
   (`ulip2_pointbert_8k_xyz`, 8192 Punkte, `input_dim=3`, SLIP-ViT-B-Turm), XYZ+RGB das
   farbige `ulip2_pointbert_10k` (10 000 Punkte, 6-Kanal-Eingang, OpenCLIP-ViT-bigG, 1280-d).
   Die saubere Ablation wäre *ein* Encoder mit genullten Farbkanälen — die gibt es nicht.
9. **Der XYZ-Turm kann prinzipiell keine Bild-Query beantworten.** Der cross-Modus vergleicht
   Query-Bild und Gallery-Punktwolke in *einem* Raum, und der ist im Code fest OpenCLIP
   ViT-bigG-14 (1280-d) — der Turm des *farbigen* Releases. Das XYZ-Release lebt in einem
   anderen, kleineren Raum (SLIP ViT-B); die Kosinus-Ähnlichkeit ist nicht einmal
   dimensionsverträglich. Deshalb existiert A6 nur im pc-Modus — und deshalb steht in Stage 2
   und 3 durchgehend der farbige Turm: **ohne ihn gäbe es dort gar keinen Shape-Kanal.**

---

## Stage 2 — MI3DOR: Überlebt die Fusion den Wegfall der Tiefe?

**Setup.** 10 500 monokulare Bild-Queries gegen 3848 CADs in 21 Kategorien (je 500 Queries).
**Warum cross-Modus:** MI3DOR liefert keine Tiefe, es existiert keine Query-Punktwolke — der
Shape-Kanal *muss* das Bild encodieren. Stage-1-A5 beziffert den Wechsel vorab: −0.054 nDCG
bzw. −19.5 % hit@1. Stage 2 prüft, ob die Fusion das trägt.

| Komponente | Wert |
|---|---|
| S_text | CLIP ViT-B/32, Query-Bild gegen Per-View-Beschreibungen, max über Views |
| S_view | DINOv2-base, mean-Pooling, 42 Views, top-k-softmax k=5, τ=0.5 |
| S_shape | ULIP-2 coloured (1280-d), **cross-Modus** (Query-Bild über den OpenCLIP-ViT-bigG-Turm); Gallery-Repräsentation partial bzw. full-mesh — beide gemessen (→ 2.4) |
| Fusion | Weighted Sum, w = (0.3, 0.4, 0.3), volle Datenbank |
| Kaskaden-Arme | CLIP-Schwellwert τ = 0.37, Top-20-Fallback |
| Geometrie | **keine** — MI3DOR hat keine Query-Punktwolken |
| Metrik-Tiefe | top-k = 15, TOP_F = 20 |

**Metriken** (andere Konvention als Stage 1 — keine Subkategorien, Relevanz binär auf
Kategorieebene): **NN** = ist der Top-1 aus der richtigen Kategorie (in %; Pendant zu hit@1) ·
**FT** (First Tier) = Anteil relevanter Treffer in den Top-C, C = Kategoriegröße (31…250, im
Mittel 183) · ST (Top-2C, normiert auf C) · F1 (bei TOP_F=20) · nDCG@2R · mAP · ANMRR (kleiner
ist besser). **Headline: NN und FT** — die beiden trennen systematisch, und genau daran zeigt
sich, was die Kaskade kann und was nicht.

### 2.1 Die Arme

| Arm | NN | FT | ST | F1 | nDCG@2R | mAP | ANMRR ↓ |
|---|---|---|---|---|---|---|---|
| CLIP-Text allein | 67.95 | 0.575 | 0.755 | 0.160 | 0.720 | 0.580 | 0.339 |
| ULIP-2 allein (cross, full-mesh) | 78.10 | 0.510 | 0.649 | 0.188 | 0.652 | 0.518 | 0.409 |
| ULIP-2 allein (cross, partial) | 68.11 | 0.453 | 0.607 | — | 0.598 | 0.451 | 0.467 |
| DINOv2 allein | 83.03 | 0.629 | 0.753 | 0.200 | 0.751 | 0.647 | 0.297 |
| CLIP+DINO+ULIP (volle Fusion, full-mesh) | 86.57 | 0.682 | 0.822 | 0.215 | 0.813 | 0.705 | 0.238 |
| **CLIP+DINO+ULIP (volle Fusion, partial)** | **88.44** | **0.692** | **0.830** | **0.216** | **0.821** | **0.714** | **0.227** |
| OSCAR-Kaskade (Hard-Max) | 84.88 | 0.575 | 0.755 | 0.160 | 0.733 | 0.592 | 0.337 |
| OSCAR-Kaskade (Softmax) | 85.04 | 0.575 | 0.755 | 0.160 | 0.734 | 0.592 | 0.337 |
| CLIP-gepruned + DINO+ULIP | 86.52 | 0.575 | 0.755 | 0.160 | 0.735 | 0.593 | 0.337 |

**Die volle Fusion gewinnt auf jeder Metrik.** Gegenüber dem stärksten Einzelkanal (DINOv2):
Full-Mesh-Fusion +3.5 NN / +0.053 FT / +0.058 mAP / −0.059 ANMRR; die beste Konfiguration
(fusioniert × partial) +5.4 NN / +0.063 FT. Der Shape-Kanal ist im cross-Modus der
**schwächste** Kanal (FT 0.510) — trägt aber trotzdem messbar bei.

### 2.2 Die Anatomie der Kaskade: NN fast auf Fusionsniveau, FT exakt auf CLIP-Niveau

`CLIP-gepruned + DINO+ULIP` erreicht 86.52 NN — 0.05 unter der Full-Mesh-Fusion — und
gleichzeitig FT 0.575, den Wert von CLIP-Text allein auf drei Nachkommastellen. Beides
zugleich ist kein Widerspruch, sondern folgt daraus, *worauf* DINO und ULIP umsortieren. Die
Shortlist S′ entsteht als `{o : sim_text(o) ≥ τ}` mit τ = 0.37, gemessen über alle 10 500
Queries:

| Fall | Queries | Anteil | Größe von S′ |
|---|---|---|---|
| Schwellwert prunt auf **leer** → Top-20-Fallback | 10 174 | **96.9 %** | genau 20 |
| Schwellwert greift | 326 | 3.1 % | Median **2** (153× nur 1; max 113) |

Die Kaskade ist damit faktisch **„CLIP-Top-20 → DINO+ULIP“**; wo τ überhaupt etwas
durchlässt, ist die Liste sogar *kürzer* — meist ein einziger Kandidat.

- **NN misst Rang 1** — der liegt immer in den umsortierten 20. Deshalb springt NN von 67.95
  auf 86.52 (**+18.6**).
- **FT misst die Top-C**, und C ist im Mittel 183 (bis 250). Die Ränge 21…3848 hat nie jemand
  angefasst; sie stehen weiter in CLIP-Reihenfolge → FT fällt exakt auf CLIPs Wert zurück,
  ebenso ST und F1.

**Ein architektonisches Ergebnis, kein Parameterproblem:** jede Kaskade, deren Shortlist
kürzer ist als die Tiefe der Metrik, kann diese Metrik strukturell nicht verbessern. τ zu
senken würde die Shortlist nicht verlängern (sie ist ja schon fast immer der 20er-Fallback);
man müsste den *Fallback* auf ≥ C anheben, womit das Pruning seinen Zweck verlöre. Die volle
Fusion bewertet alle 3848 CADs simultan und schlägt die Kaskade auf FT um **+0.107**.

> 📊 **Grafik-Empfehlung G5 — Shortlist-Schema:** vier horizontale Balken (NN=Rang 1,
> F1=Top-20, FT=Top-183, ST=Top-366), davon jeweils die ersten 20 Positionen eingefärbt
> („von DINO+ULIP umsortiert“), der Rest grau („bleibt in CLIP-Reihenfolge“) — macht auf einen
> Blick sichtbar, warum die Kaskade NN hebt und FT nicht bewegen kann. Bereits im
> [Stage-2-Artefakt](artifacts/stage2_results.html), §1.

### 2.3 Gewichtskarte

231 Gitterpunkte über dem Simplex (Schrittweite 0.05), als Tier-2-Ableitung aus einmalig
gecachten Kanal-Scores. Selbstcheck: FT bei BASE = 0.6851 gegen 0.682 aus dem Produktionslauf
— Abweichung 0.003, der Sweep ist verifiziert. *Der Sweep lief auf der Full-Mesh-Gallery; sein
Optimum (0.6902 FT) liegt damit unter dem neuen BASE auf der partiellen Gallery (0.6918, →
2.4) — die Sensitivitätsaussage bleibt gültig, die Absolutwerte sind mit 2.4 nicht
vergleichbar.*

| Gewichte (text, view, shape) | FT | NN |
|---|---|---|
| **(0.45, 0.35, 0.20)** ← Optimum | **0.6902** | 86.99 |
| (0.40, 0.35, 0.25) | 0.6897 | 86.97 |
| (0.40, 0.40, 0.20) | 0.6894 | 87.08 |
| (0.45, 0.30, 0.25) | 0.6894 | 86.99 |
| (0.30, 0.40, 0.30) ← BASE | 0.6851 | 86.84 |

**Ein Plateau, kein Gipfel:** BASE liegt nur +0.005 FT unter dem Optimum — kein Tuning nötig,
dieselbe Schlussfolgerung wie im pc-Modus auf SHREC (+0.003). **Richtung des Optimums:** ohne
Tiefe gehört der Shape-Kanal herunter (0.20 statt 0.30) — das Gewicht wandert aber **zu Text**
(0.45), nicht zu View. Eine SHREC-cross-Karte hätte das falsch vorhergesagt (dort wandert es
zu View). Gemeinsam ist beiden Datensätzen nur „Shape herunter“ — wohin das Gewicht geht, ist
datensatzspezifisch und muss gemessen werden.

> 📊 **Grafik-Empfehlung G6 — Ternäre Gewichtskarte MI3DOR:** Simplex-Karte wie G2, Farbe =
> FT, Marker für BASE und Optimum; ideal als Doppelabbildung neben der SHREC-cross-Karte, um
> die unterschiedliche Wanderrichtung des Gewichts zu zeigen. Daten:
> [stage2/weight_sweep_mi3dor.csv](stage2/weight_sweep_mi3dor.csv). Bereits im
> [Stage-2-Artefakt](artifacts/stage2_results.html), §2.

### 2.4 Partial-Views vs. Full-Mesh im cross-Modus

Die Gallery-Repräsentation ist die eine Designachse, die MI3DOR und SHREC gemeinsam haben —
und sie fällt hier **isoliert umgekehrt** aus. Beide Läufe config-verifiziert identisch
(42 Views, k=5, mean, cross, τ=0.37, gleicher Checkpoint, n=10 500); es unterscheidet sich nur
die Gallery-Repräsentation:

| ULIP-2 isoliert | partial | full-mesh | Δ |
|---|---|---|---|
| NN | 68.11 | **78.10** | **+9.99** |
| FT | 0.453 | **0.510** | +0.057 |
| ST | 0.607 | **0.649** | +0.042 |
| nDCG@2R | 0.598 | **0.652** | +0.054 |
| mAP | 0.451 | **0.518** | +0.067 |
| ANMRR ↓ | 0.467 | **0.409** | besser |

**Full-Mesh gewinnt isoliert auf jeder Metrik — exakt umgekehrt zu SHREC** (dort partial
+0.0397 isoliert). Kein Widerspruch, sondern der verwertbare Teil: SHREC fragt mit einer
*partiellen Punktwolke* → eine partielle Referenz ist geometrisch vergleichbar; MI3DOR fragt
mit einem *Bild* → das Bild zeigt das vollständige Objekt, also passt die Full-Mesh-Referenz
besser. Stage 1 bestätigt die Drehung unabhängig: im cross-Modus schrumpft der
Partial-Vorsprung dort von 0.040 auf 0.024.

**Fusioniert kehrt sich das um** (nachgerechnet am 2026-09-07; einzige Variable ist die
Gallery-Repräsentation des Shape-Kanals):

| volle Fusion | partial | full-mesh | Δ |
|---|---|---|---|
| NN | **88.44** | 86.57 | **+1.87** |
| FT | **0.6918** | 0.6818 | +0.0100 |
| ST | **0.830** | 0.822 | +0.008 |
| nDCG@2R | **0.821** | 0.813 | +0.008 |
| mAP | **0.714** | 0.705 | +0.009 |
| ANMRR ↓ | **0.227** | 0.238 | besser |

**Der schwächere Kanal trägt mehr bei.** Full-mesh ist isoliert um 9.99 NN besser und
fusioniert um 1.87 NN schlechter. Das Vorzeichen dreht nicht, weil der partielle Kanal genauer
wäre — er ist es messbar nicht —, sondern weil seine Fehler mit denen von Text und Erscheinung
**weniger korrelieren**. Ein Kanal, der dasselbe falsch macht wie DINOv2, fügt der Fusion
nichts hinzu, auch wenn er für sich stärker ist. Damit reproduziert MI3DOR das Stage-1-Muster
exakt (cross fusioniert: partial 0.5588 gegen 0.5511). Fusioniert stimmen beide
cross-Datensätze überein; isoliert widersprechen sie sich — was daran liegen dürfte, dass
MI3DORs Partialwolken aus *gerenderten* Ansichten stammen und SHRECs aus echten Scans. **Die
frühere Schlussfolgerung „im cross-Modus ist Full-Mesh die überlegene Wahl“ galt nur für den
isolierten Kanal; die beste Stage-2-Konfiguration ist cross × partial: NN 88.44 / FT 0.6918.**

**Wie dieser Lauf abgesichert ist:** es gibt auf der Maschine keine MI3DOR-`*_partial.npz`
mehr, nur einen 791-MB-Cache; ohne `SHREC_FORCE_PARTIAL_CACHE` (nicht SHREC-spezifisch) fällt
der Pass *still* auf Full-Mesh zurück. Drei Prüfungen: der Cache wurde nachweislich erzwungen
(3848 Modelle), der Config-Block sagt `ulip2_use_partial_views=True`, und — die stärkste — der
isolierte Arm trifft einen Lauf vom 2026-08-07, der noch echte `*_partial.npz` las, auf allen
fünf Metriken **exakt** (größte Abweichung 0.0e+00).

**Die MI3DOR-Meshes tragen keine Farbe.** Nachgeprüft: alle 3848 Meshes liefern beim Sampling
dieselbe Farbe (0.4, 0.4, 0.4) — trimeshs Standardgrau; die Dateien enthalten weder Vertex-
noch Flächenfarben noch Textur. Zum Vergleich (Standardabweichung der gesampelten Farbe):

| Datensatz | Mesh-Farbe |
|---|---|
| GSO, YCB-V, SHREC'18 | Textur vorhanden (0.10–0.28) |
| LM-O | Vertexfarben, 0.04–0.18 |
| **MI3DOR**, T-LESS, ITODD, HouseCat6D | **0.0 — keine Farbe in den Dateien** |

**Ein plausibler zweiter Grund für den schwachen Shape-Kanal:** der verwendete
ULIP-2-Backbone ist der *farbige* (`pointbert_colored`, 6-Kanal-Eingang) — und bekommt über
die gesamte Gallery ein konstantes, informationsloses RGB. Das verzerrt die Gallery nicht
untereinander, setzt aber jedes Gallery-Embedding in einen Bereich des Merkmalsraums, für den
der Encoder nicht trainiert wurde — während die Query durch den Bildturm kommt, der Farbe
sehr wohl sieht. Dass ULIP-2 hier der schwächste Kanal ist (FT 0.510 gegen DINOv2 0.629),
wurde bisher allein der cross-modalen Schwierigkeit zugeschrieben; **die farblose Gallery ist
ein zweiter, ebenso plausibler Grund** — belegt ist das nicht (prüfbar nur über einen
kompletten Stage-2-Neulauf mit Partialwolken aus den Renderings). Für die Schlussfolgerung des
Kapitels ändert das nichts, es verschiebt die Begründung: dass Shape ohne Tiefe
heruntergewichtet gehört, gilt weiterhin.

### 2.5 OSCAR-Legacy-Vergleich (V = 8)

Die publizierte OSCAR-Kaskade wird mit 8 Views beschrieben. Derselbe Mechanismus, unser
Evaluator, unsere Gallery — nur der View-Count wechselt:

| Arm | NN (V=8) | NN (V=42) | FT (V=8) | FT (V=42) |
|---|---|---|---|---|
| DINOv2 allein | 81.96 | 83.03 | 0.591 | 0.629 |
| OSCAR-Kaskade (Hard-Max) | 84.40 | 84.88 | 0.575 | 0.575 |
| volle Fusion (full-mesh) | **86.62** | 86.57 | 0.665 | 0.682 |

**Der View-Count ist auf MI3DOR fast wirkungslos** (volle Fusion 86.62 vs. 86.57) — OSCARs
8-View-Konfiguration war kein Handicap, anders als auf SHREC, wo mehr Views durchgehend
halfen. **Warum wir die Zahlen nicht direkt gegen die Publikation stellen:** Pullis Evaluator
wendet die CLIP-Shortlist nicht auf das Ranking an (`keep` fließt nirgends in `sims_full`);
die publizierten Zahlen sind reines DINOv2-Retrieval über eine andere Gallery (1817 Objekte ×
1 View statt 3848 × 42). Wir reproduzieren den **Mechanismus**, nicht ihre Messung.

### 2.6 Kategorien-Analyse

Per-Kategorie-NN (Anteil korrekter Top-1), alle 21 Kategorien mit je 500 Queries.
*Fusionsspalte = volle Fusion auf der Full-Mesh-Gallery (`clip_dino_ulip_full`, NN 86.57);
der fusionierte Partial-Lauf kam später hinzu und ändert das qualitative Bild nicht.*

**S_shape ist in 10 von 21 Kategorien der beste Einzelkanal** — teils dramatisch: `camera`
0.990 gegen 0.346 (text), `motorcycle` 0.974 gegen 0.868 (view), `vase` 0.416 gegen 0.182
(view). Selbst im cross-Modus, wo er insgesamt der schwächste Kanal ist, trägt er bei knapp
der Hälfte der Klassen am meisten. **S_view gewinnt ebenfalls in 10** (bed, bicycle, plant,
keyboard, rifle, wardrobe …). **S_text gewinnt genau einmal:** `monitor` (0.666) — und dort
bricht der Shape-Kanal auf 0.124 ein: ein flaches Rechteck ist ohne Tiefe geometrisch kaum
bestimmbar, sprachlich aber eindeutig.

**Die feste Gewichtung schadet in 9 von 21 Kategorien.** Alle 21, nach Δ sortiert (Δ = Fusion
− bester Einzelkanal, Metrik NN, gepaart über dieselben 500 Queries je Klasse):

| Kategorie | text | view | shape | bester Einzelkanal | Fusion | Δ |
|---|---|---|---|---|---|---|
| vase | 0.166 | 0.182 | 0.416 | shape 0.416 | 0.284 | **−0.132** |
| bookshelf | 0.614 | 0.860 | 0.540 | view 0.860 | 0.788 | −0.072 |
| pistol | 0.360 | 0.736 | 0.484 | view 0.736 | 0.670 | −0.066 |
| monitor | 0.666 | 0.584 | 0.124 | text 0.666 | 0.602 | −0.064 |
| bicycle | 0.696 | 0.962 | 0.804 | view 0.962 | 0.916 | −0.046 |
| camera | 0.346 | 0.928 | 0.990 | shape 0.990 | 0.974 | −0.016 |
| motorcycle | 0.768 | 0.868 | 0.974 | shape 0.974 | 0.964 | −0.010 |
| airplane | 0.988 | 0.988 | 1.000 | shape 1.000 | 0.992 | −0.008 |
| bed | 0.892 | 0.978 | 0.858 | view 0.978 | 0.976 | −0.002 |
| car | 0.976 | 1.000 | 1.000 | view 1.000 | 1.000 | ±0.000 |
| guitar | 0.986 | 1.000 | 0.998 | view 1.000 | 1.000 | ±0.000 |
| wardrobe | 0.502 | 0.906 | 0.890 | view 0.906 | 0.906 | ±0.000 |
| tent | 0.880 | 0.772 | 0.966 | shape 0.966 | 0.972 | +0.006 |
| keyboard | 0.728 | 0.922 | 0.832 | view 0.922 | 0.942 | +0.020 |
| plant | 0.906 | 0.968 | 0.640 | view 0.968 | 0.990 | +0.022 |
| flower_pot | 0.570 | 0.772 | 0.814 | shape 0.814 | 0.842 | +0.028 |
| rifle | 0.860 | 0.910 | 0.878 | view 0.910 | 0.950 | +0.040 |
| chair | 0.726 | 0.884 | 0.886 | shape 0.886 | 0.926 | +0.040 |
| knife | 0.630 | 0.806 | 0.846 | shape 0.846 | 0.896 | +0.050 |
| stairs | 0.494 | 0.774 | 0.778 | shape 0.778 | 0.834 | +0.056 |
| radio | 0.516 | 0.636 | 0.684 | shape 0.684 | 0.756 | +0.072 |

**Derselbe Befund wie in Stage 1 — über zwei Datensätze und beide Query-Modi.** Am
deutlichsten bei `vase`: Shape allein 0.416, Fusion 0.284 — Text (0.166) und View (0.182)
sind dort fast blind und ziehen den einen funktionierenden Kanal herunter. Umgekehrt gewinnt
die Fusion am meisten, wo die Kanäle ähnlich stark sind (`radio` 0.516/0.636/0.684 → 0.756,
`stairs` 0.494/0.774/0.778 → 0.834). In Stage 1 sind es 12 von 20 Kategorien, hier 9 von 21.

> 📊 **Grafik-Empfehlung G7 — Kategorien-Balken MI3DOR (3 Kanäle + Fusion × 21 Klassen):**
> wie G4, zusätzlich ein vierter (dickerer) Balken für die Fusion — zeigt sowohl die
> Spezialisierung als auch, wo die Fusion unter den besten Einzelkanal fällt. Daten:
> [stage2/category_table.csv](stage2/category_table.csv). Bereits im
> [Stage-2-Artefakt](artifacts/stage2_results.html), §5.

### 2.7 Diskussion (Stage 2)

1. **Der Shape-Kanal ist im cross-Modus der schwächste — und trotzdem unverzichtbar.**
   Isoliert FT 0.510 gegen DINOv2 0.629, aber er hebt die Fusion messbar und ist bei 10 von
   21 Kategorien der beste Einzelkanal. Der Mittelwert verdeckt, wo er trägt.
2. **Die Kaskade kann Listen-Metriken strukturell nicht verbessern.** FT/ST/F1 aller
   Kaskaden-Arme sind identisch mit CLIP-Text allein, weil die 20er-Shortlist die Top-C-Tiefe
   (im Mittel 183) nicht füllen kann. Sie verbessert nur den Kopf (NN +18.6). Ein
   architektonisches Argument, kein Parameterproblem.
3. **τ = 0.37 greift praktisch nie:** 96.9 % der Queries laufen in den Top-20-Fallback
   (SHREC: 98.3 %). Wo der Schwellwert greift, ist die Shortlist mit Median 2 zu kurz zum
   Umsortieren.
4. **Die partial-vs-full-mesh-Antwort kippt zwischen isoliert und fusioniert:** full-mesh für
   sich +9.99 NN, fusioniert −1.87 NN. Nicht die Genauigkeit eines Kanals entscheidet über
   seinen Beitrag, sondern die **Korrelation seiner Fehler** mit den übrigen. Fusioniert
   stimmen MI3DOR und SHREC im cross-Modus überein, isoliert nicht.
5. **Der View-Count ist auf MI3DOR fast wirkungslos** (V8 ≈ V42), anders als auf SHREC.
   Plausibel: bei Bild-Queries gegen gerenderte Ansichten reichen wenige Blickwinkel.
6. **Feste Gewichte kosten auch hier** — negativ in 9 von 21 Kategorien, bis −0.132. Zusammen
   mit Stage 1 (12 von 20) ein über zwei Datensätze reproduzierter Befund.
7. **BASE-Gewichte sind robust** (+0.005 zum Optimum), aber das Optimum verschiebt sich
   **text-lastig** (0.45/0.35/0.20). Wer im cross-Modus tunen wollte, müsste Text stärken —
   *nicht* View, wie ein SHREC-Proxy nahegelegt hätte.
8. **Die farblose Gallery ist eine offene Variable.** Ob der schwache Shape-Kanal an der
   fehlenden Tiefe, der fehlenden Farbe oder an beidem liegt, trennt dieser Datensatz nicht.

---

## Stage 3 — BOP: Taugt das gefundene CAD für die Pose?

**Setup.** 12 284 Instanzen aus YCB-V (4123), T-LESS (6716) und LM-O (1445); Gallery 1316
Objekte = 1257 Proxy-CADs (GSO 1030, HouseCat6D 199, ITODD 28) + 59 Ziel-CADs. Relevanz ist
jetzt die **Instanz**: relevant ist genau das exakte Zielmodell. Segmentierung über
GT-`mask_visib` und GT-Bounding-Box, damit Retrieval und Pose nicht von Segmentierungsfehlern
überlagert werden. Fusionsgewichte, Views, k, τ identisch zu Stage 1/2; Geometrie-Shortlist
K=5. Drei Fragen: **3a** — findet die Pipeline das exakte CAD? **3b** — was kostet ein
Ersatzmodell in Millimetern? **3c** — woher kommt dieser Fehler? Dazu **3d** — was macht
Verdeckung?

### 3a — Retrieval: findet die Pipeline das richtige CAD?

Gallery enthält das exakte Zielmodell; gemessen werden Recall@K und MRR. Die drei rechten
Spalten sind R@1 je Datensatz; „Shape allein“ ist der isolierte ULIP-2-Kanal:

| Arm | R@1 | R@5 | R@10 | MRR | Shape allein | YCB-V | T-LESS | LM-O |
|---|---|---|---|---|---|---|---|---|
| **Full-Mesh-Gallery, cross-modal** *(bester Arm)* | **0.5151** | **0.7881** | **0.8505** | **0.6379** | 0.2272 | 0.726 | **0.394** | **0.478** |
| Volle Fusion, cross-modal *(eingefroren)* | 0.4818 | 0.7330 | 0.8120 | 0.5971 | 0.1997 | **0.732** | 0.332 | 0.464 |
| Volle Fusion, Punktwolken-Query | 0.4636 | 0.7258 | 0.8080 | 0.5844 | 0.0211 | 0.671 | 0.350 | 0.400 |
| Full-Mesh-Gallery, Punktwolken-Query | 0.3878 | 0.6022 | 0.7160 | 0.4899 | 0.0089 | **0.740** | 0.159 | 0.446 |
| OSCAR-Baseline *(ohne Shape)* | 0.3198 | 0.4923 | 0.5418 | 0.4043 | — | 0.498 | 0.214 | 0.304 |

**Der Shape-Kanal trägt +0.162 Recall@1:** gegen die faithful nachgebaute OSCAR-Kaskade
(0.3198) gewinnt die volle Fusion 16.2 Punkte — auf allen drei Datensätzen, relativ am
größten auf T-LESS (0.214 → 0.332), also dort, wo Textur und Sprache am wenigsten hergeben.
„Shape allein“ zeigt zugleich, wie wenig die Form für sich trägt (2 % im pc-Modus) und wie
viel im cross-Modus, wo ULIP-2s Bildturm arbeitet (20 %).

**Repräsentation der Gallery je Datensatz** (R@1; Anteil an den Instanzen: YCB-V 34 %,
T-LESS 55 %, LM-O 12 %):

| Query · Datensatz | partial | full-mesh | Δ |
|---|---|---|---|
| *Punktwolken-Query* | | | |
| YCB-V | 0.671 | **0.740** | +0.069 |
| T-LESS | **0.350** | 0.159 | −0.191 |
| LM-O | 0.400 | **0.446** | +0.046 |
| *Cross-modale Query* | | | |
| YCB-V | **0.732** | 0.726 | −0.006 |
| T-LESS | 0.332 | **0.394** | +0.062 |
| LM-O | 0.464 | **0.478** | +0.014 |

Gewinnbilanz je Instanz: cross 884:1294 zugunsten Full-Mesh, pc 1742:811 zugunsten partial.

- **Im cross-Modus gewinnt Full-Mesh — überall.** cross × full-mesh (0.5151) ist der
  stärkste Retrieval-Arm der Arbeit.
- **Im pc-Modus bleibt partial vorn — aber allein wegen T-LESS.** Auf YCB-V (+0.069) und
  LM-O (+0.046) gewinnt auch dort das vollständige Mesh. „Partial schlägt Full-Mesh“ ist
  keine Aussage über die Repräsentation, sondern über texturlose, symmetrische Industrieteile.
- **Warum der pc-Modus anders reagiert — Domänenabgleich:** dieselbe Gallery wirkt
  gegensätzlich (cross +0.033, pc −0.076); eine Eigenschaft der Gallery allein könnte das
  nicht erklären. Im pc-Modus ist die Query eine partielle Wolke und die Partial-Gallery
  besteht aus genau solchen — gleiche Domäne, 42 Chancen je Objekt, den Blickwinkel zu
  treffen. Im cross-Modus läuft die Query durch den Bildturm; dort ist keine Variante
  domänengleich, und der Abstand fällt auf 0.018.
- **Full-Mesh ist informativ, aber auf Rang 1 unbrauchbar** (T-LESS, pc-Modus):
  Ziel-Rangverteilung partial Median 2 (6 % jenseits Rang 50), full-mesh Median 8 (14 %),
  *ohne* Shape-Kanal Median 24 (28 %). Der Kanal versagt an der Spitze der Liste — genau dem,
  was Recall@1 misst.

> 📊 **Grafik-Empfehlung G8 — R@1 je Arm und Datensatz:** Punkt- oder Balkendiagramm mit drei
> Datensatz-Facetten und den fünf Armen, das die Δ-Spalten der beiden Tabellen oben sichtbar
> macht (insbesondere den T-LESS-Ausreißer im pc-Modus). Daten:
> [stage3/](stage3/) (`combined_stage3a.json` je Lauf). *Noch in keinem Artefakt als Grafik
> umgesetzt.*

**Geometrisches Re-Ranking** (dGeDi + RANSAC über die Top-5; *Distanz* =
Registrierungsdistanz nach Ausrichtung, *Fitness* = Inlier-Anteil):

| Query · Signal | R@1 | Δ R@1 | MRR | R@5 | R@10 | Deckung |
|---|---|---|---|---|---|---|
| *cross* — ohne Geometrie | **0.4818** | — | 0.5971 | 0.7330 | 0.8120 | — |
| *cross* — Registrierungsdistanz | 0.4229 | −0.059 | 0.5576 | 0.7330 | 0.8120 | 98 % |
| *cross* — Fitness | 0.4278 | −0.054 | 0.5569 | 0.7330 | 0.8120 | 98 % |
| *pc* — ohne Geometrie | **0.4636** | — | 0.5844 | 0.7258 | 0.8080 | — |
| *pc* — Registrierungsdistanz | 0.3725 | −0.091 | 0.5215 | 0.7258 | 0.8080 | 98 % |
| *pc* — Fitness | 0.3820 | −0.082 | 0.5249 | 0.7258 | 0.8080 | 98 % |

R@5/R@10 sind über alle Zeilen identisch — das Re-Ranking ordnet nur innerhalb der Top-5 um
(derselbe Effekt wie in Stage 1, wo fünf der sieben offiziellen Metriken für die Geometrie
blind waren). **Geometrie verliert in allen vier Zellen — und das ist kein
Implementierungsfehler:** bei 98 % Registrierungsdeckung greift sie nachweislich und setzt
das richtige CAD in 58 % der Fälle auf Rang 1 (gegen 20 % Zufall bei fünf Kandidaten) — nur
schafft der Fusions-Score, den sie *ersetzt*, innerhalb derselben Shortlist 66 %. Re-Ranking
ist eine Verdrängung, keine Ergänzung. **Fitness schlägt Distanz auf BOP in beiden Modi** —
genau umgekehrt zu Stage 1: SHREC'18 ist skaleninvariant (Ausrichtungsdistanz
aussagekräftig), auf BOP mit echter metrischer Skala gewinnt das Überlappungsmaß.

### 3b — Pose: reicht ein Ersatzmodell?

Die exakten Zielmodelle sind aus der Gallery entfernt (1257 Einträge); das Top-1 ist
zwangsläufig ein Proxy, FoundationPose schätzt damit die Pose. Gemessen wird **D_sym**, die
symmetrische Oberflächendistanz zwischen GT-posiertem Zielobjekt und geschätzt-posiertem
Proxy. Die drei rechten Spalten sind D_sym-Mediane in mm je Datensatz:

| Arm | Ergebnisordner | D_sym Median | / Durchmesser | Δ zur GT-Pose | YCB-V | T-LESS | LM-O |
|---|---|---|---|---|---|---|---|
| GT-CAD *(Referenz)* | `gt` | **1.72 mm** | **0.015** | 0 *(Definition)* | 2.03 | 1.41 | 3.48 |
| **OSCAR+ Proxy, cross × partial** | `3b_cross` | **18.37 mm** | 0.139 | **15.79 mm** | 23.59 | **13.57** | **28.54** |
| OSCAR+ Proxy, cross × full-mesh | `3b_cross_fullmesh` | 18.91 mm | 0.152 | 16.63 mm | **21.78** | 14.98 | 34.04 |
| OSCAR-Baseline Proxy | `3b_oscar` | 21.73 mm | 0.173 | 18.86 mm | 24.38 | 18.43 | 31.22 |
| OSCAR+ Proxy + Geometrie | `3b_cross_geo` | 28.79 mm | 0.253 | 26.07 mm | 21.58 | 29.10 | 45.90 |

Deckung 100 %, außer full-mesh 99.98 % (3 Ausfälle) und OSCAR-Baseline 99.96 % (5 von 12 284).

**Was in der Tabelle steht:** „**/ Durchmesser**“ ist derselbe Fehler, geteilt durch den
Durchmesser des Zielobjekts (aus `models_info.json`). Ohne diese Normierung sind die
Datensätze nicht vergleichbar: in mm sieht T-LESS mit 13.57 am besten aus — relativ zum
Durchmesser (0.137) liegt es praktisch gleichauf mit YCB-V (0.134). Die absolute Spalte misst
zu einem guten Teil Objektgröße, die relative den Substitutionsfehler. „**Δ zur GT-Pose**“
ist der gepaarte Abstand je Instanz zum GT-CAD-Lauf — der Anteil des Fehlers, der auf das
*Ersatzmodell* zurückgeht und nicht auf FoundationPose; er ist kleiner als die Differenz der
Mediane, weil er je Instanz gebildet wird.

**Konfiguration der Arme** (allen OSCAR+-Armen gemeinsam: Fusion (0.3, 0.4, 0.3) über die
volle Gallery, DINOv2 mean über 42 Views, k=5, Pose durch FoundationPose mit GT-Maske und
GT-Box; die einzige Variable ist die jeweils genannte):

| Arm | Aufruf (`eval_bop_pose.py`) | Gallery | Shape-Kanal |
|---|---|---|---|
| GT-CAD | `--mode gt` | — | — |
| OSCAR+ cross × partial | `--mode 3b` | 1257 Proxy-CADs | ULIP-2 cross, partielle Referenz |
| OSCAR+ cross × full-mesh | `--mode 3b --fullmesh` | 1257 | ULIP-2 cross, Full-Mesh-Referenz |
| OSCAR-Baseline | `--mode 3b --oscar-baseline` | 1257 | **keiner** — CLIP-Schwelle τ=0.37, dann DINOv2-Best-View |
| + Geometrie | `--mode 3b` + dGeDi-Rerank | 1257 | wie cross × partial, danach Re-Ranking der Top-5 |

**Der Retrieval-Vorsprung überträgt sich nur bedingt.** Gegen die OSCAR-Baseline ja: 18.37
gegen 21.73 mm, 3.4 mm besser auf allen drei Datensätzen — die +0.162 Recall@1 schlagen bis
zur einsetzbaren Pose durch. Zwischen den beiden OSCAR+-Armen aber nicht: **Full-Mesh findet
häufiger das richtige CAD (+0.033 R@1) und posiert es schlechter** (18.91 gegen 18.37 mm).
Retrieval belohnt das Modell, das als Ganzes passt; FoundationPose registriert gegen die
*sichtbare* Oberfläche. Die Geometrie schadet der Pose noch deutlicher als dem Retrieval:
**+57 %** Fehler gegenüber demselben Arm ohne Re-Ranking — die Geometrie-Achse ist auf BOP an
keiner Stelle der Kette von Nutzen.

### 3c — Zerlegung: woher kommen die 15 mm?

Aus der Gallery wird nur das exakte GT-Modell entfernt, alles andere bleibt. Damit lässt sich
trennen, *woher* der beste verbleibende Ersatz stammt: aus dem **eigenen BOP-Datensatz** (ein
anderes Objekt derselben Szenensammlung) oder aus der **fremden Proxy-Gallery**
(GSO/HouseCat6D/ITODD). Beides sind CAD-Modelle — der Unterschied ist die Herkunft:

| Fall | partial: n | Median | full-mesh: n | Median |
|---|---|---|---|---|
| Gesamt | 12 284 | 15.34 mm | 12 284 | **13.51 mm** |
| Ersatz aus dem **eigenen** BOP-Datensatz *(anderes Objekt)* | 6 742 *(55 %)* | 10.35 mm | 7 999 *(65 %)* | **9.76 mm** |
| Ersatz aus der **fremden** Proxy-Gallery | 5 542 *(45 %)* | **20.10 mm** | 4 285 *(35 %)* | 22.65 mm |

**Die Zerlegung löst den Widerspruch zwischen 3b und 3c auf.** Ein CAD aus dem eigenen
Datensatz kostet 10.35 mm, eines aus der fremden Gallery 20.10 mm — fast das Doppelte. Was
sie trennt, ist die Nähe zur Domäne: BOP-Datensätze sind thematisch geschlossen, ein
zufälliges GSO-Objekt nicht. **Die Zusammensetzung der Gallery ist damit ein eigener Hebel,
unabhängig vom Retrieval.** Und sie erklärt, warum Full-Mesh in 3c gewinnt (13.51 gegen
15.34 mm), in 3b aber verliert: es wählt aus dem eigenen Datensatz *besser* (9.76 gegen
10.35) und aus der fremden Gallery *schlechter* (22.65 gegen 20.10); sein stärkeres Retrieval
schickt 65 statt 55 % der Instanzen in die gute Teilmenge — daher der Sieg in 3c. In 3b sind
alle Ziel-CADs entfernt, es bleibt nur die Teilmenge, in der es schwächer ist. Beide
Ergebnisse folgen aus *einer* Eigenschaft. — T-LESS erreicht mit 8.75 mm den besten Median,
obwohl es das schwächste Retrieval hat (R@1 0.332): die Objekte sind einander geometrisch so
ähnlich, dass ein *falsch* gewähltes CAD trotzdem eine gute Pose stützt.
**Retrieval-Genauigkeit und Pose-Nutzen sind nicht dasselbe.**

### 3d — Verdeckung: der stärkste Faktor, den wir nicht entworfen haben

Alle bisherigen Abschnitte vergleichen Entwurfsentscheidungen; dieser fragt, *wie viel man
vom Objekt überhaupt sehen muss*. **Wie die Zahlen entstehen** — nichts wurde neu gerechnet,
es ist eine Re-Analyse vorhandener Dateien in vier Schritten:

1. **BOPs eigene Annotation:** jede Testszene hat `scene_gt_info.json` mit `visib_fract` je
   Instanz, definiert als `px_count_visib / px_count_valid` — der Anteil der projizierten
   Objektfläche, der im Bild sichtbar ist. An einer Instanz nachgerechnet: 3362/3478 =
   0.9666, exakt der annotierte Wert.
2. **Verknüpfung** unserer Per-Instanz-Records mit dieser Annotation über den Schlüssel
   `(scene_id, im_id, gt_idx)`: **12 284 von 12 284** Instanzen zugeordnet, keine verloren.
3. **Eigene Metrik statt der fertigen Zahl:** R@1 wurde aus `target_rank == 1` selbst
   gebildet und gegen den publizierten Wert geprüft — 5918/12 284 = 0.481765 gegen 0.481765
   aus `combined_stage3a.json`. Kein Record hat `target_rank = None`.
4. **Schichtung** in vier Sichtbarkeitsklassen.

| Sichtbarkeit | n | Anteil | 3a: R@1 | 3b: D_sym |
|---|---|---|---|---|
| stark verdeckt < 50 % | 1 184 | 9.6 % | 0.098 | 27.7 mm |
| teilverdeckt 50–80 % | 2 394 | 19.5 % | 0.300 | 18.8 mm |
| leicht 80–95 % | 2 992 | 24.4 % | 0.502 | **14.4 mm** |
| frei > 95 % | 5 714 | 46.5 % | **0.627** | 18.8 mm |

Pearson r über alle Instanzen: +0.358 (3a) und −0.275 (3b). Reproduzierbar mit
`python3 tools/occlusion_analysis.py` — das Skript bricht ab, wenn die selbst gerechnete R@1
nicht zur publizierten passt.

**Wie viel man sieht, schlägt jede Entwurfsentscheidung:** zwischen frei sichtbar und stark
verdeckt liegen 0.53 R@1; die größte gemessene Designachse (Full-Mesh gegen Partial in 3a)
bewegt 0.033. Bei rund einem Zehntel der Instanzen findet das System praktisch nichts mehr.
Für die Motivation der Arbeit ist das der eigentliche Befund: Retrieval in unaufgeräumten
Szenen scheitert nicht am Encoder, sondern an der Sichtbarkeit.

**Die Kontrolle, die die Aussage einschränkt:** die vier Klassen sind ungleich besetzt *und*
unterschiedlich zusammengesetzt — T-LESS stellt 73 % des stark verdeckten Bins, aber nur 47 %
des freien, und hat ohnehin das schwächste Retrieval. Deshalb dieselbe Auswertung je
Datensatz:

| R@1 je Datensatz | < 50 % | 50–80 % | 80–95 % | > 95 % | Spanne |
|---|---|---|---|---|---|
| YCB-V | 0.371 | 0.413 | **0.880** | 0.829 | 0.51 |
| T-LESS | 0.062 | 0.201 | 0.391 | **0.429** | 0.37 |
| LM-O | 0.093 | 0.334 | 0.534 | **0.650** | 0.56 |

**Der Effekt hält, die eine große Zahl nicht:** in jedem Datensatz einzeln steigt R@1 mit
der Sichtbarkeit — die Zusammensetzung erzeugt den Effekt nicht, *überzeichnet* ihn aber
(Spannen 0.37–0.56 statt gepoolt 0.53). Ehrlich formuliert ist Verdeckung rund das **11- bis
17-fache** der größten Designachse. Der 3b-Verlauf ist zudem **nicht monoton** (freier Bin
18.8 mm über dem leicht verdeckten 14.4 mm) — das bleibt nach Durchmesser-Normierung und in
allen drei Datensätzen bestehen; kontrolliert man aber je Objekt (43 Objekte mit ≥ 25
Instanzen in beiden Klassen), steht es 26:17 bei einer Mediandifferenz von 0.01 — zu schwach,
um es zu deuten. **Interpretiert wird nur der monotone Teil bis 80–95 %.**

**Was diese Zahlen nicht sagen:** `visib_fract` vermischt Verdeckung durch andere Objekte
mit Abschneiden am Bildrand. Die Auswertung nutzt GT-Masken — gemessen ist, was Verdeckung
*bei perfektem Ausschnitt* anrichtet; mit echter Segmentierung wäre der Effekt eher größer.
Und der Befund ist korrelational; die Kontrolle je Datensatz schließt nur die naheliegendste
Alternative aus.

> 📊 **Grafik-Empfehlung G9 — R@1 über Sichtbarkeit:** Liniendiagramm R@1 über die vier
> Sichtbarkeitsklassen, eine Linie je Datensatz plus gepoolte Linie (gestrichelt);
> Balkenbreite oder Beschriftung für die Klassenbesetzung. Der 3b-Teil als zweites Panel nur
> bis 80–95 % (nicht-monotoner Rest ausgegraut). Daten:
> [stage3/occlusion_by_visibility.csv](stage3/occlusion_by_visibility.csv). *Noch in keinem
> Artefakt als Grafik umgesetzt.*

### Diskussion (Stage 3)

1. **Es gibt keine einzelne beste Konfiguration** — die Wahl hängt von Aufgabe und
   Gallery-Inhalt ab: Retrieval allein → cross × full-mesh (R@1 0.5151); Pose gegen eine
   reine Proxy-Gallery → cross × partial (18.37 mm); Pose mit verwandten CADs aus demselben
   Datensatz → cross × full-mesh (13.51 mm). Ohne Geometrie in allen drei Fällen.
2. **Wann sich Geometrie lohnt, ist operativ definierbar** — nicht über die aggregierte
   Retrieval-Zahl, sondern über die bedingte Top-1-Genauigkeit innerhalb der Shortlist und
   den Kopfraum Recall@K − Top-1 (SHREC 0.520 → Geometrie gewinnt +15.2 Pp.; BOP 0.251 →
   Geometrie verliert −8.0 Pp.). Beide Zahlen liegen aus dem Retrieval-Lauf vor, das
   Kriterium ist vorab prüfbar, ohne eine Registrierung zu rechnen. Der Kopfraum erklärt
   auch, warum tiefere Shortlists auf SHREC helfen — und warum Stage 3 mit K=5 läuft.
3. **Retrieval-Genauigkeit und Pose-Nutzen sind verschiedene Größen:** Full-Mesh gewinnt
   Retrieval und verliert Pose (3b); T-LESS hat das schwächste Retrieval und den besten
   Pose-Median; die Geometrie verliert beim Retrieval −0.059 und bei der Pose +57 %.
4. **Die Gallery-Zusammensetzung ist ein eigener Hebel** (3c): eigener Datensatz 10.35 mm
   gegen fremde Gallery 20.10 mm. Eine kuratierte oder größere Proxy-Gallery ist
   naheliegender als weitere Retrieval-Feinarbeit.
5. **Verdeckung dominiert alle Designachsen** (3d): 11- bis 17-fache Wirkung, bei < 50 %
   Sichtbarkeit bricht das Retrieval auf R@1 0.098 ein — mit perfekten Masken.

---

## Stage 4 — Latenz: Was kostet das alles?

**Setup.** Gemessen wird der Pfad, den die Pipeline ohnehin nimmt (kein Benchmark-Nachbau; →
Methodik unten). Anfrageseite: 50 Anfragen je View-Zahl auf YCB-V, **Gallery = 1278 Objekte**
(dieselbe wie Stage 3a für YCB-V: 1257 Proxy-CADs + 21 YCB-V-Ziel-CADs — die Anfrage sucht in
einem Katalog, der zu 98 % aus Objekten besteht, die mit der Szene nichts zu tun haben).
Onboarding-Seite: **Vollerhebung über alle 59 Ziel-CADs**. Hardware: RTX 4090.

### 4.1 Eine Anfrage, Schritt für Schritt

Sprachprompt → Segmentierung → Punktwolke → drei Kanäle → Fusion → Pose. Mediane:

| Schritt | 16 Views | 42 Views |
|---|---|---|
| io_load | 8.8 ms | 17.1 ms |
| segment | 230.6 ms | 242.1 ms |
| pointcloud | 1.3 ms | 1.3 ms |
| encode_query | 37.6 ms | 37.8 ms |
| clip | 19.8 ms | 18.7 ms |
| dino | 296.3 ms | **561.8 ms** |
| ulip | 176.7 ms | 219.4 ms |
| fusion | 12.0 ms | 12.0 ms |
| pose | **1402 ms** | **1487 ms** |
| **Ende zu Ende** | **2.184 s** | **2.602 s** |
| + Geometrie (K=5) | — | +1840 ms |

Kaltstart einmalig (nicht Teil der Anfrage): Gallery 8.0 s + GroundingDINO/SAM 4.6 s.

- **Die Pose ist der Flaschenhals, nicht das Retrieval:** FoundationPose macht 64 % der
  Anfrage aus (57 % bei 42 Views) und ist der unruhigste Schritt — p95 bis 4.5 s gegenüber
  2.2 s Median (Hypothesenverfeinerung, kein Messrauschen). Das gesamte Retrieval über 1278
  Objekte kostet weniger als 0.8 s.
- **Nur DINOv2 skaliert mit der View-Zahl** (296 → 562 ms); alles andere ist konstant.
- **Geometrisches Re-Ranking verdoppelt die Anfrage:** 1.84 s für fünf Kandidaten — mehr als
  die gesamte übrige Kette ohne Pose (≈ 1.05 s). Zusammen mit Stage 3 (Genauigkeit sinkt in
  allen vier Zellen) ist der Fall entschieden: **der teuerste Schritt ist zugleich der
  einzige, der schadet.**

> 📊 **Grafik-Empfehlung G10 — Zeitbudget-Balken (Anfrage + Onboarding):** horizontale
> Balken je Schritt, gemeinsame Skala über beide View-Spalten, Pose/Render/Describe farblich
> als „heiße“ Posten abgesetzt; je ein Panel für Anfrage (4.1) und Onboarding (4.2). Daten:
> [stage4/](stage4/) (`query_latency_ycbv*.json`, `onboarding*.json`). Bereits im
> [Stage-4-Artefakt](artifacts/stage4_results.html), §1–2.

### 4.2 Ein neues Objekt aufnehmen

Vom CAD-File zur auffindbaren Gallery. Mediane je CAD, **Vollerhebung n=59** für sieben der
acht Stufen (nachgezogen 2026-09-07); nur dGeDi ruht auf n=3:

| Stufe | 16 Views | 42 Views |
|---|---|---|
| render (Blender) | **10.75 s** | **25.93 s** |
| describe (LLaVA) | **10.25 s** | **13.08 s** |
| partial (HPR) | 1.35 s | 2.76 s |
| embed_ulip | 0.59 s | 1.55 s |
| embed_dino | 0.12 s | 0.29 s |
| embed_clip | 4.6 ms | 4.7 ms |
| mesh | 0.12 s | 0.10 s |
| cache laden + schreiben | 0.22 s | 0.23 s |
| **Summe** | **23.49 s** | **44.13 s** |
| + dGeDi-Deskriptoren *(n=3)* | 11.30 s | 10.21 s |

IQR der Render-Stufe: 3.46 s (16 V) bzw. 8.61 s (42 V). Render und dGeDi laufen auf dem
Host, der Rest im Container.

- **Rendern und Beschreiben sind 89 % — das Encodieren 3 %.** Die Intuition sagt, das Teure
  seien die neuronalen Encoder; gemessen ist das Gegenteil: DINOv2 + CLIP + ULIP-2 kosten
  zusammen 0.71 s (16 V), Blender + LLaVA 21.0 s — das 30-Fache.
- Zwei Posten skalieren nicht mit der View-Zahl: **CLIP-Text** (16 oder 42 kurze Strings in
  einem Batch) und der **Cache-Schreibvorgang** (hängt an der Gallery-Größe, nicht am neuen
  Objekt). Wer Views spart, spart beim Encoding — nicht beim Cache.
- **Die Geometrie kostet an zwei Stellen:** beim Onboarding einmalig 10.5 s je Objekt
  (view-unabhängig — eine einzige Punktwolke von 10 000 Punkten), bei jeder Anfrage nochmals
  1.84 s, weil der Deskriptor der Anfrage-Wolke jedes Mal neu berechnet wird.

### 4.3 Der Handel: 16 gegen 42 Ansichten

| | 16 Views | 42 Views | Kosten (16 V als Anteil) | nDCG (Stage 1) |
|---|---|---|---|---|
| Onboarding je CAD | **23.49 s** | 44.13 s | 53 % | 0.5820 vs. 0.5868 |
| Anfrage, nur Retrieval | 0.78 s | 1.12 s | 70 % | |
| Anfrage, Ende zu Ende | 2.18 s | 2.60 s | 84 % | |

**Der Hebel liegt beim Onboarding, nicht bei der Anfrage:** ein neues Objekt kostet bei 16
Ansichten die Hälfte — für 0.005 nDCG. Auf der Anfrageseite schrumpft der Vorteil auf 16 %,
weil die konstante Pose-Zeit ihn verdünnt.

### 4.4 Inkrementell gegen Invalidierung

Der Cache-Schlüssel ist ein Fingerprint über das gesamte Inventar; ein neues Objekt ändert
den Hash und entwertet alles:

| Was | Dauer | Ordnung |
|---|---|---|
| Ein Objekt aufnehmen (42 Views) | **44.13 s** | O(1) |
| — davon Eintrag einfügen | 0.1 ms | O(1) |
| — davon Cache laden + schreiben | 0.23 s | O(Gallery) |
| Was der Fingerprint erzwingt | **34.7 min** | O(Gallery) |

Invalidierung hochgerechnet aus 1.657 s je Objekt × 1257 Gallery-Objekte (gemessen am
Encoden echter Gallery-Objekte ohne Cache). Selbst ein anhängender Cache zahlt für die ganze
Gallery (monolithische `.pt`-Datei) — aber Serialisierung statt Encoding: **44 Sekunden
gegen 35 Minuten, Faktor 47.**

### 4.5 Partielle Wolken oder Full-Mesh — was kostet die Repräsentation?

Stage 1 und 2 entscheiden die Repräsentationsachse nach Qualität; hier steht, was sie an
Zeit kostet. **Onboarding: 8–10 %** (zurechenbare Differenzen, Mediane über n=59):

| Stufe | 16 V | 42 V |
|---|---|---|
| `partial` (HPR) entfällt | −1.35 s | −2.76 s |
| `embed_ulip`: N Encodes → einer | −0.55 s | −1.51 s |
| `io_load_clouds` entfällt | −0.03 s | −0.03 s |
| `mesh_sample` kommt hinzu | +0.09 s | +0.08 s |
| **zurechenbar** | **−1.84 s** | **−4.22 s** |
| gemessene Summe | 23.49 → 21.65 s | 44.13 → 39.91 s |
| Anteil | −7.8 % | −9.6 % |

Bei 42 Views war die *beobachtete* Differenz mit −5.57 s größer als die zurechenbare; der
Rest sind 1.33 s bei `describe` — einem Schritt, der sich zwischen den Konfigurationen gar
nicht unterscheiden kann (LLaVA über dieselben Renderings). Das ist Lauf-zu-Lauf-Streuung;
berichtet wird die zurechenbare Zahl.

**Anfrage: 18 % des Retrievals — aber fast alles davon ist Implementierung** (Mediane, n=50):

| Median je Anfrage | partial | full-mesh | Δ |
|---|---|---|---|
| `ulip`, 16 Views | 176.7 ms | **38.3 ms** | −138.4 ms |
| `ulip`, 42 Views | 219.4 ms | **38.1 ms** | −181.3 ms |
| Anfrage ohne Pose, 16 V | 783 ms | 641 ms | −18.2 % |
| Anfrage ohne Pose, 42 V | 1110 ms | 902 ms | −18.7 % |
| Anfrage mit Pose, 16 V | 2185 ms | 2043 ms | −6.5 % |
| Anfrage mit Pose, 42 V | 2597 ms | 2389 ms | −8.0 % |

Der Full-Mesh-Wert ist view-unabhängig (38.3 gegen 38.1 ms) — erwartungsgemäß, denn dort
liegt je Objekt genau ein Embedding. **Diese Zahl darf man nicht als Eigenschaft der
Repräsentation lesen:** die Zweige sind unterschiedlich implementiert — Full-Mesh stapelt
die Gallery und rechnet ein Matrixprodukt (`step5_shape_matching.py:1513`), Partial iteriert
in Python über alle 1278 Objekte und schiebt jedes einzeln auf die GPU (`:1490`). Die
Rechnung selbst ist nicht der Kostentreiber: Partial braucht 42-mal mehr Skalarprodukte
(6.9·10⁷ statt 1.6·10⁶ MACs) — auf dieser Karte rund **3 Mikrosekunden** Unterschied.
Gemessen sind **138 Millisekunden**, also 0.108 ms je Gallery-Objekt Schleifen-Overhead
(Kernel-Start, Datentransfer). Ein vektorisierter Partial-Pfad läge nahe bei den 38 ms. Die
Zahl gehört ins Latenzbudget als Eigenschaft *dieses Systems* — nicht in eine Begründung,
welche Repräsentation man wählen sollte.

**Die Wahl ist eine Qualitätsfrage, keine Zeitfrage:** Onboarding 8–10 %, Anfrage 6–8 % mit
Pose — das Meiste davon behebbar. Dem stehen Qualitätsunterschiede gegenüber, die je Stage
in verschiedene Richtungen zeigen (Stage 1 pc fusioniert pro Full-Mesh, Stage 2 cross
fusioniert pro Partial). Wer nach Laufzeit entscheidet, entscheidet nach der kleineren Größe.

### 4.6 Wie gemessen wurde

- **Der echte Pfad, kein Nachbau:** das Messskript ruft `run_query` bzw. die
  Onboarding-Stufen genau so auf, wie Stage 1–3 sie aufrufen; Einzelschritte werden sichtbar,
  indem das Skript die betreffenden Methoden *umhüllt* — die geteilten Module bleiben
  unverändert.
- **Synchronisiert:** CUDA-Kernel laufen asynchron; jede Messung wartet vor und nach dem
  Schritt auf die GPU (`torch.cuda.synchronize()`), sonst misst man das *Einreihen* des
  Kernels und alle GPU-Kosten rutschen in den nächsten zufällig synchronisierenden Schritt.
- **Statistik über Objekte, nicht über Aufrufe:** mehrfache Aufrufe derselben Funktion
  innerhalb eines Objekts werden vor der Aggregation summiert.
- **Median, IQR, p95** statt Mittelwert/Standardabweichung — Latenzen sind rechtsschief; p95
  steht daneben, weil für ein interaktives System der schlechte Fall zählt.
- **Kalt und warm getrennt**; das Cache-Anhängen ist echt gemessen (laden, einfügen,
  zurückschreiben), nicht simuliert.

**Stichproben:**

| Bereich | n | Was das ist |
|---|---|---|
| Anfrage, je View-Zahl | 50 | Stichprobe aus den YCB-V-Testzielen |
| Onboarding, 7 von 8 Stufen (inkl. `render`) | **59** | **Vollerhebung** — alle Ziel-CADs |
| Onboarding, `dgedi` | 3 | Stichprobe |
| Geometrie je Anfrage | 25 / 12 | zwei unabhängige Läufe |

**Was die Nachmessung ergeben hat:** Die Render-Stufe lag bis zum 2026-09-07 bei n=5 — und
war um 25 % zu hoch (14.45 → 10.75 s bei 16 V; 34.68 → 25.93 s bei 42 V). Der Grund ist kein
Zufall: `--max-objects 5` nimmt die *ersten fünf* Meshes in Sortierreihenfolge, keine
Zufallsstichprobe — und die lagen geschlossen am oberen Rand; der alte Median traf exakt das
*Maximum* über alle 59 CADs, und der scheinbare IQR von 0.32 s war in Wahrheit 3.46 s. Die
Onboarding-Summen sanken dadurch von 27.18 auf 23.49 s bzw. 52.97 auf 44.13 s; **alle
Schlussfolgerungen des Kapitels blieben bestehen** (Render+Describe 89 %, Encoding 3 %, 16 V
≈ halbe Kosten). Die Geometriezahl ist die reproduzierte: eine erste Messung ergab 5.45 s
und reproduzierte nicht; zwei spätere Läufe liefern 1.84 s (n=25) und 2.07 s (n=12).
**Offen bleiben:** die dGeDi-Onboarding-Stufe (n=3) und dass die Anfrageseite nur auf YCB-V
gemessen ist — ob die Segmentierung auf texturlosen Objekten (T-LESS, ITODD) teurer wird,
ist unbeantwortet.

### Diskussion (Stage 4)

1. **Der Flaschenhals ist die Pose, nicht das Retrieval** — 64 % der Anfrage, p95 4.5 s.
   Wer die Pipeline beschleunigen will, setzt bei FoundationPose an, nicht bei den Encodern.
2. **Beim Onboarding dominieren Rendern und Beschreiben (89 %)**, nicht die Encoder (3 %).
   16 statt 42 Views halbieren das Onboarding für 0.005 nDCG — der beste Zeit-Deal der
   ganzen Pipeline.
3. **Der Cache-Fingerprint ist der größte vermeidbare Posten:** Faktor 47 zwischen
   inkrementellem Anhängen (44 s) und erzwungener Voll-Invalidierung (34.7 min).
4. **Die Geometrie ist doppelt teuer** (10.5 s Onboarding + 1.84 s je Anfrage) **und senkt
   auf BOP die Genauigkeit** — die Kosten-Nutzen-Rechnung ist an keiner Stelle positiv,
   außer auf SHREC-artigen Daten (→ Kriterium in Stage 3).
5. **Die Repräsentationsfrage ist keine Zeitfrage:** der messbare ulip-Unterschied
   (138–181 ms) ist zu ~100 % Schleifen-Overhead der heutigen Implementierung, nicht
   Arithmetik. Eine Vektorisierung des Partial-Pfads würde die Stage-4-ulip-Zahl
   invalidieren und ist deshalb bewusst *nicht* vor Abschluss der Messreihe erfolgt.
6. **Messmethodische Lehre:** `--max-objects N` ist keine Stichprobe, sondern „die ersten
   N“ — dieselbe Verzerrung hat in dieser Messreihe zweimal zugeschlagen (Render n=5,
   mesh_sample-Smoke-Test). Konsequenz: Vollerhebung, wo immer sie bezahlbar ist.

---

## Übergreifende Diskussion

Punkte, die erst über mehrere Experimente hinweg sichtbar werden — jeweils mit den Stellen,
an denen sie belegt sind.

**1 · Die Fusionsthese von OSCAR+ hält über alle drei Datensätze — aus wechselnden Gründen.**
Der Shape-Kanal hebt die Fusion auf SHREC (+0.035 nDCG), MI3DOR (+0.053 FT auf den stärksten
Einzelkanal) und BOP (+0.162 R@1 gegen die Baseline ohne Shape) — obwohl er isoliert nie der
stärkste Kanal ist und im cross-Modus sogar der schwächste. Der Mechanismus ist überall
derselbe: er sieht, was die anderen nicht sehen (keyboard/bag auf SHREC, camera/vase auf
MI3DOR, texturloses T-LESS auf BOP).

**2 · Der Beitrag eines Kanals ist seine Fehler-Dekorrelation, nicht seine Genauigkeit.**
Dreimal unabhängig belegt: MI3DOR isoliert full-mesh +9.99 NN, fusioniert −1.87 (2.4);
SHREC isoliert partial +0.0397, fusioniert −0.0067 (1.3 A4); und die Fusion gewinnt in beiden
Kategorien-Analysen genau dort, wo die Kanäle *ähnlich stark* sind, und verliert, wo einer
dominiert (1.7, 2.6). Für Systembau heißt das: Kanäle nach Komplementarität auswählen, nicht
nach Einzel-Benchmark.

**3 · „Partial oder Full-Mesh?“ hat keine globale Antwort — aber eine Entscheidungsregel.**
Die Achse kippt mit drei Dingen: Query-Modus (Stage 1 A4b), isoliert vs. fusioniert (Stage 1
+ 2), Datensatz (Stage 3: T-LESS gegen den Rest). Fusioniert stimmen die beiden
cross-Datensätze überein (partial vorn); als reiner Retrieval-Arm auf BOP gewinnt full-mesh
überall außer T-LESS/pc. Die Zeitkosten der Achse sind klein und großteils behebbare
Implementierung (Stage 4.5) — **die Wahl ist eine Qualitätsfrage je Einsatzszenario, keine
Kostenfrage.**

> 📊 **Grafik-Empfehlung G11 — Vorzeichen-Übersicht der Repräsentationsachse:** kompakte
> Matrix (Zeilen: SHREC pc/cross, MI3DOR cross, BOP ycbv/tless/lmo × pc/cross; Spalten:
> isoliert / fusioniert), Zellfarbe = Sieger (partial vs. full-mesh), Zellwert = Δ. Eine
> einzige Abbildung, die den scheinbaren Widerspruch aller drei Stages auflöst. Daten in
> den Tabellen 1.3 A4b, 2.4, 3a. *Noch in keinem Artefakt umgesetzt.*

**4 · Der Query-Modus folgt der Aufnahmesituation.** Saubere Scans → Punktwolken-Query
überlegen (SHREC: pc +0.054 über cross); verrauschte Sensortiefe in unaufgeräumten Szenen →
Bild-Query überlegen (BOP: cross 0.4818 über pc 0.4636, und cross × full-mesh 0.5151 als
bester Arm). Belegt an zwei Datensätzen — als Handlungsregel brauchbar, als Gesetz nicht
ausgemessen, weil Tiefenqualität mit Szenenkomplexität und Gallery konfundiert ist.

**5 · Geometrie-Reranking ist eine Verdrängung — und wann sie sich lohnt, ist vorab
berechenbar.** Auf SHREC der größte Einzelhebel (+0.130 hit@1), auf BOP durchgehend negativ
(Retrieval −0.054 bis −0.091, Pose +57 %), obendrein der teuerste Schritt (Stage 4: 1.84 s/
Anfrage, 10.5 s/Objekt Onboarding). Die Auflösung: Re-Ranking ersetzt den Fusions-Score
innerhalb der Shortlist und lohnt nur, wenn es ihn schlägt — messbar als bedingte
Top-1-Genauigkeit, abschätzbar als Kopfraum Recall@K − Top-1 (SHREC 0.520, BOP 0.251), beides
ohne eine einzige Registrierung. Zusätzliche inhaltliche Grenzen: deformierbare Objekte
(pillow −0.248) und metrische Skala (auf BOP schlägt Fitness die Distanz, auf SHREC
umgekehrt).

**6 · Kaskade vs. volle Fusion ist ein Architektur-, kein Tuning-Unterschied.** Der
CLIP-Schwellwert τ=0.37 prunt auf SHREC bei 98.3 % und auf MI3DOR bei 96.9 % der Queries auf
leer (→ Top-20-Fallback); wo er greift, ist die Shortlist zu kurz zum Umsortieren.
Listen-Metriken kann eine 20er-Shortlist strukturell nicht verbessern (FT bleibt exakt auf
CLIP-Niveau), den Kopf sehr wohl (NN +18.6). Die volle Fusion schlägt die Kaskade auf allen
drei Datensätzen (SHREC +0.131 nDCG, MI3DOR +0.107 FT, BOP +0.162 R@1 und −3.4 mm Pose).

**7 · Feste Gewichte sind der dokumentierte Preis der Einfachheit.** Negativ in 12 von 20
(SHREC) bzw. 9 von 21 Kategorien (MI3DOR), Verluste bis −0.294 — dreimal so groß wie der
größte Gewinn. Die BASE-Gewichte sind zugleich robust nahe am Plateau-Optimum beider
Datensätze (+0.003 / +0.005); tunen lohnt kaum, aber die *Struktur* (ein Gewichtssatz für
alle Queries) kostet messbar. Gemeinsame Richtung beider cross-Karten: „Shape herunter“;
wohin das Gewicht wandert, ist datensatzspezifisch (SHREC → View, MI3DOR → Text). Adaptive
oder per-Query gewählte Gewichte sind der am besten belegte Ausblick der Arbeit.

**8 · Metriken müssen zur konsumierenden Stufe passen.** Listen-Metriken (nDCG, FT) und
Kopf-Metriken (hit@1, NN, R@1) küren wiederholt verschiedene Sieger: Text+View schlägt View
auf nDCG und verliert auf hit@1 (1.9); die Kaskade hebt NN um 18.6 bei unverändertem FT
(2.2); nDCG unterschätzt den Geometrie-Beitrag um Faktor 2.5 (1.9). Da die Pose-Stufe nur
den Top-1 konsumiert, ist die Kopf-Metrik die bindende — und selbst sie reicht nicht:
Retrieval-Rang und Pose-Nutzen trennen sich noch einmal (3b: full-mesh findet öfter das
richtige CAD und posiert schlechter; T-LESS: schwächstes Retrieval, bester Pose-Median).

**9 · Farbe ist die konsistenteste offene Variable.** Stage 1: Query-Farbe leicht negativ,
aber konfundiert durch den Checkpoint-Tausch (A6); der XYZ-Turm kann prinzipiell keine
Bild-Query beantworten, weshalb überall der farbige Turm steht (1.9 Punkt 9). Stage 2: die
gesamte MI3DOR-Gallery ist farblos, der farbige Encoder bekommt drei tote Eingangskanäle —
ein zweiter plausibler Grund für den schwachen Shape-Kanal, unentscheidbar ohne Neulauf.
Eine saubere Farb-Ablation (ein Encoder, genullte Farbkanäle; Gallery mit echter Farbe)
existiert in keiner Stufe und wäre die erste Zusatzmessung, wenn Zeit bliebe.

**10 · Die größten Effekte der Arbeit sind nicht die Designachsen.** Rangfolge der gemessenen
Wirkungen: Verdeckung (0.37–0.56 R@1-Spanne je Datensatz) ≫ Gallery-Zusammensetzung (Faktor
~2 im Substitutionsfehler: 10.35 vs. 20.10 mm) ≫ Geometrie auf SHREC (+0.130 hit@1) >
Query-Modus (−0.054 nDCG) > Repräsentation (±0.03) > Encoder-Wahl (+0.034) > View-Zahl,
Gewichte, Fusionsstrategie (≤ 0.02). Für ein einsetzbares System sind Sichtbarkeit und
Gallery-Kuration die Hebel — nicht weiteres Feintuning der Retrieval-Kette.

**11 · Grenzen, die für alle Stufen gelten.** (a) Alle BOP-Auswertungen nutzen GT-Masken
und GT-Boxen — reale Segmentierung käme als Fehlerquelle dazu, Effekte wie 3d wären eher
größer. (b) Verdeckungs- und Modusbefunde sind korrelational bzw. an zwei Datensätzen
belegt. (c) Einzelne Messlücken bleiben offen und sind benannt: dGeDi-Onboarding n=3,
Anfrage-Latenz nur YCB-V, Stage-2-Sweep auf der Full-Mesh-Gallery, keine saubere
Farb-Ablation. (d) Ohne Konfidenzintervalle berichtet; Stabilität stattdessen durchgehend
über gepaarte Per-Query-Bilanzen (1.6) — die in mehreren Fällen mehr sagen als der
Mittelwert.

---

## Verzeichnis der Grafik-Empfehlungen

| Nr. | Abschnitt | Grafik | Status | Daten |
|---|---|---|---|---|
| G1 | 1.3 A4b | Divergierende Kategorien-Balken full-mesh vs. partial (4 Panels) | ✔ im Artefakt S1 | `stage1/category_fullmesh_vs_partial.csv` |
| G2 | 1.4 B2 | Ternäre Gewichtskarten SHREC pc & cross | ✔ im Artefakt S1 | `stage1/weightmap_pc.csv`, `stage1/weightmap_cross.csv` |
| G3 | 1.5 C2 | Ertrag vs. Kosten der Shortlist-Tiefe K | ☐ neu zu erstellen | Tabelle C2 |
| G4 | 1.7 | Kategorien-Balken 3 Kanäle × 20 Klassen | ✔ im Artefakt S1 | `stage1/category_channels.csv` |
| G5 | 2.2 | Shortlist-Schema der Kaskade (NN/F1/FT/ST) | ✔ im Artefakt S2 | Tabelle 2.2 |
| G6 | 2.3 | Ternäre Gewichtskarte MI3DOR | ✔ im Artefakt S2 | `stage2/weight_sweep_mi3dor.csv` |
| G7 | 2.6 | Kategorien-Balken 3 Kanäle + Fusion × 21 Klassen | ✔ im Artefakt S2 | `stage2/category_table.csv` |
| G8 | 3a | R@1 je Arm und Datensatz (Facetten) | ☐ neu zu erstellen | `stage3/*/combined_stage3a.json` |
| G9 | 3d | R@1 über Sichtbarkeitsklassen, je Datensatz | ☐ neu zu erstellen | `stage3/occlusion_by_visibility.csv` |
| G10 | 4.1–4.2 | Zeitbudget-Balken Anfrage + Onboarding | ✔ im Artefakt S4 | `stage4/*.json` |
| G11 | Übergreifend 3 | Vorzeichen-Matrix der Repräsentationsachse über alle Stages | ☐ neu zu erstellen | Tabellen 1.3, 2.4, 3a |

---

*Quellen: die vier HTML-Artefakte unter [artifacts/](artifacts/) (interaktive Fassungen),
Ergebnisordner und CSVs in diesem Ordner (Zuordnung im [README.md](README.md)). Repo-Docs:
`docs/STAGE1_RESULTS.md`, `docs/STAGE2_RESULTS.md`, `docs/STAGE3_RESULTS_SUMMARY.md`,
`docs/STAGE4_RESULTS.md`, `docs/RUN_PROVENANCE.md`, `docs/RESULTS_OVERVIEW.md`.*
