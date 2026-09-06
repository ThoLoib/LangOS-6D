# Stage 2 — MI3DOR Transfer-Test: vollständige Ergebnisse

*Alle Arme der gewählten Config mit vollständigen Metriken, plus A4-Transfer,
Gewichts-Heatmap, OSCAR-Legacy-Vergleich und Kategorien-Analyse. Ergebnisordner:
`object_retrieval/results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix/fullmesh`.
Stand 2026-09-06. n = 10.500 Queries / 3.848 CADs / 21 Kategorien.*

Begleitdokumente: `STAGE1_RESULTS.md` · `EVALUATION_STORY_AND_PLAN.md` ·
`CONFIG_COMPARISON.md`.

---

## 0. Konfiguration

| Komponente | Wert |
|---|---|
| **S_text** | CLIP ViT-B/32, Query-Bild vs. Per-View-Beschreibungen, max über Views |
| **S_view** | DINOv2-base, **mean**-Pooling, **42 Views**, top-k-softmax **k=5**, τ=0.5 |
| **S_shape** | ULIP-2 coloured (1280-d), **cross-Modus** (Query-**Bild** über den OpenCLIP-ViT-bigG-Turm), **Full-Mesh**-Referenz |
| **Fusion** | Weighted Sum, w = **(0.3, 0.4, 0.3)**, volle Datenbank |
| **Kaskaden-Arme** | CLIP-Schwellwert **τ = 0.37**, Top-20-Fallback |
| **Geometrie** | **keine** — MI3DOR hat keine Query-Punktwolken |
| Metrik-Tiefe | top-k = 15, TOP_F = 20 |

> **Warum cross-Modus:** MI3DOR liefert **monokulare Bilder ohne Tiefe**. Es existiert keine
> Query-Punktwolke, der Shape-Kanal *muss* also das Bild encodieren. Genau das misst
> Stage-1-A5 vorab: der Wechsel pc → cross kostet dort −0.054 nDCG bzw. **−19,5 % hit@1**.

> ⚠️ **Full-Mesh statt Partial-Views.** Auf dieser Maschine fehlen die MI3DOR-`*_partial.npz`;
> `build_pipeline` fällt still auf Full-Mesh zurück (Logbeleg: *„no partial PCs found …
> Falling back to full-mesh encoding"*). Das war eine **Nebenwirkung, keine Entscheidung** —
> inhaltlich aber die bessere Wahl, siehe §3.

---

## 0.1 Die Metriken (MI3DOR/SHREC-08-Konvention)

Andere Konvention als Stage 1 — hier gibt es **keine** Subkategorien, die Relevanz ist
**binär auf Kategorieebene**: relevant = gleiche der 21 Kategorien.

| Metrik | Was sie misst |
|---|---|
| **NN** | Nearest Neighbour — ist der **Top-1**-Treffer aus der richtigen Kategorie? (in %) |
| **FT** (First Tier) | Anteil relevanter Treffer in den Top-**C** (C = Kategoriegröße) |
| **ST** (Second Tier) | relevante Treffer in den Top-**2C**, normiert auf **C** (nicht 2C) |
| **F1** | harmonisches Mittel aus Precision und Recall bei **TOP_F = 20** |
| **nDCG@2R** | DCG über Top-2C, normiert auf das ideale DCG über C |
| **mAP** | Average Precision über die Rangliste |
| **ANMRR** ↓ | Average Normalized Modified Retrieval Rank — **kleiner ist besser** |

**Headline für Stage 2: NN und FT.** NN ist das Pendant zu Stage-1s hit@1 (Top-1-Güte),
FT die Listenqualität. ANMRR wird als rangbasiertes Gegenstück mitberichtet.

---

## 1. Die sieben Arme

| Arm | NN | FT | ST | F1 | nDCG@2R | mAP | ANMRR ↓ |
|---|---|---|---|---|---|---|---|
| CLIP-Text allein | 67.95 | 0.575 | 0.755 | 0.160 | 0.720 | 0.580 | 0.339 |
| ULIP-2 allein (cross) | 78.10 | 0.510 | 0.649 | 0.188 | 0.652 | 0.518 | 0.409 |
| DINOv2 allein | 83.03 | 0.629 | 0.753 | 0.200 | 0.751 | 0.647 | 0.297 |
| **CLIP+DINO+ULIP (volle Fusion)** | **86.57** | **0.682** | **0.822** | **0.215** | **0.813** | **0.705** | **0.238** |
| OSCAR-Kaskade (Hard-Max) | 84.88 | 0.575 | 0.755 | 0.160 | 0.733 | 0.592 | 0.337 |
| OSCAR-Kaskade (Softmax) | 85.04 | 0.575 | 0.755 | 0.160 | 0.734 | 0.592 | 0.337 |
| CLIP-gepruned + DINO+ULIP | 86.52 | 0.575 | 0.755 | 0.160 | 0.735 | 0.593 | 0.337 |

### Der Kernbefund
**Die volle Fusion gewinnt auf jeder Metrik.** Gegenüber dem stärksten Einzelkanal (DINOv2):
**+3,5 NN, +0.053 FT, +0.058 mAP, −0.059 ANMRR**. Der Shape-Kanal ist im cross-Modus der
**schwächste** Kanal (FT 0.510) — trägt aber trotzdem messbar bei.

### ⚠️ Die Kaskade kann Listen-Metriken *strukturell* nicht verbessern

Alle drei Kaskaden-Arme haben **exakt dieselben** FT / ST / F1 wie CLIP-Text allein
(0.575 / 0.755 / 0.160), erreichen aber gleichzeitig fast das NN der vollen Fusion
(86.52 gegen 86.57). Beides zugleich folgt direkt daraus, **worauf** DINO und ULIP umsortieren.

**Auf wie viele Kandidaten wird umsortiert?** Die Shortlist ist
`S' = {o : sim_text(o) >= tau}` mit tau = 0.37. Gemessen über alle 10.500 Queries:

| Fall | Queries | Anteil | Größe von S' |
|---|---|---|---|
| Schwellwert prunt auf **leer** → Top-20-Fallback | 10.174 | 96,9 % | genau 20 |
| Schwellwert greift | 326 | 3,1 % | Median **2** (153× nur 1; max 113) |

Die Kaskade ist damit faktisch **„CLIP-Top-20 → DINO+ULIP"**. Daraus folgt beides:

- **NN misst Rang 1** — der liegt immer in den umsortierten 20. NN springt 67.95 → 86.52 (**+18,6**).
- **FT misst die Top-C**, C im Mittel **183** (31…250). Die Ränge 21…3848 hat nie jemand
  angefasst; sie stehen weiter in CLIP-Reihenfolge. Also **fällt FT exakt auf CLIPs Wert**,
  ebenso ST und F1.

Das ist ein **architektonisches** Ergebnis, kein Parameterproblem: jede Kaskade, deren Shortlist
kürzer ist als die Tiefe der Metrik, kann diese Metrik nicht verbessern. `tau` zu senken hilft
nicht — die Shortlist ist ja fast immer schon der 20er-Fallback; man müsste den **Fallback** auf
>= C anheben, womit das Pruning seinen Zweck verlöre. Die volle Fusion bewertet stattdessen alle
3848 CADs simultan und schlägt die Kaskade auf FT um **+0.107**.

---

## 2. Gewichts-Sensitivität (echte MI3DOR-Heatmap)

231 Gitterpunkte über dem Gewichts-Simplex, als Tier-2-Ableitung aus einmalig gecachten
Kanal-Scores. **Selbstcheck:** FT bei BASE = 0.6851 gegen erwartete 0.682 aus dem
Produktionslauf — Abweichung 0,003, der Sweep ist verifiziert.

| Gewichte (text, view, shape) | FT | NN |
|---|---|---|
| **(0.45, 0.35, 0.20)** ← Optimum | **0.6902** | 86.99 |
| (0.40, 0.35, 0.25) | 0.6897 | 86.97 |
| (0.40, 0.40, 0.20) | 0.6894 | 87.08 |
| (0.45, 0.30, 0.25) | 0.6894 | 86.99 |
| **(0.30, 0.40, 0.30)** ← BASE | 0.6851 | 86.84 |

**BASE ist robust:** nur **+0.005 FT** zum Optimum — kein Tuning nötig, dieselbe
Schlussfolgerung wie im pc-Modus auf SHREC (+0.003).

**Richtung des Optimums:** ohne Tiefe gehört der Shape-Kanal **herunter** (0.20 statt 0.30) —
das Gewicht wandert aber **zu Text** (0.45), nicht zu View.

> ⚠️ **Korrektur.** Zuvor stand hier (0.3, 0.6, 0.1) aus einer **SHREC-cross-Heatmap als
> Stellvertreter**. Die sagte die Richtung falsch voraus (view-lastig statt text-lastig). Die
> SHREC-cross-Heatmap taugt **nicht** als MI3DOR-Proxy — nur die Aussage „Shape herunter" trug.

---

## 3. A4-Transfer — Partial-Views vs. Full-Mesh im cross-Modus

Aus `..._tau037_dinomean/{partial,fullmesh}` (07./08.08.), dem einzigen MI3DOR-Lauf, in dem
`ulip2_use_partial_views=True` tatsächlich griff. Config verifiziert identisch zur heutigen
(42 Views, k=5, mean, cross, τ=0.37, gleicher Checkpoint, n=10500); einziger Unterschied sind
die Fusionsgewichte (0, 0.5, 0.5) — **für die isolierten Arme wirkungslos**, da sie kein
gewichtetes Ranking bilden. Die fusionierten Arme jenes Laufs werden deshalb **nicht** zitiert.

| ULIP-2 isoliert | partial | **full-mesh** | Δ |
|---|---|---|---|
| NN | 68.11 | **78.10** | **+9.99** |
| FT | 0.453 | **0.510** | +0.057 |
| ST | 0.607 | **0.649** | +0.042 |
| nDCG@2R | 0.598 | **0.652** | +0.054 |
| mAP | 0.451 | **0.518** | +0.067 |
| ANMRR ↓ | 0.467 | **0.409** | besser |

**Full-mesh gewinnt auf jeder Metrik — exakt umgekehrt zu SHREC** (dort partial +0.0397 nDCG
isoliert). Kein Widerspruch, sondern ein verwertbarer Befund:

> **Die Referenz muss zur Natur der Query passen.** SHREC fragt mit einer *partiellen
> Punktwolke* (pc-Modus) → eine partielle Referenz ist geometrisch vergleichbar. MI3DOR fragt
> mit einem *Bild* (cross-Modus) → das Bild zeigt das **vollständige** Objekt, also passt die
> Full-Mesh-Referenz besser.

Damit ist der Full-Mesh-Fallback aus §0 **inhaltlich kein Schaden** — im cross-Modus ist er die
überlegene Wahl.

### Die MI3DOR-Meshes tragen keine Farbe — ein möglicher Grund für den schwachen Shape-Kanal

Nachgeprüft: **alle 3848 Meshes liefern beim Sampling dieselbe Farbe (0.4, 0.4, 0.4)** —
trimeshs Standardgrau. Die Dateien enthalten also weder Vertex- noch Flächenfarben und keine
Textur. Zum Vergleich dieselbe Messung auf den anderen Datensätzen (Standardabweichung der
gesampelten Farbe je Objekt):

| Datensatz | Mesh-Farbe |
|---|---|
| GSO, YCB-V, SHREC'18 | Textur vorhanden (0.10–0.28) |
| LM-O | Vertexfarben, 0.04–0.18 |
| **MI3DOR**, T-LESS, ITODD, HouseCat6D | **0.0 — keine Farbe in den Dateien** |

**Was das für die Interpretation bedeutet.** Der verwendete ULIP-2-Backbone ist der *farbige*
(`pointbert_colored`, 6-Kanal-Eingang, 1280-d) und bekommt über die gesamte Gallery ein
konstantes, informationsloses RGB — drei seiner sechs Eingangskanäle tragen nichts. Das verzerrt
die Gallery nicht *untereinander* (die Konstante trifft alle 3848 Objekte gleich), setzt aber
jedes Gallery-Embedding in einen Bereich des Merkmalsraums, für den der Encoder nicht trainiert
wurde — während die Query durch den Bildturm kommt, der Farbe sehr wohl sieht.

Dass ULIP-2 hier der schwächste Kanal ist (FT 0.510 gegen DINOv2 0.629), war bisher allein der
cross-modalen Schwierigkeit zugeschrieben. **Die farblose Gallery ist ein zweiter, ebenso
plausibler Grund.** Belegt ist das nicht — prüfbar wäre es über partielle Punktwolken aus den
vorhandenen Renderings, was einen kompletten Stage-2-Neulauf bedeutet und deshalb nicht gemacht
wurde (`DECISIONS.md`).

Für die Schlussfolgerung des Kapitels ändert der Befund nichts, er **verschiebt nur die
Begründung**: dass Shape ohne Tiefe heruntergewichtet gehört (§2), gilt weiterhin — die Ursache
ist aber vermutlich nicht nur die fehlende Tiefe, sondern auch eine Gallery, der genau die
Information fehlt, mit der dieser Encoder rechnet.

---

## 4. OSCAR-Legacy-Vergleich (V = 8 Views)

Die publizierte OSCAR-Kaskade wird mit **8 Views** beschrieben. Derselbe Mechanismus, unser
Evaluator, unsere Gallery — nur der View-Count wechselt:

| Arm | NN (V=8) | NN (V=42) | FT (V=8) | FT (V=42) |
|---|---|---|---|---|
| DINOv2 allein | 81.96 | 83.03 | 0.591 | 0.629 |
| OSCAR-Kaskade (Hard-Max) | 84.40 | 84.88 | 0.575 | 0.575 |
| volle Fusion | **86.62** | 86.57 | 0.665 | 0.682 |

**Der View-Count ist auf MI3DOR fast wirkungslos.** DINOv2 allein verliert bei 8 Views 1,07 NN,
die Kaskade 0,48 NN, und die volle Fusion ist praktisch identisch (86.62 vs 86.57). OSCARs
8-View-Konfiguration war also **kein Handicap** — anders als auf SHREC, wo mehr Views
durchgehend halfen (A2/A7).

*Zur Einordnung gegen die Publikation:* Pullis Evaluator wendet die CLIP-Shortlist **nicht** auf
das Ranking an (`keep` fließt nirgends in `sims_full`), ihre Zahlen sind also reines
DINOv2-Retrieval über eine **andere Gallery** (1817 Objekte × 1 View statt 3848 × 42). Ein
direkter Zahlenvergleich ist damit nicht zulässig; wir reproduzieren den **Mechanismus**, nicht
ihre Messung.

---

## 5. Kategorien-Analyse — Top-1-Treffer je Kanal

Per-Kategorie-NN (Anteil korrekter Top-1-Treffer), alle 21 Kategorien mit je 500 Queries:

| Kategorie | text | view | shape | Fusion | bester Einzelkanal |
|---|---|---|---|---|---|
| car | 0.976 | **1.000** | 1.000 | 1.000 | view |
| guitar | 0.986 | **1.000** | 0.998 | 1.000 | view |
| airplane | 0.988 | 0.988 | **1.000** | 0.992 | shape |
| plant | 0.906 | **0.968** | 0.640 | 0.990 | view |
| bed | 0.892 | **0.978** | 0.858 | 0.976 | view |
| camera | 0.346 | 0.928 | **0.990** | 0.974 | shape |
| tent | 0.880 | 0.772 | **0.966** | 0.972 | shape |
| motorcycle | 0.768 | 0.868 | **0.974** | 0.964 | shape |
| rifle | 0.860 | **0.910** | 0.878 | 0.950 | view |
| keyboard | 0.728 | **0.922** | 0.832 | 0.942 | view |
| chair | 0.726 | 0.884 | **0.886** | 0.926 | shape |
| bicycle | 0.696 | **0.962** | 0.804 | 0.916 | view |
| wardrobe | 0.502 | **0.906** | 0.890 | 0.906 | view |
| knife | 0.630 | 0.806 | **0.846** | 0.896 | shape |
| flower_pot | 0.570 | 0.772 | **0.814** | 0.842 | shape |
| stairs | 0.494 | 0.774 | **0.778** | 0.834 | shape |
| **bookshelf** | 0.614 | **0.860** | 0.540 | 0.788 | view |
| **radio** | 0.516 | 0.636 | **0.684** | 0.756 | shape |
| **pistol** | 0.360 | **0.736** | 0.484 | 0.670 | view |
| **monitor** | **0.666** | 0.584 | 0.124 | 0.602 | text |
| **vase** | 0.166 | 0.182 | **0.416** | 0.284 | shape |

### Die Spezialisierung ist auch ohne Tiefe systematisch
- **S_shape gewinnt in 10 Kategorien** — und teils dramatisch: `camera` 0.990 vs. 0.346 (text),
  `motorcycle` 0.974 vs. 0.868 (view), `vase` 0.416 vs. 0.182 (view). Selbst im cross-Modus,
  wo der Shape-Kanal insgesamt der schwächste ist, ist er bei einem Drittel der Klassen der
  **beste**.
- **S_view gewinnt in 10 Kategorien** — vor allem bei texturreichen, gut sichtbaren Objekten
  (bed, bicycle, plant, keyboard, rifle, wardrobe).
- **S_text gewinnt genau einmal**: `monitor` (0.666) — und dort bricht der Shape-Kanal auf
  **0.124** ein. Ein Monitor ist ein flaches Rechteck; ohne Tiefe ist er geometrisch praktisch
  nicht bestimmbar, aber sprachlich eindeutig.
- **`vase` ist der Ausreißer nach unten**: alle Kanäle schwach (0.166 / 0.182 / 0.416), erst die
  Fusion hebt auf 0.476. Vasen sind sprachlich generisch, visuell variabel und geometrisch
  ähnlich zu flower_pot.

### ⚠️ Die feste Gewichtung schadet in 9 von 21 Kategorien

Alle 21 Kategorien, nach Δ sortiert. Δ = volle Fusion − bester Einzelkanal (Metrik NN,
gepaart über dieselben 500 Queries je Klasse).

| Kategorie | text | view | shape | bester Einzelkanal | Fusion | Δ |
|---|---|---|---|---|---|---|
| vase | 0.166 | 0.182 | 0.416 | shape 0.416 | 0.284 | −0.132 |
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

Derselbe Effekt wie in Stage 1 (dort 12 von 20): wo **ein** Kanal dominiert, zieht die feste
Gewichtung ihn zu den schwachen herunter. Der Gesamtgewinn der Fusion ist ein **Mittelwert über
gegenläufige Effekte** — und der Befund reproduziert sich über zwei Datensätze und beide
Query-Modi.

---

## 6. Diskussionswürdige Punkte

1. **Der Shape-Kanal ist im cross-Modus der schwächste — und trotzdem unverzichtbar.** Isoliert
   FT 0.510 (gegen DINOv2 0.629), aber er hebt die Fusion um **+0.053 FT** und ist bei **10 von
   21 Kategorien der beste Einzelkanal**. Der Mittelwert verdeckt, wo er trägt.
2. **Die Kaskade kann Listen-Metriken strukturell nicht verbessern.** FT/ST/F1 aller
   Kaskaden-Arme sind identisch mit CLIP-Text allein, weil die ~20er-Shortlist die Top-C-Tiefe
   nicht füllen kann. Sie verbessert nur den Kopf (NN +16,9). Das ist ein **architektonisches**
   Argument gegen die Kaskade, kein Parameterproblem.
3. **τ = 0.37 greift praktisch nie:** bei **96,9 %** der Queries prunt der Schwellwert auf leer,
   es übernimmt der Top-20-Fallback. Die „Schwellwert-Kaskade" ist faktisch „CLIP-Top-20 →
   DINO". Konsistent mit SHREC (98,3 %).
4. **Die partial-vs-full-mesh-Antwort kippt mit dem Query-Modus** — die Referenz muss zur Natur
   der Query passen (§3). Das verbindet A4 und A5 zu einer Aussage statt zweier Einzelbefunde.
5. **Der View-Count ist auf MI3DOR fast wirkungslos** (V8 ≈ V42, volle Fusion 86.62 vs 86.57),
   anders als auf SHREC. Plausibel: bei Bild-Queries auf gerenderte Ansichten reichen wenige
   Blickwinkel, während die pc-Query auf SHREC von jeder zusätzlichen Partialansicht profitiert.
6. **Feste Gewichte kosten auch hier** — negativ in 9 von 21 Kategorien, bis −0.132. Zusammen
   mit Stage 1 (12 von 20) ist das ein über zwei Datensätze reproduzierter Befund und das
   stärkste Argument für adaptive Gewichte.
7. **BASE-Gewichte sind robust** (+0.005 zum Optimum), aber das Optimum verschiebt sich
   **text-lastig** (0.45/0.35/0.20). Wer im cross-Modus tunen wollte, müsste Text stärken und
   Shape schwächen — nicht View stärken, wie ein SHREC-Proxy nahegelegt hätte.
