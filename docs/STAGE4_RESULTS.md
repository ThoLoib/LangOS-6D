# Stage 4 — Onboarding- und Query-Latenz

**Status (2026-09-04):** vollständig gemessen. Beide Seiten in einem zusammenhängenden
Lauf (`scripts/run_stage4_full.sh`), Onboarding über alle 59 Ziel-CADs, Query-Latenz über
50 Anfragen je View-Zahl. Die früheren, aus Teilläufen zusammengesetzten Zahlen sind damit
ersetzt.

Stage 1–3 beantworten *wie gut*. Stage 4 beantwortet *zu welchem Preis* — die Frage, die
ein Leser stellt, sobald der Genauigkeitsnachweis steht, und die darüber entscheidet, ob
das System einsetzbar ist.

## 1. Aufruf

```bash
bash scripts/stage4_onboarding.sh          # alle 59 CADs, 16 und 42 Views
bash scripts/stage4_query.sh               # ycbv, 50 Queries, 16 und 42 Views
```

Optionen mit `-h`. Nützlich: `-n` (weniger Objekte/Queries), `-v 42` (eine View-Zahl),
`--no-pose`, `--geometry`, `-d lmo`.

Beide geben dieselbe Tabellenform aus: je Schritt eine Zeile, die View-Zahlen als
Spalten, darunter die Gesamtzeit, eine Kosten-Nutzen-Zeile gegen die Stage-1-Qualität und
den Kaltstart getrennt ausgewiesen.

**Zwei Umgebungen.** Der Onboarding-Wrapper teilt sich auf: Blender liegt unter
`/home/tessa/Cap3D/…/blender-3.4.1-linux-x64/` und ist nicht ins Compose gemountet, läuft
also auf dem Host; Encoder und LLaVA brauchen den Container.

**Nicht mit `-n 4` messen.** Nach zwei Warmläufen bleiben dann zwei gewertete Messungen,
und ein Median aus zweien kippt bei einem einzelnen Ausreißer — ein Smoke-Test zeigte
prompt 16 Views langsamer als 42. Ab `-n 50` liegt die IQR unter 10 ms.

## 2. Was gemessen wird

**Onboarding** (Kette nach `PREPROCESSING.md` §1):

| Stufe | was |
|---|---|
| `mesh` | laden, Vertices verschweißen, Normalen, Durchmesser |
| `render` | Blender Cycles, V Views von Ikosphären-Vertices, FPS-geordnet |
| `partial` | Teilwolke je View aus Mesh + gespeicherter Kameramatrix (HPR) |
| `describe` | LLaVA, eine Beschreibung je View |
| `embed_dino` / `embed_clip` / `embed_ulip` | die drei Kanäle einzeln |
| `cache_load` / `cache_insert` / `cache_save` | das simulierte inkrementelle Anhängen |

SYNC und VERIFY (rclone auf Drive) sind bewusst nicht enthalten — Netzzeit, keine
Eigenschaft der Pipeline.

**Query** (Kette nach `run_pipeline.py`, Schritte 1–8 + B2):

`io_load` → `segment` (GroundingDINO + SAM2.1) → `pointcloud` → `encode_query` →
`clip` → `dino` → `ulip` → `fusion` → [`geometry`] → `pose` (FoundationPose).

**Schritt 7 (Skalenbestimmung) fehlt bewusst** — als eigenständige Komponente verworfen,
läuft auch in der Stage-3-Konfiguration nicht.

**Grundsätze.** CUDA-Kernel laufen asynchron, jede Messung synchronisiert vor und nach
dem Schritt. Berichtet werden Median, IQR und p95 statt Mittelwert und Standardabweichung,
weil Latenzverteilungen rechtsschief sind und für ein interaktives System der schlechte
Fall zählt. Kalt und warm sind getrennt: die Modelle einmal zu laden kostet ein Vielfaches
einer Query, eine gemischte Zahl sagt nur, über wie viele Queries gemittelt wurde.
I/O ist von Rechnung getrennt, sonst steckt in `embed_dino` die PNG-Dekodierung.

## 3. Query-Latenz — vollständig gemessen

ycbv, Gallery 1278, 50 Queries je View-Zahl, mit Pose, **0 Detektionsausfälle**:

| Schritt | 16 Views | 42 Views |
|---|---|---|
| io_load | 8,8 ms | 17,1 ms |
| **segment** (GroundingDINO + SAM2.1) | 230,6 ms | 242,1 ms |
| pointcloud | 1,3 ms | 1,3 ms |
| encode_query | 37,6 ms | 37,8 ms |
| clip | 19,8 ms | 18,7 ms |
| **dino** | 296,3 ms | **561,8 ms** |
| ulip | 176,7 ms | 219,4 ms |
| fusion | 12,0 ms | 12,0 ms |
| **pose** (FoundationPose) | **1402,0 ms** | 1487,0 ms |
| **Ende zu Ende** | **2,184 s** | **2,602 s** |

Kaltstart einmalig: Gallery-Assembly 8,0 s + GroundingDINO/SAM 4,6 s.

**Die Pose dominiert** mit 64 % bzw. 57 % und ist der unruhigste Schritt (p95 bis 4,7 s) —
FoundationPose' Hypothesenverfeinerung, kein Messrauschen. **Der einzige Posten, der mit der
View-Zahl skaliert, ist DINO**; alles andere ist konstant.

### Geometrisches Re-Ranking

Separat gemessen (K=5, ohne Pose, 5 von 5 Registrierungen erfolgreich je Anfrage):

| | |
|---|---|
| `geometry` (dGeDi + RANSAC + ICP) | **≈ 2,0 s** |
| Anteil an der Query ohne Pose | ~65 % |

Das ist etwa **das Doppelte der gesamten übrigen Kette ohne Pose** (≈1,05 s). Zusammen mit
dem Stage-3-Befund, dass Geometrie die Genauigkeit in allen vier gemessenen Zellen senkt, ist
die Sache von zwei Seiten entschieden: der teuerste Schritt ist zugleich der einzige, der
schadet.

> **Vorbehalt zur absoluten Zahl.** Eine erste Messung am 2026-09-01 ergab 5,45 s. Sie
> reproduziert nicht: zwei Messungen am 2026-09-04 liefern 1,84 s (n=25) und 2,07 s (n=12,
> dieselbe Stichprobengröße wie die erste). Dazwischen wurde der dGeDi-Container zweimal neu
> erzeugt; die alte Instanz lief seit über 26 Stunden unter Dauerlast. Das ist eine Vermutung,
> keine Erklärung — belastbar ist der reproduzierte Wert von ~2 s, und die Richtung der
> Aussage hängt nicht daran.

## 4. Onboarding — Einzelposten belastbar, Gesamtsumme vorläufig

| Stufe | 16 Views | 42 Views | skaliert mit V? |
|---|---|---|---|
| **render** (Blender, n=5) | 14,45 s | **34,68 s** | ja, linear |
| **describe** (LLaVA) | 10,25 s | 13,08 s | ja, unterlinear |
| partial (HPR) | 1,35 s | 2,76 s | ja |
| embed_ulip | 0,59 s | 1,55 s | ja |
| embed_dino | 0,12 s | 0,29 s | ja |
| embed_clip | 4,6 ms | 4,7 ms | **nein** (Batch) |
| mesh | 0,12 s | 0,10 s | nein |
| cache_load + save | 0,22 s | 0,23 s | **nein** (Gallery-Größe) |
| **Gesamt** | **27,18 s** | **52,97 s** | **16 Views = 51 %** |

n = 59 Ziel-CADs, Render auf 5 Objekten (Blender läuft auf dem Host). IQR über die 59 CADs:
0,96 s bei 16 Views, 2,10 s bei 42 — die Streuung über reale Meshes unterschiedlicher
Komplexität ist klein, das Onboarding ist gut vorhersagbar.

**Render und Beschreibung machen zusammen rund 90 % aus.** Das Encodieren, das man
intuitiv für den teuren Teil hält, sind unter 4 %.

**Zwei Posten skalieren nicht mit der View-Zahl.** `embed_clip` nicht, weil 16 oder 42
kurze Strings in einem Batch durch den Textencoder gehen und die GPU dabei nicht
ausgelastet ist. `cache_save` nicht, weil die Schreibkosten an der *Gallery-Größe*
hängen, nicht am neuen Objekt. Wer 16 statt 42 Views nimmt, spart beim Encoding, nicht
beim Cache.

### Inkrementell gegen Invalidierung

Der Cache-Schlüssel ist ein Fingerprint über das gesamte Inventar
(`_get_partial_cache_path`: je Objekt je View eine Zeile). Ein neues Objekt ändert den
Hash und invalidiert alles.

| | |
|---|---|
| Inkrementell (Encoding + Anhängen, ein Objekt) | **≈ 2,3 s** |
| Was der aktuelle Fingerprint erzwingt (1257 Objekte neu) | **≈ 34,7 min** |

Ein Faktor von rund 1000. Gemessen wurde über echte Stückkosten (1,635 s je Objekt bei
42 Views, hochgerechnet auf die Gallery), nicht geschätzt.

Bemerkenswert ist die Aufschlüsselung des Anhängens selbst: `cache_insert` ist mit
0,1 ms echtes O(1), aber `cache_load` + `cache_save` sind zusammen 292 ms und O(Gallery),
weil der Cache eine einzige monolithische `.pt`-Datei ist. **Selbst ein anhängender Cache
zahlt in dieser Ablageform pro neuem Objekt für die ganze Gallery** — nur Serialisierung
statt Encoding.

## 5. Der View-Handel

| | 16 Views | 42 Views | nDCG (Stage 1, O4) |
|---|---|---|---|
| Onboarding je CAD | **51 %** | 100 % | 0.5820 vs 0.5868 |
| Query, nur Retrieval | 66 % | 100 % | |
| Query, Ende zu Ende | 84 % | 100 % | |

**Onboarding ist der Hebel, nicht die Query.** Ein neues Objekt kostet bei 16 Views die
Hälfte — für 0.005 nDCG, und Stage 1 zeigt die Qualitätskurve ab 16 Views flach (V32
liegt mit 0.5800 sogar *unter* V16). Ende zu Ende schrumpft der Vorteil auf 11 %, weil
die konstante Pose-Zeit ihn verdünnt.

## 6. Offen

- **Nur ycbv** auf der Query-Seite. T-LESS und LM-O würden zeigen, ob die Segmentierung auf
  texturlosen Objekten teurer wird.
- Die Onboarding-Stufe `dgedi` (GeDi-Deskriptoren für ein neues Objekt) ist implementiert,
  aber nicht gelaufen — nur relevant, wenn geometrisches Re-Ranking benutzt wird, was Stage 3
  für BOP widerlegt hat.
