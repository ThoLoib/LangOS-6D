# Stage 4 — Onboarding- und Query-Latenz

**Status (2026-09-01):** Skripte vollständig und verifiziert, Messungen teilweise
vorläufig. Die Query-Seite ist fertig gemessen; auf der Onboarding-Seite stammen die
Einzelposten aus verschiedenen Läufen, weil drei Stufen erst am 01.09. gemessen werden
konnten. Ein vollständiger Durchlauf steht aus (siehe §5).

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
| io_load | 9,0 ms | 17,2 ms |
| **segment** | 229,6 ms | 228,7 ms |
| pointcloud | 1,2 ms | 1,2 ms |
| encode_query | 37,5 ms | 37,7 ms |
| clip | 18,6 ms | 17,9 ms |
| **dino** | 293,0 ms | **537,4 ms** |
| ulip | 172,9 ms | 190,6 ms |
| fusion | 12,0 ms | 12,0 ms |
| **pose** | **1409,3 ms** | 1417,9 ms |
| **Ende zu Ende** | **2,194 s** | **2,471 s** |

Kaltstart einmalig: Gallery-Assembly 7,6 s + GroundingDINO/SAM 3,7 s.

**Die Pose dominiert** mit 57–64 % und ist der unruhigste Schritt: Median 1,4 s, IQR
1,1 s, p95 bis 4,7 s — FoundationPose' Hypothesenverfeinerung, kein Messrauschen. Der
einzige Posten, der mit der View-Zahl skaliert, ist DINO; alles andere ist konstant.

## 4. Onboarding — Einzelposten belastbar, Gesamtsumme vorläufig

| Stufe | 16 Views | 42 Views | n | skaliert mit V? |
|---|---|---|---|---|
| render (Blender) | 14,19 s | 34,96 s | 3 | ja, linear |
| describe (LLaVA) | 8,90 s ¹ | 11,68 s ¹ | 3 | ja, unterlinear |
| partial (HPR) | 1,95 s | 3,45 s | 3 | ja |
| embed_ulip | 0,61 s | 1,61 s | 3 | ja |
| embed_dino | 0,11 s | 0,28 s | 3 | ja |
| embed_clip | 4,6 ms | 4,7 ms | 3 | **nein** (Batch) |
| cache_load + save | 0,23 s | 0,24 s | 3 | **nein** (Gallery-Größe) |
| cache_insert | 0,04 ms | 0,1 ms | 3 | — |
| **Summe** | **≈ 26,0 s** | **≈ 52,2 s** | | **16 Views = 50 %** |

¹ Im 59-Objekt-Lauf lag `describe` bei 10,18 s (V16) und 18,12 s (V42) — mehr Objekte,
kälteres LLaVA. Für den Spaltenvergleich unerheblich, für die absolute Zahl gilt der
größere Lauf.

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
| Was der aktuelle Fingerprint erzwingt (1257 Objekte neu) | **≈ 34,3 min** |

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
| Onboarding je CAD | **50 %** | 100 % | 0.5820 vs 0.5868 |
| Query, nur Retrieval | 66 % | 100 % | |
| Query, Ende zu Ende | 89 % | 100 % | |

**Onboarding ist der Hebel, nicht die Query.** Ein neues Objekt kostet bei 16 Views die
Hälfte — für 0.005 nDCG, und Stage 1 zeigt die Qualitätskurve ab 16 Views flach (V32
liegt mit 0.5800 sogar *unter* V16). Ende zu Ende schrumpft der Vorteil auf 11 %, weil
die konstante Pose-Zeit ihn verdünnt.

## 6. Offen

- **Ein vollständiger Onboarding-Lauf** über alle 59 CADs mit der kompletten Kette
  (`mesh,partial,describe,embed`) und Render auf dem Host. Die Einzelposten oben stammen
  aus mehreren Läufen; die Gesamtsumme ist deshalb zusammengesetzt, nicht gemessen.
- **`--dgedi`** ist implementiert, aber nicht gelaufen. Nur relevant, wenn geometrisches
  Re-Ranking benutzt wird — was Stage 3 für BOP gerade widerlegt hat.
- **Nur ycbv** auf der Query-Seite. T-LESS und LM-O würden zeigen, ob die Segmentierung
  auf texturlosen Objekten teurer wird.
