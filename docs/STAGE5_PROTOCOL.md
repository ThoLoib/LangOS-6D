# Stage 5 — Proxy-Greifstudie: vorab festgelegtes Protokoll

*Festgeschrieben am 2026-09-10, **vor** dem ersten Lauf des Rang-1–10-Sets. Alles, was ein
Ergebnis beeinflusst, steht hier oder in `grasping/experiment_proxy_grasp.py::PROTOCOL` (wird je
Lauf in `_s5_out/proxy_grasp/manifest.json` mitgeschrieben). Änderungen nach dem Lauf sind
Verstöße gegen dieses Protokoll und werden in `AI_LOG.md` als solche benannt.*

## Frage

Stage 3b hat gezeigt, was ein Ersatz-CAD in Millimetern kostet (D_sym-Median 18.4 mm gegen
1.7 mm mit dem exakten CAD). Stage 5 fragt, ob das für eine Handlung reicht: **Lassen sich
Griffe, die auf dem von OSCAR+ gefundenen Proxy-CAD geplant wurden, auf dem echten Objekt
ausführen?** Gemessen wird die Greif-Erfolgsrate in der Simulation — kein Beitrag zur
Greifplanung, sondern ein Nutzen-Test des Retrievals (Masterarbeit, Abschnitt
`subsec:eval_setup_grasping`).

## Objekte und Instanzen (fixiert)

- **Objektmenge:** Thomas' Top-20 vom 2026-09-10 (`grasping/proxy_grasp_cases.py`). Die
  Erfolgsrate wird über **Rang 1–10** berichtet; Rang 11–20 wird nur zugezogen, wenn die
  Fallzahl es braucht (`--ranks 1-20`). Mustard (ycbv 5) und tless 16 sind
  **Mechanismus-Exhibits** neben der Rate (`--exhibits`). Ausgeschlossen mit Begründung:
  ycbv 1 (102 mm > 80 mm Greifer, Anwendbarkeit aus der Geometrie), tless 20 (FoundationPose
  scheitert im Sim schon mit dem eigenen CAD, Pilot 2026-09-06), lmo 11 (Pose-Median 64 mm).
- **Proxy fixiert:** je Objekt genau das CAD, das der eingefrorene Arm (`3b_cross`, cross ×
  partial) am häufigsten auf Rang 1 gesetzt hat. Verwendet werden **nur die Instanzen, in
  denen dieser Proxy tatsächlich Rang 1 war** — sonst verwässert die Proxy-Lotterie die
  Stichprobe. Quelle: `results_bop_stage3_v2/3b_cross/<ds>_stage3b/records.json`, eingefroren
  in `grasping/proxy_grasp_instances.json` (1440 Instanzen, 25 Fälle, mit den Stage-3-Posen).
- **Ziehung je Objekt:** Instanzen mit `visib_fract ≥ 0.5` (BOP-Annotation; „stark verdeckt"
  bleibt draußen, Stage 3d), davon **6 je Objekt**: reihum über die Szenen (nach Szenen-ID),
  innerhalb einer Szene gleichmäßig über die sortierten Frames verteilt. Kein Zufall, kein
  Ergebnis fließt ein. Plan: `python3 grasping/experiment_proxy_grasp.py --plan` → `plan.json`.
  Rang 1–10: 10 Objekte × 6 Instanzen × 4 Bedingungen = **240 Trials**.

## Bedingungen (gepaart je Instanz; Voreinstellung `gt,proxy` — Thomas 2026-09-10: einfach ein Sim-Lauf mit den GTs und den zugehörigen Proxies; `gt_pose` und `random` sind Zusatzkontrollen über `--conditions`)

| Bedingung | CAD für Planung | Pose | misst |
|---|---|---|---|
| `gt_pose` | eigenes CAD des Ziels | wahre (gesetzte) Sim-Pose | Obergrenze von Sampler + Ausführung in dieser Szene |
| `gt` | eigenes CAD | FoundationPose | Verlust durch die Pose-Schätzung allein |
| `proxy` | **fixierter Proxy** (nativ, unskaliert — Schritt 7 ist verworfen) | FoundationPose | die implementierte Pipeline |
| `random` | ein per Hash gezogenes CAD aus G_proxy (1257) | FoundationPose | Zufalls-Baseline: was das Retrieval über ein beliebiges CAD hinaus bringt |
| `proxy_scaled` (optional) | Proxy auf die beobachtete Tiefenwolke skaliert | FoundationPose | Ablation: Anteil reiner Größenabweichung |

`random`: `crc32("random-proxy/<ds>/<obj_id>") mod 1257` über die sortierte Gallery-Liste — ein
CAD je Objekt, ohne Kategorienregel (es gibt kein konsistentes Label über GSO/HouseCat6D/ITODD),
ohne Testergebnis, ohne RNG-Zustand.

## Messkette (der Pfad, den die Pipeline nimmt)

1. **Szene:** der BOP-Frame wird in PyBullet nachgebaut — alle annotierten Objekte an ihren
   GT-Posen, Ziel dynamisch (V-HACD-Kollision, 0.2 kg, µ 1.6), Rest statisch. Weltrahmen
   (`--world auto`): **BOP-Extrinsics**, wo der Split sie hat (YCB-V, T-LESS; Tischhöhe =
   tiefster Objektvertex), sonst die **Tischebene per RANSAC aus dem echten Tiefenbild** (LM-O;
   Objektpixel maskiert, Fenster ±0.5 m um die Objekttiefen, 6 mm Schwelle, und nur Ebenen,
   über denen alle annotierten Objekte 0–40 cm hoch liegen). Die Ebene wird in jedem Fall
   gefittet und als Selbstprüfung mitgeschrieben: Winkel zur BOP-Aufwärtsachse
   (`plane_angle_deg`; YCB-V 1.9°, T-LESS 0.4–0.7° im Check), tiefster Objektvertex gegen den
   Tisch (`bottom_gap_mm`), Setz-Verschiebung des Ziels (`settle_mm`). **Roboter:** Franka Panda
   auf 0.35 m Sockel, **auf der Kameraseite** des Tisches, 0.55 m vom Objektschwerpunkt entlang
   der auf den Tisch projizierten Blickrichtung — als säße die Kamera am Roboter. Damit ist die
   erreichbare Seite eine Eigenschaft des Frames, nicht einer Weltachse.
2. **Pose:** FoundationPose bekommt das **echte Sensor-RGB-D und die GT-Sichtbarkeitsmaske**,
   exakt wie Stage 3 (`eval_bop_pose.estimate_pose`, 5 Refinement-Iterationen, mm-Meshes mit
   Skala 0.001). Bewertung mit **Stage 3s D_sym** (`stage3_metrics.d_sym`, 10 000 Punkte, Seed
   0) gegen die GT-Pose. Ziel-CAD = BOP `models_eval` (T-LESS, LM-O); YCB-V hat lokal kein
   `models_eval`, dort das texturierte YCB-Mesh (wird im Check gemeldet).
3. **Griffe:** analytischer Antipodal-Sampler auf dem posierten CAD (800 Oberflächenkontakte,
   µ 0.5, 4 Anfahrrichtungen, Kollisionsprüfung, NMS, Top-40, Seed 0; Greifer 5–80 mm).
   Kandidaten werden je CAD gecacht (`_grasp_cache/`), da sie objektfest sind.
4. **Ausführung:** Kandidaten nach Erreichbarkeit geordnet, IK-Filter (Vor- und Anfahrpose
   ≤ 30 mm Fehler), dann bis zu **5 ausgeführte Versuche**: Vorgriff 12 cm → Anfahrt → prüfen,
   ob der Greifer angekommen ist (> 3 cm Rest = `blocked`, zählt nicht als Versuch) →
   zweistufiges Schließen (20 N, dann 120 N) → 15 cm heben. **Vor jedem Versuch vollständiger
   Reset:** alle Objekte auf die gesetzte Ausgangspose, Arm in Ruhelage, Finger offen,
   Geschwindigkeiten null.
5. **Erfolg eines Versuchs:** Objekt ≥ 5 cm gehoben **und** 1 s (240 Schritte) in der Hand
   **und** ±5 cm-Rucke in x/y/z überstanden (ACRONYM-Schütteltest). Nur das Ziel ist dynamisch —
   ein anderes Objekt kann nicht „versehentlich" gehoben werden.
6. **Erfolg eines Trials (Kopfzahl):** mindestens einer der ≤ 5 Versuche erfolgreich
   (`success@k`). Daneben `success@1` (erster ausgeführter Kandidat) und die Rate je Versuch.

Berichtet wird **je Datensatz und gepoolt**, je Objekt, und **gepaart** (dieselbe Instanz unter
beiden Bedingungen): Δ in Prozentpunkten plus die Bilanz „nur A : nur B : beide : keine".
Keine Konfidenzintervalle, keine Tests (Vereinbarung 2026-09-03). Fehlerklassen je Trial:
`no_candidates`, `unreachable`, `all_blocked`, `grasp_failed`, `fp_error`. Ein Objekt, dessen
**eigenes** CAD keinen Kandidaten innerhalb des Greifers liefert, ist eine vorab erklärbare
Anwendbarkeits-Ausnahme, kein Fehlschlag.

## Korrekturen vor dem Lauf (Smoke-Test 2026-09-10, je eine Instanz von ycbv14/tless22/lmo9)

- Der **unbeschränkte** Ebenen-Fit wählte in YCB-V 48/1133 eine Wand (89° zur BOP-Achse, Objekte
  52 cm „über" dem Tisch) — daher die Objekt-Höhenbedingung im RANSAC und BOP-Extrinsics als
  Primärquelle, wo vorhanden. Der Check auf 48/1074 (1.9°) hatte das nicht gezeigt.
- Die Roboterposition „0.55 m entlang −x der Welt" machte dieselbe Griffmenge in der einen
  Weltkonvention erreichbar (Pilot: tless 22 Proxy 8 erreichbar) und in der anderen nicht (0 von
  18) — daher die Kameraseiten-Regel. Beide Änderungen betreffen die Mechanik, nicht die
  Bedingungen; sie wurden festgelegt, bevor ein Trial des Rang-1–10-Sets lief.

## Was die Studie nicht kann

Simulation, keine reale Hardware; statisches Clutter (LM-O-Clutter ohne Annotation fehlt im
Sim); GT-Masken statt Segmentierung (wie Stage 3); ein einfacher Sampler ohne Bewegungsplanung
— ein Ziel ohne freien Anfahrkorridor scheitert ehrlich als `all_blocked`. `gt_pose` nutzt die
gesetzte Sim-Pose und begrenzt daher auch Sampler und Ausführung, nicht nur die Pose.

## Vorhersagen (vor dem Lauf, 2026-09-10)

1. `gt_pose` erreicht ≥ 80 % success@k; Ausfälle stammen aus verdeckten Anfahrkorridoren
   und Greiferbreite (Cracker-Box 71 mm Kante, Ente 3 mm Luft).
2. `gt` liegt höchstens 10 Prozentpunkte unter `gt_pose` — der Pose-Verlust mit eigenem CAD
   ist klein (Stage-3-gt-Median 1.7 mm).
3. `proxy` liegt 15–30 Prozentpunkte unter `gt`. Der Verlust entsteht über **fehlende
   Kandidaten** (`n_reach`, `blocked`, `no_candidates`), nicht über schlechtere Griffe: die
   Rate je *ausgeführtem* Versuch bleibt innerhalb von 10 Prozentpunkten der `gt`-Rate.
4. Je Datensatz: T-LESS mit ITODD-Proxys am nächsten an `gt` (Industrieteil → Industrieteil,
   kleine dimDev); YCB-V dazwischen; LM-O (Ente ↔ CHICKEN_RACER, 3 mm Luft) am weitesten weg.
5. `random` ≤ 15 % success@k, überwiegend `no_candidates`/`unreachable`/`all_blocked`;
   D_sym-Median ≥ 30 mm.
6. D_sym-Mediane: `gt` ≈ 2 mm, `proxy` nahe den pMed-Werten der Auswahltabelle (4–12 mm).

Die Auswertung nach dem Lauf steht in `docs/STAGE5_RESULTS.md` (wird aus `REPORT.md` erzeugt).
Quellen der verwendeten Verfahren: Referenzliste in `grasping/README.md` (im Code als [Rn]).
