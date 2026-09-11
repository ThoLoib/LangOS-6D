# Objektauswahl — Szenario 2: Einzelobjekt allein auf dem Tisch

Erstellt 2026-09-11. Szenariowechsel gegenüber `docs/STAGE5_PROTOCOL.md`: **kein
BOP-Clutter** — jedes Objekt steht allein auf dem Tisch, geplant wird auf dem **besten
häufigen Stage-3-Proxy** dieses Objekts (bester `dimDev` unter allen Proxys mit ≥ 8
Rang-1-Instanzen im eingefrorenen `3b_cross`-Lauf). Damit entfallen: Verdeckung,
Korridor-Blockaden, Szenen-/Instanzlisten und das Proxy-Anteils-Kriterium.

**Gates (neu):**
- **G1′ Greiferband:** 20 mm ≤ kleinste Bbox-Abmessung ≤ 78 mm. Die Untergrenze ist die
  Lehre aus Stage 5: flach aufliegende Objekte (< 20 mm Aufliegehöhe) sind für den
  Parallelgreifer ohne Vor-Manipulation ungreifbar — Schere (16 mm) und Marker (19 mm)
  scheiterten in *allen* Bedingungen inkl. wahrer Pose.
- **G3′ Proxy-Form:** bester häufiger Proxy mit dimDev ≤ 0.30 (Kern ≤ 0.20).
- **FP-Prior:** GT-Lauf-Median aus den echten BOP-Bildern (Spalte gtMed) als Anhaltspunkt;
  im Solo-Sim ist FP auf texturlosen Einzelobjekten der härteste Fall — die `gt`-Bedingung
  des Laufs ist der eingebaute Check.

## Die 20 Objekte

**Kern (dimDev ≤ 0.20):**

| # | Objekt | Bbox sortiert [mm] | gtMed | Proxy | dimDev | pMed |
|---|---|---|---|---|---|---|
| 1 | tless 30 | 51·80·80 | 1.5 | GSO-Ramekin | 0.14 | 5.0 |
| 2 | ycbv 2 cracker box | 72·164·213 | 2.8 | Coffeecake-Box | 0.09 | 5.0 |
| 3 | tless 1 | 35·35·61 | 1.1 | itodd obj_000024 | 0.14 | 5.5 |
| 4 | tless 4 | 40·40·78 | 1.3 | Germanium-Flasche | 0.10 | 5.5 |
| 5 | tless 19 | 47·66·76 | 1.3 | itodd obj_000018 | 0.13 | 6.6 |
| 6 | tless 21 | 43·77·79 | 1.6 | Android-Figur | 0.10 | 7.6 |
| 7 | tless 10 | 42·64·81 | 1.3 | itodd obj_000018 | 0.16 | 8.1 |
| 8 | tless 22 | 44·77·79 | 1.6 | Android-Figur | 0.10 | 8.2 |
| 9 | lmo 9 duck | 77·86·104 | 3.3 | CHICKEN_RACER | 0.14 | 11.7 |
| 10 | ycbv 14 mug¹ | 81·93·117 | 2.2 | cup-red_heart | 0.07 | 4.4 |

¹ mug reißt die 78-mm-Formalgrenze (81 mm), ist aber der empirisch beste Fall aus Stage 5
(Rand-/Henkelgriffe, Proxy 5/6) — behalten mit Fußnote.

**Erweitert (0.20 < dimDev ≤ 0.30, nach pMed geordnet):**

| # | Objekt | Bbox [mm] | gtMed | Proxy | dimDev | pMed |
|---|---|---|---|---|---|---|
| 11 | tless 5 | 54·59·95 | 1.5 | itodd obj_000018 | 0.25 | 4.9 |
| 12 | tless 25 | 61·62·96 | 1.3 | itodd obj_000018 | 0.27 | 5.0 |
| 13 | tless 18 | 64·99·99 | 1.5 | itodd obj_000013 | 0.26 | 5.8 |
| 14 | ycbv 5 mustard² | 67·97·191 | 1.6 | Nesquik-Dose | 0.25 | 6.8 |
| 15 | tless 28 | 48·100·100 | 1.6 | itodd obj_000018 | 0.26 | 6.8 |
| 16 | ycbv 3 sugar box | 50·94·176 | 2.2 | Nesquik-Dose | 0.26 | 7.0 |
| 17 | tless 24 | 43·43·81 | 1.4 | Sanitizer-Flasche | 0.26 | 8.3 |
| 18 | tless 9 | 63·78·121 | 1.8 | itodd obj_000018 | 0.29 | 8.7 |
| 19 | tless 2 | 43·43·62 | 1.2 | Germanium-Flasche | 0.25 | 8.8 |
| 20 | tless 7 | 62·89·150 | 2.0 | itodd obj_000013 | 0.24 | 9.9 |

² mustard war im Clutter-Szenario Mechanismus-Exhibit (Nesquik-Achslage → Griffe in die
Nachbarn). Ohne Nachbarn ist genau das der interessante Test: bleibt der Achslagen-Effekt
auch frei stehend? — jetzt regulärer Kandidat.

**Reserve (falls Ausfälle):** lmo 10 eggbox (0.27/10.4), lmo 1 ape (0.28/12.9),
tless 6 (0.32/3.5), tless 11 (0.33/10.8).

## Ausgeschlossen — mit Grund

| Objekt | Grund |
|---|---|
| ycbv 17 scissors, ycbv 18 marker | **flach** (16/19 mm Aufliegehöhe) — in Stage 5 in allen Bedingungen 0 Treffer, auch mit wahrer Pose; braucht Vor-Manipulation/Sauger |
| ycbv 1 chef can, ycbv 11 pitcher, ycbv 16 wood block, lmo 5 can, lmo 12 holepuncher | **breit** (> 78 mm kleinste Abmessung) |
| tless 20 | FoundationPose scheitert im Sim schon mit eigenem CAD (Pilot 2026-09-06) — im Solo-Sim eher schlimmer (texturlos, symmetrisch, ohne Kontext) |
| lmo 11 glue | Bbox-Traumwert 0.05 täuscht: Pose-Median 64.5 mm |

## Eigenschaften der Liste

3× YCB-V + 1 Fußnote, 14× T-LESS, 1× LM-O (+2 LM-O Reserve). T-LESS dominiert erneut
(ITODD-Proxys passen strukturell); Vorsicht: texturlose Einzelobjekte sind FoundationPoses
härtester Fall — die `gt`-Bedingung je Objekt zuerst prüfen, bevor der volle Lauf startet.
Gegenüber der Clutter-Liste neu dabei: tless 1/2/4/5/24/25 und mustard (als regulärer
Kandidat); entfallen: scissors (flach) und die Szenen-/Anteils-Spalten (gegenstandslos).

Reproduktion: `grasp_solo.py` (Chat-Scratchpad) über `models_info.json`, `3b_cross`-Records
und Proxy-Meshes — liest nur vorhandene Dateien.
