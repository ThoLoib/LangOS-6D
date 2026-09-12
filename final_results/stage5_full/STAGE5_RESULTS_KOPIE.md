# Stage 5 — Greifen mit abgerufenen Proxy-Modellen

## Fragestellung: Was wird evaluiert, und warum?

Ein Roboter soll ein Objekt greifen, fuer das er **kein eigenes CAD-Modell** besitzt.
OSCAR+ liefert ihm aus einer Modell-Datenbank das aehnlichste bekannte Objekt (Stage 3);
auf diesem **Proxy** werden Pose und Griffpunkte berechnet, gegriffen wird das echte
Objekt. Stage 5 misst, ob das reicht — ob sich Retrieval-Qualitaet in Handlungserfolg
uebersetzt.

Evaluiert werden **beide Datenbank-Annahmen von Stage 3, getrennt**:

- **3b — katalogfremde Datenbank:** 1257 Haushalts-, Spielzeug- und Industrieobjekte
  (GSO, HouseCat6D, ITODD). Das Zielobjekt hat dort keine Verwandten; der Proxy ist
  zwangslaeufig ein Fremdobjekt.
- **3c — domaenennahe Datenbank:** zusaetzlich enthaelt die Datenbank Objekte derselben
  Domaene (z. B. andere Industrieteile derselben Serie); nur das exakte Modell des Ziels
  bleibt entfernt. Das Substitut kann also ein "Geschwisterobjekt" sein.

Der Vergleich beantwortet die zentrale Frage: Liegt die Grenze des Systems in der
**Pipeline** (Retrieval + Pose + Griffplanung) oder im **Angebot der Datenbank**?

Je Objekt wird das CAD verwendet, das der jeweilige eingefrorene Stage-3-Lauf **am
haeufigsten auf Rang 1** gesetzt hat (Quelle: `results_bop_stage3_v2/3b_cross` bzw.
`3c_cross`, `records.json`). Das ist die Mehrheitsantwort des Retrievals — **keine
manuelle Auswahl, keine Kuration**; Fehlgriffe des Retrievals fliessen ungefiltert ein.

## Simulation: Wie wird evaluiert?

PyBullet mit Franka-Panda-Arm (Zwei-Backen-Greifer, Oeffnung 5–80 mm). Das Objekt steht
**allein auf dem Tisch**; Kamera (Intrinsik und Blickrichtung) stammt aus einem realen
BOP-Frame des Objekts.

1. **Aufstellung:** kanonische Standpose — die stabilste Ruhelage (trimesh
   `compute_stable_poses`), deren horizontale Ausdehnung in den Greifer passt
   (≤ 78 mm). Je Instanz wird das Objekt deterministisch um Instanz-Index × 36° um die
   Hochachse gedreht → bis zu 10 verschiedene Aufstellungen pro Objekt.
2. **Wahrnehmung:** FoundationPose erhaelt das Sim-Rendering (RGB-D + Segmentmaske) und
   das **CAD unter Test** (eigenes Modell bzw. Proxy) und schaetzt die 6-DoF-Pose.
3. **Griffplanung:** ein antipodaler Sampler berechnet Griffkandidaten **auf dem CAD
   unter Test** (800 Kontaktpaare, Reibkoeffizient 0.5, Top-40 nach Qualitaet). Die
   Griffe werden mit der geschaetzten Pose in Weltkoordinaten transformiert.
4. **Ausfuehrung:** bis zu 5 Greifversuche je Trial. **Erfolg** = Objekt 15 cm gehoben,
   1 s gehalten und einen ±5-cm-Shake ueberstanden (ACRONYM-Kriterium). Massen: bekannte
   YCB-Werte, sonst 0.2 kg.

Drei Bedingungen laufen **gepaart auf identischen Aufstellungen**:

| Bedingung | CAD fuer Pose + Griffplanung | misst |
|---|---|---|
| `gt` | eigenes Modell des Objekts | Referenzdecke der Pipeline |
| `proxy` | 3b-Proxy (katalogfremd) | Proxy-Kosten, fremde Datenbank |
| `proxy3c` | 3c-Substitut (domaenennah) | Proxy-Kosten, domaenennahe Datenbank |

Zusaetzlich wird je Trial **D_sym** protokolliert: die symmetrische Oberflaechendistanz
zwischen dem Ziel in wahrer Pose und dem CAD unter Test in geschaetzter Pose (Geometrie-
und Posefehler in einem Mass).

### Metriken und Einordnung in die Literatur

Hauptmetrik ist die **Lauf-Erfolgsrate**: Ein Lauf (eine Aufstellung) gilt als Erfolg,
wenn einer von **bis zu fuenf** Griffkandidaten — in Ranglisten-Reihenfolge, Szene vor
jedem Versuch zurueckgesetzt — die drei Phasen Heben (15 cm, Objekt >= 5 cm mit),
Halten (1 s) und Schuettelprobe (±5 cm entlang x/y/z) besteht; das Kriterium folgt
ACRONYM (Eppner et al., ICRA 2021). Blockierte Anfahrten verbrauchen keinen Versuch;
der erste Erfolg beendet den Lauf.

Dieses Budget-Design entspricht der Praxis von Clearing-/Bin-Picking-Evaluationen —
**Contact-GraspNet** (Sundermeyer et al., ICRA 2021) und **Dex-Net 4.0** (Mahler et al.,
Science Robotics 2019) berichten Erfolg ueber wiederholte Picks und geben beide Sichten
an: die Rate pro Versuch und die Abschluss-/Budget-Metrik. Wir tun dasselbe:

| | gt | 3b-Proxy | 3c-Substitut |
|---|---|---|---|
| Lauf-Erfolg (Budget <= 5 Griffe) | 68 % (352/520) | 41 % (213/520) | 57 % (294/520) |
| Erfolg je ausgefuehrtem Griff | 40 % (352/880) | 25 % (213/856) | 29 % (294/1029) |
| schon der 1. Griff sass | 47 % (242/520) | 27 % (141/520) | 35 % (182/520) |

Die Zeile "1. Griff" entspricht dem Ein-Griff-pro-Trial-Protokoll der klassischen
Grasp-Synthesis-Arbeiten (**Dex-Net 2.0**, Mahler et al., RSS 2017; **6-DOF GraspNet**,
Mousavian et al., ICCV 2019) — sie ist in `trials.csv` als `first_succ` protokolliert.
Tatsaechlich ausgefuehrt werden im Mittel 1.6–2.0 Griffe je Lauf (von max. 5), da
erfolgreiche Laeufe meist sofort enden und gescheiterte oft weniger als fuenf
erreichbare Kandidaten haben.

## Objektauswahl

Alle 59 BOP-Zielobjekte (YCB-V 21, T-LESS 30, LM-O 8) mit **einem einzigen Gate:
physische Greifbarkeit** — kleinste Objektabmessung 20–78 mm (Greiferoeffnung 80 mm);
die Mug (81 mm) laeuft als dokumentierte, empirisch greifbare Ausnahme mit. Das ergibt
**52 Objekte** (16 YCB-V, 30 T-LESS, 6 LM-O). Ausgeschlossen: ycbv 1, 11, 16 und
lmo 5, 12 (zu breit), ycbv 17, 18 (zu flach). Die Qualitaet des verfuegbaren Proxys war
**ausdruecklich kein Auswahlkriterium**.

## Ergebnisse

520 gepaarte Aufstellungen × 3 Bedingungen = 1560 Trials.

| | gt | 3b-Proxy | 3c-Substitut |
|---|---|---|---|
| **gesamt** | 352/520 (**68 %**) | 213/520 (**41 %**, Δ −26.7 Pp.) | 294/520 (**57 %**, Δ −11.2 Pp.) |
| YCB-V | 100/160 (62 %) | 37/160 (23 %) | 66/160 (41 %) |
| T-LESS | 222/300 (74 %) | 154/300 (51 %) | 208/300 (69 %) |
| LM-O | 30/60 (50 %) | 22/60 (37 %) | 20/60 (33 %) |
| D_sym Median | 5.2 mm | 16.2 mm | 11.1 mm |

**Kernbefund: Das domaenennahe Substitut (3c) schlaegt den katalogfremden Proxy (3b) um
+15.6 Prozentpunkte** (57 % vs. 41 %) bei identischer Pipeline — der Katalog, nicht die
Methode, ist der begrenzende Faktor.

### Tabelle 1 — Greifen mit dem 3b-Proxy (katalogfremde Datenbank)

| Objekt | Name | CAD, auf dem geplant wurde | gt | Proxy | Δ Pp. |
|---|---|---|---|---|---|
| ycbv2 | cracker box | bottle-dettol_washing_machine (housecat6d) | 3/10 | 4/10 | +10 |
| ycbv3 | sugar box | Ultra_JarroDophilus (gso) | 9/10 | 2/10 | -70 |
| ycbv4 | tomato soup can | Don_Franciscos_Gourmet_Coffee_Medi (gso) | 10/10 | 0/10 | -100 |
| ycbv5 | mustard bottle | Nestle_Nesquik_Chocolate_Powder_Fl (gso) | 1/10 | 2/10 | +10 |
| ycbv6 | tuna fish can | Don_Franciscos_Gourmet_Coffee_Medi (gso) | 8/10 | 0/10 | -80 |
| ycbv7 | pudding box | Nestle_Pure_Life_Exotics_Sparkling (gso) | 9/10 | 0/10 | -90 |
| ycbv8 | gelatin box | Nestle_Pure_Life_Exotics_Sparkling (gso) | 8/10 | 0/10 | -80 |
| ycbv9 | potted meat can | Polar_Herring_Fillets_Smoked_Peppe (gso) | 4/10 | 0/10 | -40 |
| ycbv10 | banana | COAST_GUARD_BOAT (gso) | 6/10 | 3/10 | -30 |
| ycbv12 | bleach cleanser | bottle-dettol_washing_machine (housecat6d) | 9/10 | 6/10 | -30 |
| ycbv13 | bowl | cup-red (housecat6d) | 0/10 | 8/10 | +80 |
| ycbv14 | mug | cup-red_heart (housecat6d) | 10/10 | 10/10 | +0 |
| ycbv15 | power drill | Remington_TStudio_Hair_Dryer (gso) | 2/10 | 0/10 | -20 |
| ycbv19 | large clamp | HELICOPTER (gso) | 1/10 | 0/10 | -10 |
| ycbv20 | extra large clamp | HeavyDuty_Flashlight (gso) | 10/10 | 2/10 | -80 |
| ycbv21 | foam brick | Ecoforms_Plant_Container_QP6CORAL (gso) | 10/10 | 0/10 | -100 |
| tless1 | obj_01 | itodd obj 26 | 0/10 | 1/10 | +10 |
| tless2 | obj_02 | itodd obj 27 | 4/10 | 8/10 | +40 |
| tless3 | obj_03 | itodd obj 27 | 9/10 | 10/10 | +10 |
| tless4 | obj_04 | itodd obj 26 | 10/10 | 2/10 | -80 |
| tless5 | obj_05 | itodd obj 13 | 9/10 | 1/10 | -80 |
| tless6 | obj_06 | itodd obj 18 | 4/10 | 3/10 | -10 |
| tless7 | obj_07 | itodd obj 18 | 10/10 | 5/10 | -50 |
| tless8 | obj_08 | itodd obj 18 | 10/10 | 7/10 | -30 |
| tless9 | obj_09 | itodd obj 18 | 6/10 | 7/10 | +10 |
| tless10 | obj_10 | itodd obj 13 | 6/10 | 9/10 | +30 |
| tless11 | obj_11 | itodd obj 13 | 6/10 | 3/10 | -30 |
| tless12 | obj_12 | itodd obj 13 | 3/10 | 3/10 | +0 |
| tless13 | obj_13 | Ecoforms_Plant_Pot_GP9_SAND (gso) | 10/10 | 0/10 | -100 |
| tless14 | obj_14 | itodd obj 26 | 10/10 | 0/10 | -100 |
| tless15 | obj_15 | itodd obj 27 | 9/10 | 9/10 | +0 |
| tless16 | obj_16 | itodd obj 27 | 10/10 | 9/10 | -10 |
| tless17 | obj_17 | itodd obj 27 | 5/10 | 7/10 | +20 |
| tless18 | obj_18 | itodd obj 27 | 5/10 | 6/10 | +10 |
| tless19 | obj_19 | itodd obj 18 | 8/10 | 4/10 | -40 |
| tless20 | obj_20 | itodd obj 18 | 7/10 | 7/10 | +0 |
| tless21 | obj_21 | itodd obj 26 | 5/10 | 8/10 | +30 |
| tless22 | obj_22 | itodd obj 26 | 10/10 | 9/10 | -10 |
| tless23 | obj_23 | itodd obj 01 | 10/10 | 6/10 | -40 |
| tless24 | obj_24 | bottle-sanitizer_small_white (housecat6d) | 10/10 | 9/10 | -10 |
| tless25 | obj_25 | Avengers_Thor_PLlrpYniaeB (gso) | 5/10 | 0/10 | -50 |
| tless26 | obj_26 | Avengers_Thor_PLlrpYniaeB (gso) | 7/10 | 0/10 | -70 |
| tless27 | obj_27 | itodd obj 27 | 8/10 | 6/10 | -20 |
| tless28 | obj_28 | itodd obj 18 | 10/10 | 9/10 | -10 |
| tless29 | obj_29 | Avengers_Thor_PLlrpYniaeB (gso) | 9/10 | 0/10 | -90 |
| tless30 | obj_30 | BIA_Porcelain_Ramekin_With_Glazed_ (gso) | 7/10 | 6/10 | -10 |
| lmo1 | ape | Ortho_Forward_Facing (gso) | 0/10 | 3/10 | +30 |
| lmo6 | cat | Toysmith_Windem_Up_Flippin_Animals (gso) | 8/10 | 6/10 | -20 |
| lmo8 | driller | Thomas_Friends_Wooden_Railway_Port (gso) | 10/10 | 4/10 | -60 |
| lmo9 | duck | CHICKEN_RACER (gso) | 2/10 | 0/10 | -20 |
| lmo10 | eggbox | Toysmith_Windem_Up_Flippin_Animals (gso) | 10/10 | 8/10 | -20 |
| lmo11 | glue | bottle-cleansing_lotion_small (housecat6d) | 0/10 | 1/10 | +10 |
| **gesamt** | | | **352/520 (68 %)** | **213/520 (41 %)** | **-26.7** |

### Tabelle 2 — Greifen mit dem 3c-Substitut (domaenennahe Datenbank)

| Objekt | Name | CAD, auf dem geplant wurde | gt | Proxy | Δ Pp. |
|---|---|---|---|---|---|
| ycbv2 | cracker box | bottle-dettol_washing_machine (housecat6d) | 3/10 | 4/10 | +10 |
| ycbv3 | sugar box | Ultra_JarroDophilus (gso) | 9/10 | 2/10 | -70 |
| ycbv4 | tomato soup can | ycbv obj 01 | 10/10 | 0/10 | -100 |
| ycbv5 | mustard bottle | Nestle_Nesquik_Chocolate_Powder_Fl (gso) | 1/10 | 2/10 | +10 |
| ycbv6 | tuna fish can | ycbv obj 01 | 8/10 | 0/10 | -80 |
| ycbv7 | pudding box | ycbv obj 08 | 9/10 | 8/10 | -10 |
| ycbv8 | gelatin box | ycbv obj 07 | 8/10 | 3/10 | -50 |
| ycbv9 | potted meat can | ycbv obj 04 | 4/10 | 1/10 | -30 |
| ycbv10 | banana | lmo obj 09 | 6/10 | 3/10 | -30 |
| ycbv12 | bleach cleanser | bottle-dettol_washing_machine (housecat6d) | 9/10 | 6/10 | -30 |
| ycbv13 | bowl | cup-red (housecat6d) | 0/10 | 8/10 | +80 |
| ycbv14 | mug | cup-red_heart (housecat6d) | 10/10 | 10/10 | +0 |
| ycbv15 | power drill | lmo obj 08 | 2/10 | 8/10 | +60 |
| ycbv19 | large clamp | ycbv obj 20 | 1/10 | 4/10 | +30 |
| ycbv20 | extra large clamp | ycbv obj 19 | 10/10 | 7/10 | -30 |
| ycbv21 | foam brick | Ecoforms_Plant_Container_QP6CORAL (gso) | 10/10 | 0/10 | -100 |
| tless1 | obj_01 | tless obj 04 | 0/10 | 10/10 | +100 |
| tless2 | obj_02 | tless obj 17 | 4/10 | 6/10 | +20 |
| tless3 | obj_03 | tless obj 17 | 9/10 | 6/10 | -30 |
| tless4 | obj_04 | tless obj 15 | 10/10 | 6/10 | -40 |
| tless5 | obj_05 | tless obj 06 | 9/10 | 9/10 | +0 |
| tless6 | obj_06 | tless obj 05 | 4/10 | 7/10 | +30 |
| tless7 | obj_07 | tless obj 08 | 10/10 | 5/10 | -50 |
| tless8 | obj_08 | tless obj 07 | 10/10 | 0/10 | -100 |
| tless9 | obj_09 | tless obj 07 | 6/10 | 5/10 | -10 |
| tless10 | obj_10 | tless obj 06 | 6/10 | 7/10 | +10 |
| tless11 | obj_11 | tless obj 12 | 6/10 | 3/10 | -30 |
| tless12 | obj_12 | tless obj 11 | 3/10 | 6/10 | +30 |
| tless13 | obj_13 | tless obj 16 | 10/10 | 9/10 | -10 |
| tless14 | obj_14 | tless obj 15 | 10/10 | 10/10 | +0 |
| tless15 | obj_15 | tless obj 16 | 9/10 | 10/10 | +10 |
| tless16 | obj_16 | tless obj 17 | 10/10 | 9/10 | -10 |
| tless17 | obj_17 | itodd obj 27 | 5/10 | 7/10 | +20 |
| tless18 | obj_18 | tless obj 17 | 5/10 | 9/10 | +40 |
| tless19 | obj_19 | tless obj 20 | 8/10 | 6/10 | -20 |
| tless20 | obj_20 | tless obj 19 | 7/10 | 10/10 | +30 |
| tless21 | obj_21 | tless obj 22 | 5/10 | 9/10 | +40 |
| tless22 | obj_22 | tless obj 21 | 10/10 | 9/10 | -10 |
| tless23 | obj_23 | tless obj 20 | 10/10 | 9/10 | -10 |
| tless24 | obj_24 | tless obj 04 | 10/10 | 10/10 | +0 |
| tless25 | obj_25 | tless obj 26 | 5/10 | 7/10 | +20 |
| tless26 | obj_26 | tless obj 25 | 7/10 | 6/10 | -10 |
| tless27 | obj_27 | tless obj 20 | 8/10 | 3/10 | -50 |
| tless28 | obj_28 | tless obj 20 | 10/10 | 5/10 | -50 |
| tless29 | obj_29 | tless obj 26 | 9/10 | 4/10 | -50 |
| tless30 | obj_30 | BIA_Porcelain_Ramekin_With_Glazed_ (gso) | 7/10 | 6/10 | -10 |
| lmo1 | ape | Ortho_Forward_Facing (gso) | 0/10 | 3/10 | +30 |
| lmo6 | cat | lmo obj 01 | 8/10 | 4/10 | -40 |
| lmo8 | driller | ycbv obj 15 | 10/10 | 3/10 | -70 |
| lmo9 | duck | teapot-wooden_color (housecat6d) | 2/10 | 1/10 | -10 |
| lmo10 | eggbox | Toysmith_Windem_Up_Flippin_Animals (gso) | 10/10 | 8/10 | -20 |
| lmo11 | glue | bottle-cleansing_lotion_small (housecat6d) | 0/10 | 1/10 | +10 |
| **gesamt** | | | **352/520 (68 %)** | **294/520 (57 %)** | **-11.2** |

## Diskussion

1. **Der Katalog ist der Hebel.** Gleiche Pipeline, gleiche Aufstellungen: fremde
   Datenbank kostet 26.7 Pp., domaenennahe nur 11.2 Pp. T-LESS zeigt es am staerksten
   (51 % → 69 %): Geschwister-Substitute erreichen 61 % Erfolg, katalogfremde
   3c-Substitute nur 45 %.
2. **Die gt-Decke liegt bei 68 %.** Auch mit dem eigenen Modell scheitert ein Drittel
   der Trials (flache und schmale Sonderfaelle; LM-O nur 50 %). Die Proxy-Differenzen
   sind relativ zu dieser Decke zu lesen, nicht zu 100 %.
3. **Retrieval-Ausreisser schlagen ungefiltert durch.** Die Mehrheitsantwort ist
   manchmal grotesk (Actionfigur fuer tless 25/26/29, Pflanztopf fuer ycbv 21) und
   produziert 0/10-Bloecke. Ein einfacher Groessen-Sanity-Check vor der Griffplanung
   (Bounding-Box-Vergleich) wuerde die schlimmsten Fehlgriffe abfangen — Ausblick.
4. **Modelltreue ist nicht hinreichend und nicht notwendig.** ycbv 13 (Bowl): eigenes
   CAD 0/10, Proxy 8/10 — der becherfoermige Proxy induziert Griffe, die funktionieren,
   waehrend die Griffe auf dem eigenen flachen Modell scheitern. Auch tless 10 greift
   mit Proxy besser als mit eigenem Modell (9/10 vs. 6/10). Entscheidend ist, wo das
   Modell die Griffe platziert, nicht wie exakt es das Ziel abbildet.
5. **3c hilft nicht ueberall.** LM-O bleibt schwach (33 %), weil die Datenbank auch
   domaenennah kaum Formverwandte der LM-O-Objekte enthaelt. Und ycbv 4 zeigt eine
   Grenze der Substitutwahl: Das 3c-Substitut (102-mm-Dose) passt selbst nicht in den
   Greifer → 0/10. Substitutwahl muss Greifer-Constraints kennen.

## Daten und Reproduktion

- Ergebnisse: `final_results/stage5_full/trials.csv` (1560 Trials),
  `plan.json` (Objekte, Instanzen, Proxys), Kopien in `_s5_out/solo_full/`.
- Skripte: `grasping/stage_5_full.py` (kompletter Lauf: baut Plan + startet Serie),
  `grasping/stage_5.py` (Einzelserien per `--object`/`--proxy`),
  `grasping/stage_5_viz.py` (Video eines Greifversuchs):
  `--all --canonical --conditions gt,proxy,proxy3c`.
- Stage-3-Quellen: `object_retrieval/results_bop_stage3_v2/{3b_cross,3c_cross}/…/records.json`.
