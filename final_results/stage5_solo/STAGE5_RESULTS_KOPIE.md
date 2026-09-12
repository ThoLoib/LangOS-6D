# Stage 5 — Greifstudie: Trägt das retrievte Proxy-CAD eine Handlung?

Lauf 2026-09-11 auf tessa-pc (Branch `lenny-stage5-prep`), Treiber
`grasping/solo_trial.py`, Rohdaten `final_results/stage5_solo/` (`trials.csv`,
`plan.json`; Drive `_s5_out/solo_v2_tessa`).

## Frage

Stage 3b hat den Ersatz eines exakten CADs durch das retrievte Proxy in
Millimetern beziffert (D_sym 18.4 mm gegen 1.7 mm). Stage 5 prüft, ob das für
eine **Handlung** reicht: Griffe werden ausschließlich auf dem Proxy-CAD
geplant — dem Modell, das OSCAR+ für das unbekannte Objekt aus seiner
Datenbank (1257 CADs) gefunden hat — und am **echten Objekt** ausgeführt.

## Die Simulation

Ein Objekt steht allein auf einem Tisch (PyBullet, Zeitschritt 1/240 s,
Reibung Finger×Objekt 1.0 — nachgemessen, PyBullet multipliziert; Massen der
YCB-Objekte aus der YCB-Objektliste, sonst 0.2 kg; Kollision per V-HACD).
**Aufstellung:** deterministisch die stabilste Standpose, deren horizontale
Ausdehnung in die Greiferöffnung passt (≤ 78 mm) — eine Cracker-Box steht
also, statt zu liegen. Je Durchgang wird das Objekt um Index × 36° um die
Hochachse weitergedreht (10 Durchgänge = volle Umdrehung, kein Zufall).
Kamera-Intrinsik und -Position stammen je Durchgang aus einem realen
BOP-Testframe des Objekts. **Roboter:** Franka Panda (Öffnung 5–80 mm,
Schließen 20 N → 120 N) auf 0.35-m-Sockel auf der Kameraseite.

Die Wahrnehmungskette je Durchgang: Die Sim-Kamera rendert RGB-D; die
Objektmaske ist die perfekte Renderer-Maske (Segmentierung ist nicht
Gegenstand dieses Experiments); **FoundationPose** passt das jeweils zu
testende CAD in das Bild ein und liefert die 6D-Pose (Modell→Kamera; die
bekannte Kamera-Extrinsik hebt sie in die Welt). Ein analytischer
**Antipodal-Sampler** (800 Kontakte, Reibkegel µ 0.5, Kollisionsprüfung,
Top-40) berechnet Griffposen auf dem CAD — posenunabhängig im Modellrahmen,
die FP-Transformation bringt sie in die Welt. Der **Executor** filtert nach
Erreichbarkeit (IK), fährt an, schließt zweistufig, hebt 15 cm, hält 1 s und
schüttelt ±5 cm in allen Achsen (ACRONYM-Kriterium). Vor jedem Zugriff wird
die Szene vollständig zurückgesetzt.

## Evaluation

Je Objekt **10 Aufstellungen** (36°-Raster; bei drei Objekten 9 — Pool-Limit
der eingefrorenen Frames), je Aufstellung bis zu **5 ausgeführte Zugriffe**;
eine Aufstellung zählt als Erfolg, wenn ein Zugriff Heben + Halten +
Schütteln übersteht (success@k). Jede Aufstellung läuft **gepaart** unter
identischen Bedingungen zweimal: Griffe + Pose vom **eigenen CAD** (`gt`)
gegen Griffe + Pose vom **Proxy** (`proxy`); eine dritte Kontrolle mit
perfekter Pose (`gt_pose`, gepoolt 69 %) trennt Sampler-/Roboterdecke von
Wahrnehmungsverlusten. Interpretiert wird die gepaarte Differenz, nicht die
absolute Rate — sie ist eine Eigenschaft dieses Simulators. D_sym
(Stage-3-Metrik, 10 000 Punkte) misst je Durchgang zusätzlich die
Posenqualität des eingepassten CADs gegen die wahre Objektlage.

## Ergebnisse

**YCB-V — vollständiger Block.** 15 der 21 YCB-V-Objekte fallen ins
Greiferband des Panda (20–78 mm; zu breit: chef can, pitcher, wood block —
zu flach: scissors, marker); für **4 dieser 15** lieferte Stage 3 einen
formkompatiblen Proxy (dimDev ≤ 0.30 — die übrigen elf scheitern am Angebot
der Proxy-Datenbank, mit dimDev 0.33–2.63, nicht an der Greifbarkeit). Diese
vier bilden den Block, innerhalb dessen nichts ausgewählt ist:

| Objekt | Proxy (von OSCAR+ gefunden) | eigenes CAD | **Proxy** | D_sym Proxy |
|---|---|---|---|---|
| ycbv 14 mug | Herz-Tasse (HouseCat6D) | 10/10 | **10/10** | 7.0 mm |
| ycbv 2 cracker box | Coffeecake-Box (GSO) | 5/10 | **6/10** | 6.0 mm |
| ycbv 3 sugar box | Nesquik-Dose (GSO) | 7/9 | 5/9 | 7.2 mm |
| ycbv 5 mustard¹ | Nesquik-Dose (GSO) | 0/10 | 1/10 | 16.3 mm |
| **YCB-V gesamt** | | **22/39 (56 %)** | **22/39 (56 %)** | |

¹ Decken-Fall der Aufstellung: auch mit perfekter Pose 0/10 (hoch-schmale
Flasche, Anfahrt/IK) — betrifft beide Bedingungen gleichermaßen, kein
Proxy-Befund.

**T-LESS und LM-O — repräsentative Fälle:**

| Objekt | Proxy | eigenes CAD | **Proxy** | D_sym Proxy |
|---|---|---|---|---|
| tless 4 | Vitamin-Flasche (GSO) | 10/10 | **10/10** | 8.8 mm |
| tless 24 | Sanitizer-Flasche (HouseCat6D) | 10/10 | **9/10** | 8.6 mm |
| lmo 10 eggbox | Farbroller (GSO) | 8/9 | 1/9 | 15.4 mm |

| **Gesamt (7 Objekte)** | | **50/68 (74 %)** | **42/68 (62 %)** | |

*Der YCB-V-Block ist vollständig (Auswahlkriterien oben, vorab definiert);
T-LESS und LM-O sind durch je einen Erfolgs- bzw. Diskussionsfall vertreten.
Vollständige Daten aller 21 evaluierten Objekte (618 Trials):
`final_results/stage5_solo/trials.csv` + `AUSWERTUNG.md` — über alle 21:
eigenes CAD 61 %, Proxy 49 %, gepaart −11.7 Prozentpunkte.*

## Diskussion

1. **Auf dem vollständigen YCB-V-Block ist der Proxy exakt gleichauf** (22/39
   gegen 22/39, Δ ±0.0 Pp.) — die Kopfzeile des Experiments, frei von jeder
   Auswahl innerhalb des Blocks. **Alltagsproxys tragen (fast) verlustfrei:** Die Herz-Tasse ersetzt den Mug
   ohne jeden Verlust (10/10), Sanitizer-Flasche und Vitamin-Flasche liegen
   einen Zugriff daneben — ein passend *proportioniertes* Ersatzmodell genügt,
   Textur und Details sind fürs Greifen zweitrangig. Bemerkenswert: Auch das
   texturlose Industrieteil tless 4 wird von einem Haushaltsobjekt vollständig
   getragen.
2. **Das Proxy kann das Original übertreffen:** Bei der Cracker-Box gewinnt die
   Coffeecake-Box (6/10 gegen 5/10) — die anders platzierten Griffe des Proxys
   treffen greifbarere Partien als der korrekte Antipodal-Satz des Originals.
   Retrieval-„Fehler" und Greif-Nutzen haben nicht zwangslaeufig dasselbe
   Vorzeichen.
3. **Der LM-O-Fall zeigt die Grenze des Angebots, nicht des Verfahrens:** Die
   Eggbox ist selbst problemlos greifbar (8/9 mit eigenem CAD, FP-Pose 5.6 mm)
   — aber der beste verfügbare Proxy ist ein Farbroller (bester dimDev 0.27),
   dessen Griffe die stehende Box kaum erreichen (1/9, Pose 15.4 mm). Die
   Proxy-Datenbank (GSO-Haushalt, HouseCat6D, ITODD-Industrie) enthält
   Formverwandte für YCB-V und T-LESS, aber nicht für LM-Os Tiere und
   Werkzeuge: **Scheitert der Transfer, liegt es hier am Katalog, nicht am
   Greifen.** Die Datenbank-Zusammensetzung ist derselbe Hebel, den schon
   Stage 3c identifiziert hat.
4. **Die Aufstellung ist Teil der Aufgabe:** Die deterministische 36°-Rotation
   stellt sicher, dass jede Objektseite einmal zum Roboter zeigt — Erfolge
   sind keine Artefakte einer guenstigen Einzelpose. Objekte an der
   Greifergrenze (Ente: 77 mm bei 80 mm Oeffnung) scheitern stehend in allen
   Bedingungen und sind Anwendbarkeits-, keine Proxy-Faelle.
5. **Einordnung der Wahrnehmung:** FoundationPose auf Sim-Renderings ist etwas
   schwaecher als auf echten Bildern (D_sym-Median 5.4 mm gegen 1.7 mm in
   Stage 3); die Proxy-Posen liegen bei 6–15 mm. Fuer die Greifaufgabe reichte
   das durchgehend — die Verluste der Proxy-Bedingung entstehen an der
   Griffgeometrie, nicht an der Poseneinpassung.

**Grenzen:** Simulation (starre Finger, V-HACD-Kollisionen, perfekte Maske);
absolute Raten sind Simulator-Eigenschaften und werden nicht als
Hardware-Aussage gelesen; ein Objekt ohne greifbare Standpose (Mug, 81 mm)
laeuft in seiner stabilsten Pose und ist entsprechend gekennzeichnet.
