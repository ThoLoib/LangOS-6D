# Vereinbarungen

*Laufendes Protokoll dessen, was zwischen Thomas und mir festgelegt wurde — wie gemessen,
wie berichtet, was in Ordnung geht. Neueste zuerst. Eine Zeile Begründung genügt; die
ausführliche Fassung mit Alternativen steht in `DECISIONS.md`, der Hergang in `AI_LOG.md`.*

**Zweck:** Verabredungen, die mitten in einer Sitzung getroffen werden, gehen zwischen den
Sitzungen verloren. Am 2026-09-01 war die inkrementelle Cache-Simulation in Stage 4 auf
eine Attrappe zusammengeschrumpft, weil die Absprache nirgends stand. Vor jedem Lauf
gegenlesen, ob er noch dem entspricht, was hier steht.

---

## 2026-09-01 (Nachtrag)

- **Gemessen wird der Pfad, den die Pipeline nimmt** — nicht ein dafür erfundener.
  `embed_clip` fehlte, weil ich nach einem `encode_text` griff, das es nie gab, statt
  `_encode_texts_batch` zu nehmen, das `load_descriptions` selbst aufruft. Nebenwirkung:
  meine Einzelstring-Schleife hätte eine lineare Skalierung gezeigt, die es wegen des
  Batchings gar nicht gibt (4,6 ms bei 16 wie bei 42 Views).
- **Die Onboarding-Kette ist mesh → render → partial → describe → embed → cache**,
  gemäß `docs/PREPROCESSING.md` §1. `partial` fehlte in allen Läufen bis zum 2026-09-01,
  weil `reuse_renders` die Kameramatrizen nicht mitkopierte. SYNC und VERIFY (rclone) bleiben
  draußen — Thomas: "Sync ist unwesentlich". Netzzeit, keine Eigenschaft der
  Pipeline.
- **Schritt 7 (Skalenbestimmung) fehlt in der Query-Kette bewusst** und wird als
  Auslassung benannt — er wurde als eigenständige Komponente verworfen und läuft auch
  in der Stage-3-Konfiguration nicht.
- **Jede Unterprozess-Stufe prüft ihr Ergebnis, nicht den Rückgabewert.** `render` zählt
  erzeugte Bilder, `partial` erzeugte Wolken. Beide hatten `rc=0` bzw. plausible Zeiten
  gemeldet und nichts produziert.
- **Ein Aufruf je Seite:** `scripts/stage4_onboarding.sh` und `scripts/stage4_query.sh`.
  Der Onboarding-Wrapper teilt sich auf Host (Blender) und Container (Rest) auf.

## 2026-09-01

- **Stage 4a simuliert einen inkrementellen Cache — und zwar echt.** Gemessen wird, was
  ein anhängender Cache tatsächlich täte: bestehenden Cache laden, den einen neuen
  Objekteintrag einfügen, zurückschreiben. Keine Attrappe mit einem Dummy-Tensor.
  *Warum:* das ist der Kern des Experiments — die Behauptung „Onboarding kostet O(1)"
  muss gegen die tatsächliche Schreibarbeit geprüft werden, nicht gegen eine erfundene.
- **Der Invalidierungsaufschlag wird gemessen, nicht geschätzt.** Ein echter Neuaufbau
  über eine Stichprobe der Gallery, hochgerechnet auf 1257 Objekte — nicht die Assembly
  mit warmen Caches (das war der Fehler im ersten Lauf, 6,1 s statt ~38 min).
- **Full-Mesh gegen Partial wird je Datensatz und je Query-Modus berichtet**, nie als
  Aggregat. *Warum:* T-LESS trägt 93 % des Abfalls, auf LM-O gewinnt Full-Mesh in beiden
  Modi — das Aggregat zeigt in zwei von drei Fällen in die falsche Richtung.
- **Ursache dieses Abstands ist der Domänenabgleich zwischen Query und Gallery**, belegt
  durch die pc-gegen-cross-Asymmetrie (−0.113 gegen −0.018), nicht der Farbfehler.
- **In den Ergebnisdokumenten steht kein Fehlerhergang.** Befunde und Zahlen dort;
  Fehlerbeschreibungen ausschließlich in `AI_LOG.md` und `DECISIONS.md`.
- **Vorhersagen vor dem Lauf festhalten.** Für den Farb-Neulauf notiert: GSO und YCB-V
  leicht besser, T-LESS unverändert, Gesamtbefund bleibt. *Warum:* verhindert, dass ein
  Ergebnis nachträglich zur Erklärung passend gemacht wird.
- **Stage 2 wird wegen der farblosen Meshes nicht neu gerechnet**, nur dokumentiert. Ein
  Fix hieße 3848 × 42 Teilwolken erzeugen plus kompletter Neulauf; die Schlussfolgerung
  („ohne Tiefe gehört Shape heruntergewichtet") kippt dadurch nicht.

## 2026-08-31

- **Stage 4 misst jeden Schritt einzeln**, I/O getrennt von Rechnung, kalt getrennt von
  warm. *Warum:* eine Zahl, die Modell-Ladezeit enthält, sagt nur, über wie viele Queries
  gemittelt wurde.
- **Stage 4a nimmt die 3b-Datenbank als Basis** (G_proxy, 1257) und onboardet die 59
  Ziel-CADs einzeln; ausgewertet wird die Verteilung über diese 59 Fälle.
  *Warum:* echte CADs unterschiedlicher Komplexität erzeugen die Streuung, die das
  Ergebnis interpretierbar macht.
- **Stage 4b misst bis zur Pose**, inklusive FoundationPose.
- **16 gegen 42 Views ist ein Kosten-Nutzen-Argument**, keine reine Latenzzahl — die
  Stage-1-Qualität (V16 0.5820 gegen V42 0.5868) steht in derselben Tabelle.
- **Der Sprachprompt kommt aus derselben Quelle wie der Textkanal**, nicht handgeschrieben.
  *Warum:* sonst vermischt sich eine Qualitätsfrage mit der Latenzmessung.
- **Gallery-Assembly prüft ihre eigene Deckung und bricht unter 95 % ab.**

## Ältere Festlegungen

- **Kein eingefrorener „Sieger" über die Stages hinweg** — Stage 2 und 3 berichten die
  *Spanne* der Konfigurationen, weil die Stage-1-optimale Wahl sich nicht überträgt.
- **Geometrie bleibt in der Pose-Pipeline aus** (Stage-3-Befund, alle vier Zellen).
- **τ = 0.37 und Mittelwert-Pooling sind fix**, nicht Teil der Ablationen.
- **Auf `tessa-pc` pushen, nie auf `main`. Nur committen, wenn danach gefragt wird.**

## 2026-09-03

- **Vollständiger Ergebnisüberblick jederzeit.** `docs/RESULTS_OVERVIEW.md` wird von
  `tools/results_overview.py` aus den echten Ergebnisordnern **generiert**, nie von Hand
  gepflegt, und nach jedem Lauf neu erzeugt. Er meldet fehlende Zellen selbst.
  *Warum:* mir war entgangen, dass Stage 1 die Zelle cross × full-mesh gar nicht hat —
  ausgerechnet die, die auf BOP gewinnt — und dass ein Lauf im falschen Verzeichnis lag.
- **Aktuelle Ergebnisse gehören sofort in die Docs**, nicht erst auf Nachfrage.
- **Per-Query-Dateien bleiben erhalten.** `results_per_query.json` trägt die GT-Kategorie
  je Query und ist die Grundlage für `tools/compare_arms_by_category.py`. Ohne sie lässt
  sich später nicht mehr sagen, in welchen Kategorien ein Modus besser war.
- **Nichts überschreiben.** Der Stage-1-Aggregator schreibt `best_config.json` und die
  Summary-Dateien aus *nur den gelaufenen* Armen neu. Nachträgliche Arme laufen deshalb
  in einen eigenen Ordner; danach werden ausschließlich die Arm-Verzeichnisse
  hinüberkopiert, mit Sicherung und Gegenprüfung.

## 2026-09-03 (Nachtrag)

- **Ergebnisse ohne Konfidenzintervalle und ohne Wilcoxon berichten** — jetzt und künftig.
  *Warum:* Thomas' Entscheidung; der Apparat wirkt für den Nutzen zu aufwendig und muss in
  der Verteidigung erklärt werden. In der Computer Vision ist er ohnehin unüblich.
- **Stattdessen: Δ plus die Bilanz gewonnener Queries.** Das ist eine rein beschreibende
  Zahl, kein Test, trennt aber weiterhin einen breiten Vorsprung von einem, der auf wenigen
  Ausreißern beruht — genau die Unterscheidung, an der der Uni3D-„Sieg" (1009:1027) und die
  Config-Umstellung (938:1064) als Nulleffekte erkennbar wurden.
- Die Skripte `paired_significance*.py` bleiben im Repo, ihre Ausgabe wird aber nicht mehr
  in die Ergebnisdokumente übernommen.
