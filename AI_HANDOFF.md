# AI_HANDOFF — Stand Branch `eval_final` (2026-09-07)

## Wo das Projekt steht

Die Evaluation (Stage 1–4) ist **abgeschlossen**. Alle Ergebnisse sind konsolidiert in
`final_results/RESULTS.md` (+ vier HTML-Artefakte + Summary-Daten, gespiegelt auf
`gdrive:Masterthesis/OSCAR/final_results/`). Branch `tessa-pc` hält die vollständige
Historie der Evaluationsphase; `eval_final` ist der bereinigte Reproduzierbarkeits-Stand.

## Die zwei Einstiege

| | |
|---|---|
| `repro_preprocess.py` | generisches Preprocessing je Datensatz (render/partial/describe/embed/dgedi/check); Eval-Parameter als Defaults; fremde Datensätze via `--cad-dir/--id-mode` |
| `repro_experiment.py` | ein Aufruf je RESULTS.md-Ergebnis; setzt alle stillen Env-Schalter selbst, wrappt sich in den oscar-Container, schreibt `run_config.json`, druckt Headline + Referenzwert |

Zuordnung Terminalzeile → Ergebnis: **`docs/REPRODUCE.md`** ·
Datenbeschaffung: **`docs/DATASETS.md`** · jeder Konfigurationswert mit Quelle:
**`docs/REPRO_SPEC.md`** · Lauf-Historie: `docs/RUN_PROVENANCE.md`.

End-to-End-Lauf (eine Anfrage bis zur Pose): REPRODUCE.md §5
(`python3 -m pipeline.run_pipeline …`).

## Was auf diesem Branch entfernt wurde

Einmal-Bash-Skripte (`scripts/` komplett, `rendering/*.sh`), Scratch-/Debug-Dateien,
der tote `gedi`-HTTP-Dienst (ersetzt durch `dgedi`), Legacy-Evalskripte der
Vor-Stage-Ära, überholte Ergebnisordner (lokal + Drive). Die 11 finalen
Ergebnisordner mit `results_per_query`-Daten bleiben erhalten (Keep-Liste in
`docs/AGREEMENTS.md`, 2026-09-07).

## Betriebswissen (bleibt gültig)

- Zwei-Container-Architektur: `oscar` + `foundationpose` (HTTP, Port 5050, ICP-Fallback);
  `dgedi` (Port 5061) mit `DGEDI_CACHE_DIR` zur Gallery-Wahl — `repro_experiment.py`
  prüft die geladene Gallery vor jedem Geometrie-Lauf.
- SHREC/MI3DOR-Partialwolken existieren nur noch als `.ulip_partial_cache_*.pt`;
  die Erzwingung übernimmt `repro_experiment.py` (Arm-abhängig).
- Blender muss exakt 3.4.1 sein (3.3.x scheitert still mit rc=0).
- GPU-Jobs strikt sequenziell (24-GB-Karte).
- Reproduktionsläufe schreiben nach `results_repro_*`; Originalordner sind tabu.

## Offene Punkte

- Voller Identitätslauf je Stage als Belegkette (Stage 1 BASE läuft/lief zuerst;
  Stage 2/3 sind Stunden-Läufe — bei Bedarf über REPRODUCE.md starten).
- Deferred (docs/AGREEMENTS.md früherer Einträge): Vektorisierung des
  Partial-Pfads in `step5_shape_matching.py:1490` (würde die Stage-4-ulip-Zahl
  invalidieren — erst nach Abschluss aller Messungen), Kanal-Score-Persistierung
  (REPRO_SPEC Anforderung 4), dgedi-Onboarding n=3, Query-Latenz nur YCB-V.
