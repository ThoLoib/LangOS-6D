#!/usr/bin/env python3
"""
repro_experiment.py — ein Aufruf pro Ergebnis aus final_results/RESULTS.md.

Flach und flag-basiert. Das Skript setzt alle "stillen" Lauf-Variablen
(SHREC_FORCE_PARTIAL_CACHE, SHREC_DINO_POOLING, STAGE1_GEOMETRY_BACKEND,
DGEDI_CACHE_DIR, PYTHONHASHSEED, ...) SELBST aus den Flags ab — der Aufrufer
muss keine Env-Variablen kennen (docs/REPRO_SPEC.md, Anforderung 1). Stufen,
die im oscar-Container laufen muessen, wrappt es automatisch in
`docker compose run`. Jeder Lauf schreibt seine volle Konfiguration als
run_config.json neben die Ergebnisse und druckt am Ende die Headline-Metrik.

Die vollstaendige Zuordnung Terminalzeile -> RESULTS.md-Zahl: docs/REPRODUCE.md.

Beispiele:
    python3 repro_experiment.py --stage 1 --arm E1c_full_fusion
    python3 repro_experiment.py --stage 1 --arm E2_chamfer_ransac      # Geometrie automatisch
    python3 repro_experiment.py --stage 2 --gallery partial
    python3 repro_experiment.py --stage 2 --gallery fullmesh --views 8 # OSCAR-Legacy
    python3 repro_experiment.py --stage 2 --sweep                      # Gewichtskarte
    python3 repro_experiment.py --stage 3 --mode 3a --query cross --gallery fullmesh
    python3 repro_experiment.py --stage 3 --mode 3b --query cross --gallery partial
    python3 repro_experiment.py --stage 3 --mode 3a --query cross --geo fitness
    python3 repro_experiment.py --stage 4 --side query --views 16,42
    python3 repro_experiment.py --stage 4 --side onboarding --stages render
Smoke (verkleinert, nur Funktionspruefung — Metriken NICHT vergleichbar):
    python3 repro_experiment.py --stage 1 --arm E1c_full_fusion --limit 25
    python3 repro_experiment.py --stage 3 --mode 3a --query cross --limit 20
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
IN_CONTAINER = os.path.exists("/.dockerenv")

# Partial-Caches (Fingerprints der Eval-Galerien; siehe docs/REPRO_SPEC.md P4)
CACHE = {
    "shrec_coloured": "object_images/shrec18_v2/.ulip_partial_cache_c3b88090d599c522.pt",
    "shrec_xyz":      "object_images/shrec18_v2/.ulip_partial_cache_641102dfbaf4e90c.pt",
    "shrec_uni3d":    "object_images/shrec18_v2/.ulip_partial_cache_eabcf9b9096553c9.pt",
    "mi3dor":         "object_images/MI3DOR/.ulip_partial_cache_f6bcf93bb6c92c68.pt",
}

# Stage-1-Arme: welcher Shape-Cache, ob Geometrie. Alles andere leitet der
# Treiber (experiments/experiment1_shrec18_stage1.py) aus dem Arm-Namen ab.
S1_FULLMESH = {"E2b_fullmesh", "E2b_fullmesh_geo", "E2b_fullmesh_shape_only",
               "E7_ulip2_cross_fullmesh", "E7_ulip2_cross_fullmesh_shape_only"}
S1_XYZ      = {"O5_xyz_only", "O5_xyz_shape_only"}
S1_UNI3D    = {"E7_uni3d", "E7_uni3d_shape_only"}
S1_GEOMETRY = {"E2_fitness", "E2_chamfer_ransac", "E2_both",
               "O1c_gedi_post_fusion", "O1e_gedi_with_base", "E2b_fullmesh_geo"}

# Headline-Referenzwerte aus final_results/RESULTS.md (zum Sofort-Abgleich)
EXPECTED = {
    ("1", "E1c_full_fusion"):    "nDCG 0.5868 / hit@1 0.341",
    ("1", "E2_chamfer_ransac"):  "nDCG 0.6405 / hit@1 0.472",
    ("1", "E2b_fullmesh"):       "nDCG 0.5935 / hit@1 0.360",
    ("1", "E2b_fullmesh_geo"):   "nDCG 0.6417 / hit@1 0.481",
    ("2", "partial"):            "NN 88.44 / FT 0.6918 (Arm clip_dino_ulip_full)",
    ("2", "fullmesh"):           "NN 86.57 / FT 0.6818 (Arm clip_dino_ulip_full)",
    ("3", "3a-cross-partial"):   "R@1 0.4818",
    ("3", "3a-cross-fullmesh"):  "R@1 0.5151",
    ("3", "3b-cross-partial"):   "D_sym Median 18.37 mm",
    ("3", "gt"):                 "D_sym Median 1.72 mm",
}


def log(m):  print(f"[repro] {m}", flush=True)
def die(m):  sys.exit(f"[repro] ABBRUCH: {m}")


def run(cmd, env_extra=None, cwd=None):
    env = dict(os.environ); env.update({k: str(v) for k, v in (env_extra or {}).items()})
    log("$ " + " ".join(shlex.quote(str(c)) for c in cmd)
        + ("" if not env_extra else "   [env: " + " ".join(
            f"{k}={v}" for k, v in env_extra.items()) + "]"))
    rc = subprocess.call([str(c) for c in cmd], env=env, cwd=cwd or ROOT)
    if rc != 0:
        die(f"Treiber endete mit rc={rc}")


def reexec_in_container(env_extra):
    cmd = ["docker", "compose", "run", "--rm", "oscar"]
    for k, v in env_extra.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["python3", "/app/repro_experiment.py"] + sys.argv[1:]
    log("laeuft im oscar-Container — wrappe automatisch.")
    os.execvp("docker", cmd)


def check_dgedi(expected_n, cache_hint):
    """dGeDi-Dienst muss laufen UND die richtige Gallery geladen haben."""
    for url in ("http://localhost:5061/health", "http://dgedi:5061/health"):
        try:
            h = json.load(urllib.request.urlopen(url, timeout=5))
            n = h.get("n_gallery", -1)
            if n == expected_n:
                log(f"dGeDi ok: n_gallery={n}")
                return
            die(f"dGeDi laeuft, hat aber n_gallery={n} statt {expected_n}. "
                f"Umschalten: DGEDI_CACHE_DIR={cache_hint} docker compose up -d "
                "--force-recreate dgedi   — dann erneut starten.")
        except die.__class__:
            raise
        except Exception:
            continue
    die("dGeDi-Dienst nicht erreichbar. Start: docker compose up -d dgedi "
        f"(mit DGEDI_CACHE_DIR={cache_hint}).")


def write_config(outdir, args, env_extra):
    os.makedirs(outdir, exist_ok=True)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    json.dump({"argv": sys.argv[1:], "args": vars(args), "env": env_extra,
               "git": rev, "time": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(outdir, "run_config.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["1", "2", "3", "4"])
    # Stage 1
    ap.add_argument("--arm", help="Stage 1: Arm-Name wie in RESULTS.md 1.8 "
                                  "(z.B. E1c_full_fusion, E2_chamfer_ransac)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Smoke: Stage 1 = Query-Limit, Stage 3 = Target-Limit, "
                         "Stage 2 = Queries je Kategorie, Stage 4 = n Queries")
    ap.add_argument("--subset", default="",
                    help="Stage 1: JSON-Datei mit Query-IDs, z.B. "
                         "object_retrieval/shrec18_splits/test_split_ids.json "
                         "fuer den Track-Vergleich (RESULTS.md 1.2, n=649)")
    # Stage 2/3
    ap.add_argument("--gallery", choices=["partial", "fullmesh"], default="partial",
                    help="Shape-Gallery-Repraesentation (Stage 2 und 3)")
    ap.add_argument("--views", default="",
                    help="Stage 2: 8 fuer den OSCAR-Legacy-Vergleich (Default 42). "
                         "Stage 4: Kommaliste, Default 16,42")
    ap.add_argument("--sweep", action="store_true",
                    help="Stage 2: Gewichtskarte (231 Punkte) statt Hauptlauf")
    ap.add_argument("--mode", choices=["3a", "gt", "3b", "3c"],
                    help="Stage 3: Retrieval / GT-Referenz / Proxy-Pose / Zerlegung")
    ap.add_argument("--query", choices=["cross", "pc"], default="cross",
                    help="Stage 3: Query-Modus des Shape-Kanals")
    ap.add_argument("--oscar-baseline", action="store_true",
                    help="Stage 3: OSCAR-Kaskade statt voller Fusion")
    ap.add_argument("--geo", choices=["", "distance", "fitness", "borda"], default="",
                    help="Stage 3: dGeDi-Re-Ranking der Top-5 mit diesem Signal")
    ap.add_argument("--gt-records",
                    default="object_retrieval/results_bop_stage3_v2/gt/combined_gt.json",
                    help="Stage 3 --mode 3b: Datei des GT-Laufs (fuer Delta je Instanz)")
    ap.add_argument("--from-3a", default="",
                    help="Stage 3 --mode 3c: 3a-Ergebnisordner als Quelle")
    # Stage 4
    ap.add_argument("--side", choices=["query", "onboarding"],
                    help="Stage 4: Anfrage- oder Onboarding-Seite")
    ap.add_argument("--stages", default="mesh,partial,describe,embed",
                    help="Stage 4 onboarding: Stufenliste; 'render' und 'dgedi' "
                         "laufen auf dem Host (Blender/dGeDi)")
    ap.add_argument("--shape-source", choices=["partial", "fullmesh"], default="partial",
                    help="Stage 4: Gallery-Repraesentation des Shape-Kanals")
    ap.add_argument("--geometry", action="store_true",
                    help="Stage 4 query: dGeDi-Re-Ranking (K=5) mitmessen")
    ap.add_argument("--out", default="", help="Ergebnisordner/-datei (sonst Default je Stage)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-docker", action="store_true",
                    help="nicht automatisch in den Container wrappen")
    args = ap.parse_args()

    # =======================================================================
    if args.stage == "1":
        if not args.arm:
            die("--arm fehlt (Liste: RESULTS.md 1.8 bzw. --list im Treiber).")
        env = {"PYTHONHASHSEED": "0", "SHREC_DINO_POOLING": "mean"}
        prefix = "/app/" if IN_CONTAINER else ""
        if args.subset:
            env["SHREC_QUERY_SUBSET"] = prefix + args.subset
        if args.arm in S1_FULLMESH:
            pass                                   # KEIN Partial-Cache erzwingen
        elif args.arm in S1_XYZ:
            env["SHREC_FORCE_PARTIAL_CACHE"] = prefix + CACHE["shrec_xyz"]
        elif args.arm in S1_UNI3D:
            env["SHREC_FORCE_PARTIAL_CACHE"] = prefix + CACHE["shrec_uni3d"]
        else:
            env["SHREC_FORCE_PARTIAL_CACHE"] = prefix + CACHE["shrec_coloured"]
        geo = args.arm in S1_GEOMETRY
        if geo:
            env.update({"STAGE1_GEOMETRY_BACKEND": "dgedi",
                        "DGEDI_CACHE_DIR": ".dgedi_gallery_shrec",
                        "GEOM_VOXEL": "0.02"})
            check_dgedi(3308, ".dgedi_gallery_shrec")
        if not IN_CONTAINER and not args.no_docker:
            reexec_in_container(env)
        out = args.out or "object_retrieval/results_repro_stage1"
        cmd = ["python3", "-u", "experiments/experiment1_shrec18_stage1.py",
               "--ablations", args.arm, "--results-root", out]
        if geo:
            cmd += ["--with-geometry", "--geom-k", "50"]
        if args.limit:
            cmd += ["--limit-queries", args.limit]
        if args.overwrite:
            cmd += ["--overwrite"]
        write_config(os.path.join(ROOT, out), args, env)
        run(cmd, env_extra=env)
        ms = os.path.join(ROOT, out, args.arm, "metrics_summary.json")
        if not os.path.isfile(ms):
            die(f"kein {ms} — Lauf unvollstaendig.")
        d = json.load(open(ms))
        log(f"ERGEBNIS {args.arm}: nDCG {d['metrics']['nDCG']:.4f} / "
            f"hit@1 {d['metrics_depth']['NN_sub']:.4f}"
            + (f"   (RESULTS.md: {EXPECTED[('1', args.arm)]})"
               if ("1", args.arm) in EXPECTED and not args.limit else "")
            + ("   [SMOKE — nicht vergleichbar]" if args.limit else ""))
        return

    # =======================================================================
    if args.stage == "2":
        env = {"PYTHONHASHSEED": "0", "MI3DOR_DINO_POOLING": "mean",
               "MI3DOR_NUM_VIEWS": args.views or "42",
               "MI3DOR_MODES": args.gallery,
               "MI3DOR_RESULT_FOLDER": args.out or "results_repro_stage2"}
        if args.limit:
            env["MI3DOR_MAX_QUERIES_PER_CAT"] = str(args.limit)
        if args.gallery == "partial":
            env["SHREC_FORCE_PARTIAL_CACHE"] = (
                ("/app/" if IN_CONTAINER else "") + CACHE["mi3dor"])
        if not IN_CONTAINER and not args.no_docker:
            reexec_in_container(env)
        outdir = os.path.join(ROOT, "object_retrieval", env["MI3DOR_RESULT_FOLDER"])
        write_config(outdir, args, env)
        if args.sweep:
            run(["python3", "-u", "mi3dor_weight_sweep.py"],
                env_extra=env, cwd=os.path.join(ROOT, "object_retrieval"))
            log("Gewichtskarte fertig (CSV im Ergebnisordner).")
            return
        run(["python3", "-u", "retrieval_mi3dor_eval_oscarplus.py"],
            env_extra=env, cwd=os.path.join(ROOT, "object_retrieval"))
        ms = os.path.join(outdir, args.gallery, "metrics_summary_topk_15.json")
        if not os.path.isfile(ms):
            die(f"kein {ms}.")
        v = json.load(open(ms))["variants"]["clip_dino_ulip_full"]
        log(f"ERGEBNIS volle Fusion ({args.gallery}): NN {v['NN_accuracy']:.2f} / "
            f"FT {v['FT_mean']:.4f}"
            + (f"   (RESULTS.md: {EXPECTED[('2', args.gallery)]})"
               if not args.limit and (args.views or '42') == '42' else "")
            + ("   [SMOKE]" if args.limit else ""))
        return

    # =======================================================================
    if args.stage == "3":
        if not args.mode:
            die("--mode fehlt (3a|gt|3b|3c).")
        env = {"PYTHONHASHSEED": "0"}
        if args.geo:
            env["STAGE3_GEO_SIGNAL"] = args.geo
            check_dgedi(1316, ".dgedi_gallery")
        if not IN_CONTAINER and not args.no_docker:
            reexec_in_container(env)
        out = args.out or "results_repro_stage3"
        cmd = ["python3", "-u", "eval_bop_pose.py", "--datasets", "all",
               "--mode", args.mode, "--output", out, "--seed", 0]
        if args.query == "pc":
            cmd += ["--pc-query"]
        if args.gallery == "fullmesh":
            cmd += ["--fullmesh"]
        if args.oscar_baseline:
            cmd += ["--oscar-baseline"]
        if args.geo:
            cmd += ["--dgedi", "--dgedi-repo", "--dgedi-top-k", 5]
        if args.mode == "3b":
            cmd += ["--gt-records", os.path.join("/app" if IN_CONTAINER else ROOT,
                                                 args.gt_records)]
        if args.mode == "3c":
            if not args.from_3a:
                die("--from-3a fehlt fuer --mode 3c.")
            cmd += ["--from-3a", args.from_3a]
        if args.limit:
            cmd += ["--max-targets", args.limit]
        outdir = os.path.join(ROOT, "object_retrieval", out)
        write_config(outdir, args, env)
        run(cmd, env_extra=env, cwd=os.path.join(ROOT, "object_retrieval"))
        comb = [p for p in os.listdir(outdir) if p.startswith("combined_")]
        if not comb:
            die("keine combined_*.json entstanden.")
        d = json.load(open(os.path.join(outdir, comb[0])))
        key = ("R@1 " + format(d.get("recall@1", float("nan")), ".4f")
               if "recall@1" in d else
               "D_sym Median %.2f mm" % d["dsym"]["d_sym_median"])
        ref = EXPECTED.get(("3", f"{args.mode}-{args.query}-{args.gallery}")
                           if args.mode != "gt" else ("3", "gt"))
        log(f"ERGEBNIS {args.mode}: {key}"
            + (f"   (RESULTS.md: {ref})" if ref and not args.limit
               and not args.oscar_baseline and not args.geo else "")
            + ("   [SMOKE]" if args.limit else ""))
        return

    # =======================================================================
    if args.stage == "4":
        if not args.side:
            die("--side fehlt (query|onboarding).")
        views = args.views or "16,42"
        if args.side == "query":
            env = {"PYTHONHASHSEED": "0"}
            if not IN_CONTAINER and not args.no_docker:
                reexec_in_container(env)
            out = args.out or ("results_stage4/repro_query"
                               + ("_fullmesh" if args.shape_source == "fullmesh" else "")
                               + ("_geo" if args.geometry else "") + ".json")
            cmd = ["python3", "-u", "experiments/experiment4_query_latency.py",
                   "--dataset", "ycbv", "--n-queries", args.limit or 50,
                   "--views", views, "--warmup", 2, "--seed", 0,
                   "--shape-source", args.shape_source, "--out", out]
            if args.geometry:
                cmd += ["--geometry", "--geo-k", 5]
            write_config(os.path.join(ROOT, "results_stage4"), args, env)
            run(cmd, env_extra=env)
        else:
            host_stages = set(args.stages.split(",")) <= {"render", "dgedi"}
            env = {"PYTHONHASHSEED": "0"}
            if not host_stages and not IN_CONTAINER and not args.no_docker:
                reexec_in_container(env)
            if host_stages and IN_CONTAINER:
                die("render/dgedi laufen auf dem HOST (Blender bzw. dGeDi-Dienst).")
            out = args.out or ("results_stage4/repro_onboarding"
                               + ("_fullmesh" if args.shape_source == "fullmesh" else "")
                               + ".json")
            cmd = ["python3", "-u", "experiments/experiment4_onboarding.py",
                   "--stages", args.stages, "--num-views", views,
                   "--shape-source", args.shape_source, "--out", out]
            if "render" not in args.stages:
                cmd += ["--reuse-renders"]
            if args.stages == "mesh,partial,describe,embed":
                cmd += ["--measure-invalidation", "--inv-sample", 15]
            write_config(os.path.join(ROOT, "results_stage4"), args, env)
            run(cmd, env_extra=env)
        p = os.path.join(ROOT, out)
        if not os.path.isfile(p):
            die(f"keine Ergebnisdatei {out}.")
        d = json.load(open(p))
        for v, blk in sorted(d.get("by_views", {}).items(), key=lambda kv: int(kv[0])):
            steps = blk.get("per_step", {})
            tot = sum(s.get("median", 0) for s in steps.values())
            log(f"ERGEBNIS {args.side} {v} Views: Summe der Median-Schritte "
                f"{tot:.2f} s ({len(steps)} Schritte)")
        return


if __name__ == "__main__":
    main()
