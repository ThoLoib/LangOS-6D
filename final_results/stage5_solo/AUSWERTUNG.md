# Solo v2 — kanonisch aufgestellt + deterministischer Yaw (2026-09-11)

591 Trials: 20 Objekte × bis zu 10 Aufstellungen (stabilste Standpose mit
horizontaler Ausdehnung ≤ 78 mm; je Instanz Index × 36° Yaw) × {gt_pose, gt,
proxy}. Ausnahme geloggt: ycbv14 mug hat keine ≤78-mm-Standpose (81 mm) —
stabilste Pose verwendet. Kamera je Instanz aus dem BOP-Frame; Objekt allein.

## Erfolgsrate je Objekt (gt / proxy / Kontrolle gt_pose)

| Objekt | gt | proxy | gt_pose | | Objekt | gt | proxy | gt_pose |
|---|---|---|---|---|---|---|---|---|
| tless4 | 10/10 | 10/10 | 10/10 | | tless21 | 5/10 | 6/10 | 9/10 |
| ycbv14 mug | 10/10 | 10/10 | 10/10 | | tless22 | 5/10 | 5/10 | 10/10 |
| tless24 | 10/10 | 9/10 | 10/10 | | ycbv3 sugar | 7/9 | 5/9 | 9/9 |
| tless28 | 10/10 | 8/10 | 9/10 | | tless25 | 7/10 | 1/10 | 10/10 |
| tless9 | 8/10 | 7/10 | 7/10 | | tless5 | 4/10 | 1/10 | 10/10 |
| tless7 | 10/10 | 0/10 | 10/10 | | tless10 | 3/10 | 0/10 | 2/10 |
| tless19 | 9/10 | 6/10 | 10/10 | | tless2 | 1/9 | 5/9 | 4/9 |
| tless30 | 9/10 | 4/10 | 0/10¹ | | tless1 | 0/9 | 7/9 | 0/9 |
| ycbv2 cracker | 5/10 | 6/10 | 6/10 | | ycbv5 mustard | 0/10 | 1/10 | 0/10² |
| tless18 | 4/10 | 7/10 | 9/10 | | lmo9 duck | 0/10 | 2/10 | 1/10² |

**Summen: gt 117/197 (59 %) · proxy 100/197 (51 %) · gt_pose 136/197 (69 %)**

¹ Wiederauftreten der gt>gt_pose-Anomalie in Extremform (0/10 mit wahrer Pose,
9/10 mit FP-Pose) — ungeklärt, betrifft die gt↔proxy-Paarung nicht.
² Regressions-Fälle der Aufstellregel: Ente und Senfflasche waren in der
BOP-Liegepose gut greifbar (10/10 bzw. 5/10) und sind es stehend nicht mehr
(hoch/schmal → Anfahrt/IK); die Regel optimiert Seitengreifbarkeit, nicht
Erreichbarkeit.

## Gepaart gt→proxy
| | n | gt | proxy | Δ |
|---|---|---|---|---|
| YCB-V | 39 | 56 % | 56 % | ±0.0 Pp. |
| T-LESS | 148 | 64 % | 51 % | −12.8 Pp. |
| LM-O | 10 | 0 % | 20 % | +20.0 Pp. |
| **gesamt** | 197 | **59 %** | **51 %** | **−8.6 Pp.** (Bilanz 48:31:69:49) |

D_sym-Mediane: gt 5.4 mm, proxy 9.2 mm.

## Lesart über alle drei Designs
| Design | gt | proxy | Δ gepaart |
|---|---|---|---|
| Clutter (Stage 5) | 52 % | 32 % | −20 Pp. |
| Solo, BOP-Posen (v1) | 51 % | 43 % | −7.7 Pp. |
| Solo, stehend + Yaw (v2) | 59 % | 51 % | −8.6 Pp. |

**Der Proxy-Preis ist robust ~8–9 Pp. ohne Clutter** — über zwei völlig
verschiedene Posen-Regimes. Der Rest des Clutter-Verlusts (~12 Pp.) war Szene.
Die Aufstellregel öffnete drei alte Deckenfälle (tless 28: 0→10, ycbv 3: 0→7,
tless 7 gt: 1→10) und schuf zwei neue (duck, mustard) — Greifbarkeit ist eine
Eigenschaft von Objekt UND Aufstellung, nicht des Objekts allein. Auffällig
stabil: tless 1 und tless 2 gelingen mit dem Proxy besser als mit dem eigenen
CAD, in v1 wie v2.
