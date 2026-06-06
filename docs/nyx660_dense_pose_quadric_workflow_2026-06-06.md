# NYX660 dense pose + tracking + quadric workflow

作成日: 2026-06-06

## 目的

`NYX660_2025_12_01_17_33_27_0135` について、COLMAP由来のカメラpose、DEIMv2 + ByteTrack の追跡bbox、Depth gate済みtrackletを使い、GTSAM quadrics による3D quadric候補を作る。

今回の主眼は、keyframe poseだけでなく中間フレームにもposeを割り当て、MOTの全フレーム観測をできるだけGTSAM側へ渡すことにある。global scale は今回は重視しない。

## 全体フロー

| 手順 | 入力 | 処理 | 出力 |
| --- | --- | --- | --- |
| 1 | COLMAP keyframe model | keyframe poseを確認 | `colmap_text_stride20` |
| 2 | keyframe pose | 未登録frameをSE(3)補間 | `colmap_text_stride20_dense_interpolated` |
| 3 | RGB連番 | DEIMv2 ONNX + ByteTrack | `coco.json` |
| 4 | `coco.json` | Optunaでpost-MOT filter探索 | `coco_good_optuna.json` |
| 5 | `coco_good_optuna.json` + mapped_depth | tracklet代表Depth `<=1.5m` gate | `coco_good_optuna_depth15.json` |
| 6 | dense pose + depth-gated COCO | GTSAM observations作成 | `observations_dense_optuna.json` |
| 7 | observations | quadric初期化・factor graph構築 | `quadrics_dense_optuna.json` |

## Poseの扱い

| 種類 | 件数 | 扱い |
| --- | ---: | --- |
| COLMAP keyframe pose | `61` | 実登録poseとして使用 |
| SE(3)補間pose | `1158` | 中間フレーム用のfallback poseとして使用 |
| 合計 | `1219` | tracking観測とjoin |

補間poseは、前後の登録済みposeから `slerp + lerp` で作る。これはCOLMAP BA済みposeではないため、summaryの `pose_source_counts` で必ず区別する。

## Tracking filter結果

| 指標 | 従来filter | Optuna filter |
| --- | ---: | ---: |
| tracks before depth gate | `495` | `831` |
| tracks after depth gate | `495` | `829` |
| annotations after depth gate | `13133` | `18044` |
| frames | `1219` | `1219` |
| representative depth median | `0.5835m` | `0.585m` |

Optunaの最良値は `min_len=3`, `min_score=0.3511`, `max_area_cv=0.7938`。候補数を増やす方向に寄っているため、最終採用前には動画で短いtrackの混入を確認する。

## GTSAM出力

| 指標 | dense baseline | dense + Optuna |
| --- | ---: | ---: |
| measurements | `13133` | `18044` |
| tracks | `495` | `829` |
| output quadrics | `495` | `829` |
| factor count | `13134` | `18045` |
| pose count | `1219` | `1219` |
| mean bbox center jitter px | `108.65` | `85.81` |
| optimizer status | `not_requested` | `not_requested` |

## 成果物

| 成果物 | パス |
| --- | --- |
| Optuna summary | `/home/inaho-omen/Project/tomato_tracking_deim_mot/outputs/nyx660_colmap_deimv2_bytetrack/optuna_tracklets/optuna_summary.md` |
| GTSAM入力COCO | `/home/inaho-omen/Project/tomato_tracking_deim_mot/outputs/nyx660_colmap_deimv2_bytetrack/optuna_tracklets/coco_good_optuna_depth15.json` |
| observations | `outputs/quadric_slam_nyx660_17_33_27_dense_optuna/observations_dense_optuna.json` |
| quadrics | `outputs/quadric_slam_nyx660_17_33_27_dense_optuna/quadrics_dense_optuna.json` |
| visualization video | `outputs/quadric_slam_nyx660_17_33_27_dense_optuna/quadric_slam_reprojection_dense_optuna.mp4` |

`outputs/` 配下は生成物であり、git管理対象にしない。

## 実行コマンド

```bash
PYTHONPATH=build:build/gtsam/python \
LD_LIBRARY_PATH=build:build/gtsam/gtsam:build/gtsam/gtsam/3rdparty/metis/libmetis:${LD_LIBRARY_PATH:-} \
.pixi/envs/default/bin/python -m scripts.quadric_slam_pipeline.build_observations \
  --colmap-text-model /home/inaho-omen/data/colmap_workspaces/NYX660_2025_12_01_17_33_27_0135_colmap/colmap_text_stride20_dense_interpolated \
  --tracking-coco /home/inaho-omen/Project/tomato_tracking_deim_mot/outputs/nyx660_colmap_deimv2_bytetrack/optuna_tracklets/coco_good_optuna_depth15.json \
  --out outputs/quadric_slam_nyx660_17_33_27_dense_optuna/observations_dense_optuna.json

PYTHONPATH=build:build/gtsam/python \
LD_LIBRARY_PATH=build:build/gtsam/gtsam:build/gtsam/gtsam/3rdparty/metis/libmetis:${LD_LIBRARY_PATH:-} \
.pixi/envs/default/bin/python -m scripts.quadric_slam_pipeline.reconstruct_quadrics \
  --observations outputs/quadric_slam_nyx660_17_33_27_dense_optuna/observations_dense_optuna.json \
  --out outputs/quadric_slam_nyx660_17_33_27_dense_optuna/quadrics_dense_optuna.json \
  --min-measurements 3
```

## DoD

| DoD | 判定 | 根拠 |
| --- | --- | --- |
| dense poseを使う | 達成 | `pose_source_counts`: keyframe `61`, interpolated `1158` |
| tracking観測を全フレームposeとjoinする | 達成 | `dropped_unregistered_measurements=0` |
| Depth gateで1.5m以内に限定する | 達成 | Optuna後 `829/831` tracks kept、dropは `no_valid_depth` のみ |
| quadricsを生成する | 達成 | `output_quadrics=829` |
| 可視化動画を生成する | 達成 | `1219` frames / 800x600 / 15fps / 122,492,593 bytes |
| 生成物をgitに混ぜない | 達成 | `outputs/` はgit管理外。docsのみcommit対象 |

## 注意点

- 今回は `reconstruct_quadrics.py` のoptimizerは未実行で、`optimizer_status=not_requested`。
- dense poseの大半は補間poseであり、COLMAP BAで直接最適化されたposeではない。
- Optunaはpost-MOT filterだけを探索しており、DEIMv2推論やByteTrack内部パラメータは再探索していない。
- 候補数は大きく増えたが、最良値は緩めである。最終採用前に動画可視化で品質確認する。
