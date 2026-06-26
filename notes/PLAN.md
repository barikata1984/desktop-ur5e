# 設計方針

## 目的

mjwarp_ur5e (慣性パラメータ同定) と mj-2d-push (2D pusher-slider) の 2 つの UR5e シミュレーション環境を, 単一コンテナ上の統合パッケージ `ur5e-sim` に統合する.

統合は 2026-06-19 に完了した.

## 構成原則

1. **モジュラーシーン**: 環境 (テーブル, カメラ) / ロボット構成 (FT300s 有無) / タスク対象物 を MuJoCo `<include>` で合成
2. **モジュラーシミュレーション**: Controller, Sensor, Logger を Protocol ベースで差し替え可能にし, SimRunner が組み合わせてループを回す
3. **単一パッケージ**: `ur5e_sim` パッケージ内に `core/`, `pushing/`, `identification/`, `trajectories/` のサブパッケージ

## シーンの基盤

pusher-slider のシーン (テーブル + 作業板 + カメラ配置) を環境のベースとした.

## シーン XML 構成

7 ファイルを `<include>` で合成する設計.

- scenes/common/environment.xml — テーブル, 作業板, カメラ
- scenes/robots/ur5e_gripper.xml — グリッパのみ構成
- scenes/robots/ur5e_ft300s_gripper.xml — FT300s + グリッパ構成
- scenes/objects/slider.xml — pusher-slider タスク用物体
- scenes/objects/payload_box.xml — 慣性同定タスク用ペイロード (gripper_mount 内に include して剛体接続)
- scenes/tasks/push.xml — push タスクのトップレベルシーン
- scenes/tasks/identification.xml — 慣性同定タスクのトップレベルシーン
- scenes/tasks/trajectory_design.xml — 軌道設計専用シーン (FT300s + グリッパ, ペイロードなし)
- scenes/common/robots/ur5e_ft300s_gripper_no_payload.xml — FT300s ロボット変形版 (ペイロード include を除いた構成; 現状は全行コピーで技術的負債あり → ISSUES 参照)

meshdir はトップレベルシーン (scenes/tasks/) からの相対パスで解決する. robot/object フラグメントの単独ロードは不可.

## グリッパ XML 命名規則

グリッパ構成要素 (body/joint/site/geom/actuator) の名前は `gripper_` プレフィックスを付けない.
`pinch`, `base_mount`, `right_*`, `left_*`, `split`, `fingers_actuator` が標準形.
push 版 (`ur5e_gripper.xml`) は 2026-06-23 にこの規則へ統一済み.

## SimRunner の契約

`controller.compute_control(state: dict) -> ctrl(np.ndarray)` を統一インタフェースとする.
PusherSliderMPC はこのシグネチャと異なるため, アダプタ controller で包む.

## 移行方針の結果

import パスの書き換えが主で, ロジック変更は最小限に収まった.
pusher_slider/sim/runner.py の monolithic run() は SimRunner + PushTask に分解した.

## モデルビルダー (2026-06-25 追加)

MuJoCo モデルの組み立てを monolithic XML から `MjSpec.attach()` ベースの Python コードへ移行した.

### 設計決定

- `src/ur5e_sim/core/model_builder.py` に `build_ur5e_model()` を実装
- Menagerie の `ur5e.xml` と `2f85.xml` をそのまま参照 (コピーしない)
- FT300s は `scenes/common/sensors/ft300s.xml` としてスタンドアロン XML を新設
- オフセット値 (attachment_site=0.094 m, gripper_base_mount=0.004 m) は XML ではなく model_builder.py で実行時に上書き
- 全同定パイプラインスクリプトは `build_ur5e_model()` でモデルをロードする

### 残課題

push migration plan (`notes/PLAN_push_model_builder_migration.md`) に従い, pushing/ スクリプト群の移行が未実施.

## ワークスペース・ペイロード・ホームポーズ (2026-06-25~26 追加)

### ワークスペース定義

ワークスペースはロボットベース中心ではなく, 同定初期姿勢におけるピンチサイト位置を中心とする.

- 中心: [0, 0.65, 0.636]
- 半幅: [0.25, 0.35, 0.275]

`build_ur5e_model()` 内の `workspace_region_geom` (contype=0, group=4) にこの値を埋め込み,
最適化スクリプトはこのジオメトリから境界を読み取る.

### ペイロードアタッチポイント

ペイロードは `ft300s_ft_sensor` ではなく `gripper_pinch` にアタッチする.
FT センサ以降の質量を測定対象に含めるための設計判断.

### ホームポーズ

`_ARM_HOME_QPOS` は同定初期姿勢 `[1.324683, -1.468515, 1.368294, -1.470575, -1.570796, -0.246113]` とする.
これにより最適化開始点がワークスペース内に収まる.
