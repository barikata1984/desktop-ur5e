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

meshdir はトップレベルシーン (scenes/tasks/) からの相対パスで解決する. robot/object フラグメントの単独ロードは不可.

## SimRunner の契約

`controller.compute_control(state: dict) -> ctrl(np.ndarray)` を統一インタフェースとする.
PusherSliderMPC はこのシグネチャと異なるため, アダプタ controller で包む.

## 移行方針の結果

import パスの書き換えが主で, ロジック変更は最小限に収まった.
pusher_slider/sim/runner.py の monolithic run() は SimRunner + PushTask に分解した.
