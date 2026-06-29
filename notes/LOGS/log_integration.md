# 統合作業ログ

## 2026-06-19 — ur5e-sim 統合パッケージ構築完了

### 完了した作業

**1. プロジェクト基盤構築**

pyproject.toml (ur5e-sim パッケージ定義), pixi.toml (全依存集約), ディレクトリ構造 `src/ur5e_sim/{core,pushing,identification,trajectories}` を作成した.
外部アセットとして mujoco_menagerie submodule と ft300s を assets/ に配置.
scenes/, scripts/, configs/, tests/, notes/ も同時に作成した.

**2. モジュラー MuJoCo シーン XML 構成**

7 ファイルを作成した.

- scenes/common/environment.xml
- scenes/robots/ur5e_gripper.xml
- scenes/robots/ur5e_ft300s_gripper.xml
- scenes/objects/slider.xml
- scenes/objects/payload_box.xml
- scenes/tasks/push.xml
- scenes/tasks/identification.xml

両シーン (push, identification) のロード・ステップ動作を検証済み.

**3. core/ 共通シミュレーション基盤**

14 ファイルを作成した: env.py, robot.py, ik.py, sensors.py, renderer.py, types.py, runner.py, controllers/base.py, controllers/registry.py, logging/base.py, logging/npz.py, __init__.py 群.
全モジュールの import を検証済み.

**4. pushing/ 2D push タスクコード移行**

12 ファイルを移行した: mpc.py, kinematics.py, config.py, task.py, keyframe.py, io.py, paths.py, analytical/push_com_sim.py, viz/plots.py, viz/grid_video.py 等.
import パスを pusher_slider → ur5e_sim.pushing に変更した. ロジックは変更なし.

**5. identification/ 慣性同定コード移行**

22 ファイルを移行した. trajectories/ も 7 ファイルを移行.
import パスを mjwarp_ur5e → ur5e_sim に変更した. ロジックは変更なし.

**6. scripts/ と configs/ の作成**

7 スクリプトを作成した: run_push.py, run_identification.py, make_keyframe.py, analytical_push.py, render_video.py, optimize_trajectory.py, setup_assets.sh.
設定ファイルは push_default.yaml, identification_default.yaml の 2 ファイル.

**7. テスト移行・検証**

結果: 94 passed, 10 skipped (warp GPU), 14 xfailed.
ruff check / ruff format ともにクリーン.

### 設計決定

- pusher-slider のシーン (テーブル + 作業板 + カメラ) を環境ベースとした
- MuJoCo シーンは `<include>` でモジュラーに合成する設計 (environment + robot + object = task scene)
- SimRunner は `controller.compute_control(state: dict) -> ctrl(np.ndarray)` 契約とした. push の PusherSliderMPC はシグネチャが異なるため, アダプタ controller で包む設計とした
- load_model はパス必須とし, assets モジュールへの依存を切断した
- payload_box.xml は ur5e_ft300s_gripper.xml の gripper_mount 内に include することで剛体接続を保証した
- meshdir はトップレベルシーン (scenes/tasks/) からの相対パスで解決する. robot/object フラグメント単独でのロードは不可

### インフラ

リモートオリジンを git@github.com:barikata1984/desktop-ur5e.git に設定した.

### 発見された課題

なし.

---

## 2026-06-19 — /simplify レビューと end-to-end 検証

### /simplify で適用したクリーンアップ

3 回の /simplify レビューで以下を修正した.

- `damped_pinv` 重複削除: `keyframe.py` が独自定義していたものを `core.ik` から import に変更
- `R_TOOL0_DES` を `core` から `pushing` に移動し, `orientation_error` を `R_des` 引数化
- `get_pusher_slider_contact_force` を `core.sensors.ContactSensor` に統合し, 呼び出し側を更新
- FT センサ読み取りを `core.sensors.FTSensor` に統合
- `compute_kinematics=False` フラグを追加 (リグレッサのホットパスで冗長な順運動学計算をスキップ)
- `body_to_contact()` dead code 削除
- `q_weights_arr` / `r_weights_arr` プロパティ削除 (インライン化)
- `break` パスの log append 重複を統合
- `_named_object_id` を `get_named_object_id` に統合
- `step_model`, `get_object_names`, `get_home_qpos`, `apply_joint_overrides` dead code 削除
- `compute_fourier_velocity_bounds` compat alias 削除
- `renderer.py` のカメラ分岐をフラット化
- `.copy().tolist()` の冗長な `.copy()` 削除

### end-to-end 検証で発見・修正したバグ

- **DOF パディング**: `TrajectoryPlayback.execute` で 6-DOF 軌道を 14-DOF モデルで実行する際にゼロパディングが欠けていた. パディング追加で修正.
- **imageio-ffmpeg 欠落**: 動画生成時に依存パッケージが不足していた. `pixi.toml` に追加し, `render_excitation.py` を新規作成.

### end-to-end 検証結果

- push タスク: 軌道実行から動画生成まで完遂
- identification タスク: 軌道最適化 → 慣性パラメータ推定 → 動画生成まで完遂

### README 更新

テンプレート文から ur5e-sim プロジェクト固有の内容に全面書き換えた.

---

## 2026-06-19 — devcontainer postCreateCommand 修正

### 問題

`postCreateCommand: python --version` が exit code 127 ("not found") で失敗していた.

### 根本原因

`postCreateCommand` は非インタラクティブシェルで実行されるため `~/.zshrc` が source されない. そのため pixi の shell-hook が走らず, pixi 環境の PATH (`${WORKSPACE_DIR}/.pixi/envs/default/bin`) が設定されない状態だった.

### 修正内容

`.devcontainer/Dockerfile` の `/etc/zsh/zshenv` heredoc に `${WORKSPACE_DIR}/.pixi/envs/default/bin` を追加した (既存の `~/.local/bin` の前に挿入). `zshenv` はインタラクティブ・非インタラクティブ問わず全 zsh コンテキストで読み込まれるため, `python` コマンドが全ての実行環境で解決できるようになった.

`devcontainer.json` の `postCreateCommand` はそのまま `python --version` を維持 (絶対パス不要).

---

## 2026-06-19 — devcontainer 環境修正と雑整理

### XDG_RUNTIME_DIR 警告の修正

MuJoCo の GUI/EGL レンダリング起動時に "XDG_RUNTIME_DIR is invalid or not set" 警告が出ていた.

`.devcontainer/entrypoint.sh` に root として `/run/user/<uid>` を 0700 で作成するブロックを追加した.
`docker exec` 経由のシェル (entrypoint.sh を経由しない) への対処として, `.devcontainer/Dockerfile` の `/etc/zsh/zshenv` heredoc に `/tmp` フォールバック付きの `XDG_RUNTIME_DIR` 設定を追加した.

### README の pip install -e . 削除

セットアップ手順から `pip install -e .` を削除した.
pixi が `[pypi-dependencies]` の editable install を管理しており, pixi 環境内で pip は使用不可のため, 手順として誤りだった.

### results/ を .gitignore に追加

シミュレーション出力ディレクトリ `results/` が未追跡かつ未 ignore の状態だったため `.gitignore` に追加した.

---

## 2026-06-28 — pushing/ スクリプト群の build_ur5e_model() 移行完了

`notes/PLAN_push_model_builder_migration.md` に定義した Task A/B/C をすべて完了した.

### 変更内容

**Task A — model_builder.py の ft300s_xml=None 対応**

`build_ur5e_model()` に `ft300s_xml: str | None = None` パラメータを追加した.
`None` の場合は FT300s をアタッチせず, FT300s なし構成 (push タスク用) のモデルを返す.

**Task B — pushing/scene.py の新設と build_push_model() 実装**

`src/ur5e_sim/pushing/scene.py` を新設し, `build_push_model()` を実装した.
`build_ur5e_model(ft300s_xml=None)` を呼び出してグリッパのみ構成のモデルをビルドする.

**Task C — task.py / keyframe.py / grid_video.py の移行**

`pushing/task.py`, `pushing/keyframe.py`, `pushing/viz/grid_video.py` を `build_push_model()` に移行した.
monolithic XML ロードを廃止し, モデルビルダー経由に統一した.

全 94 テストがパスしたことを確認した. push 実行でゴール到達を確認した.

### 付随する変更

**グリッドビデオ自動出力**

push タスクは毎回 4 視点グリッド動画 `result_grid.mp4` を出力するようになった (旧: 単視点 `push_sim.mp4`).

**オフスクリーンレンダリング解像度の明示設定**

`model_builder.py` と `pushing/scene.py` に `offwidth` / `offheight` を明示設定した.
`environment.xml` のビジュアル設定が `MjSpec.attach()` 経由では継承されないため, Python 側で上書きが必要なことが判明した.

---

## 2026-06-29 — IK 統合・定数化・バグ修正・テストクリーンアップ

### IK ソルバー統合

`pushing/keyframe.py` の `solve_ik` と `view_pose.py` の `solve_ik_quiet` を `core/ik.py` の `solve_ik()` に統合した (~90 行削減).
`R_des` パラメータ追加と `verbose` フラグで両用途 (詳細ログあり/なし) に対応した.

### グリッパー定数化

`GRIPPER_CLOSED_CTRL = 255` を `core/ik.py` に定義し, `pushing/keyframe.py`, `pushing/task.py`, `identification/execution.py`, `scripts/make_keyframe.py` の 4 ファイルのハードコード値を置換した.

### render_excitation.py マルチカメラ修正

`render_excitation.py` の `_render_single_view` が `view_x` 等の名前付きカメラを参照していたが, `build_ur5e_model()` 出力にはこれらのカメラが存在しない.
名前付きカメラ参照を `MjvCamera` による programmatic カメラに変更した.
`isinstance` による型分岐も不要になり, /simplify で削除した.

### nq=14 不整合修正 (metrics.py / loop.py)

`identification/metrics.py` の `data.qpos[:] = q_trajectory[i]` が nq=14 モデルに 6 要素軌道を代入しようとしてエラーになっていた. `data.qpos[:nq_traj]` に修正した.
`identification/loop.py` の `q_current` / `dq_current` がモデル全体 (nq=14) を返していたため, arm 6 関節のみ抽出するよう修正した.

### xfail マーク整理

不要な xfail マーク 10 個を削除した.

| | 変更前 | 変更後 |
|---|---|---|
| passed | 94 | 106 |
| xfailed | 9 | 2 |
| xpassed | 5 | 0 |

残存する xfail 2 件:

- `test_playback_and_estimation_pipeline`: FT センサがグリッパー質量 (~0.9 kg) も計測するため, ペイロード推定質量が 2.05 kg に膨らむ (真値 1.0 kg).
- `test_feasible`: デフォルト制約でプランナーが実行可能解を見つけられない (加速度制約違反 -0.034).

---

## 2026-06-29 — 残存 xfail 2 件のテスト設計不備を確認

### 背景

xfail マーク整理後に残った 2 件を精査し, プロダクションコードのバグではなくテスト設計の不備と判断した.

### test_playback_and_estimation_pipeline

期待値が"ペイロード単体質量 1.0 kg"になっているが, FT センサーはグリッパー質量 (~0.9 kg) も含めた先端側全質量を計測する. 推定値 ~2.0 kg はセンサー原理に照らして正しい動作であり, テスト側の期待値が誤り. 修正方針は 2 択: (a) 期待値をグリッパー + ペイロードの全質量に変更する, (b) パイプライン結合テストに徹して質量の絶対値検証を外す.

### test_feasible

デフォルト制約 + 2 リスタートで `feasible == True` を要求しているが, 軌道最適化は確率的であり, 2 リスタート程度では実行可能解に到達する保証がない. テストは最適化の性質を無視した強すぎる条件になっている. 修正方針は 3 択: (a) 制約を緩和する, (b) リスタート回数を増やす, (c) `feasible == True` を `余裕 > -0.1` 程度の弱い条件に置き換える.
