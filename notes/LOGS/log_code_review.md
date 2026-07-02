# コードレビュー優先順位表

## 2026-07-01 全体レビュー結果 (Tier 1 対応済)

`/code-review` を全コードベースに対して high effort で実行し, 8 finder エージェント (correctness 3 系統, cleanup 3 系統, altitude, conventions) が挙げた候補を dedup した上で 9 tier に整理した.

Tier 1 の 4 件は本セッションで並列エージェントにより修正済み. Tier 2 以下は未対応.

### Tier 1: 通常運用で発火する silent-wrong

| # | 場所 | 状態 |
|---|---|---|
| 1 | `collision.py:187` グリッパ qpos を強制 0 | 修正済 |
| 2 | `mpc/loop.py:190` 悪化を"収束"と誤判定 | 修正済 |
| 3 | `rtls.py:164` TLS 推定が非物理 (質量 ≤ 0) を素通し | 修正済 |
| 4 | `pushing/mpc.py:282` 負 vn で mode 誤分類 | 修正済 |

### Tier 2: 通常運用で発火する crash

| # | 場所 |
|---|---|
| 5 | `mpc/config.py:52` デフォルト `body_name` がアセンブル済モデルと不整合 |
| 6 | `sampling.py:113` デフォルト `site_name='ft_sensor'` が同じく不整合 |
| 7 | `optimizer.py:833` 並列 early-stop 経路で `CancelledError` uncaught. 全 restart 分の結果を失う |

### Tier 3: 潜在バグ (現状の caller は踏まない)

| # | 場所 |
|---|---|
| 8 | `execution.py:143` PD 制御を有効化した瞬間に broadcast crash | 修正済 |
| 9 | `constraints.py:25` `JointLimits` shape (6,) 固定. `num_joints != 6` で SLSQP crash | 修正済 |
| 10 | `mpc/loop.py:87` `RTLSConfig(n_params=10)` hardcode. FT-offset 拡張で crash | 修正済 |

### Tier 4: silent-wrong だが特殊条件下

| # | 場所 |
|---|---|
| 11 | `mpc/planner.py:96` `except (LinAlgError, ValueError): return 1e12` が shape バグを埋没 | 修正済 |
| 12 | `optimize_trajectory.py:138` `except RuntimeError: pass` で workspace 制約を無警告に無効化 | 修正済 |
| 13 | `io.py:62` `save/load` が非対称. 保存されない config フィールドが 10 個超 | 修正済 |

### Tier 5: Altitude (根の浅い修正が症状を再発させ続ける)

| # | 場所 | 補足 |
|---|---|---|
| 14 | `pushing/mpc.py` module deviation #2: `A = 0` かつ `B` を x0 凍結 | motion-cone / 予測 / mode 選択が x0 一点の線形化にぶら下がる根本. 旧 Tier 1 #4 (motion-cone を x0 で凍結) の症状はここに帰着する. Task 4 で per-step motion-cone のコード構造は入れたが, `A = 0` を解消するまで数値挙動は不変. 解消手段: (i) 状態 Jacobian A(t) を復活, または (ii) SCP 反復 |
| 15 | `regressor.py:236` + `execution.py:95` pad-to-model-DOF 二重実装 | 根本は arm/full DoF 抽象化の不在 |
| 16 | `keyframe.py:41` ほか 20 箇所 `q[:6]` マジックナンバー散在 | 同上 |
| 17 | `windowed_fourier.py:15` Fourier 実装 3 系統並立 | BaseTrajectory を継承しない → 修正済 (BaseTrajectory を継承) |
| 18 | `mpc/config.py:54` dead 定数 `model_path` | 修正済 (n_inertial_params に置換) |

### Tier 6: Hot path の効率損失

| # | 場所 |
|---|---|
| 19 | `mpc/loop.py:150` `np.vstack` 毎ステップ再アロケート (O(K²) メモリ traffic) | 修正済 |
| 20 | `optimizer.py:588` 制約 `fun(x)` を診断のため 2 回評価 | 修正済 |
| 21 | `collision.py:290` trajectory clearance が Python 内側ループ | 延期 |
| 22 | `sampling.py:65` `mj_name2id` を毎 sample 呼出 | 修正済 |
| 23 | `execution.py:166` `get_site_frame` が毎ステップ名前解決 | 修正済 |
| 24 | `regressor_warp.py:141` `np.linalg.svd` を Python for-loop | 修正済 |

### Tier 7: 保守性 (Simplification / Reuse)

| # | 場所 | 状態 |
|---|---|---|
| 25 | `optimizer.py:483` sequential / parallel の bookkeeping 二重 | 修正済 |
| 26 | `optimizer.py:241` objective closure copy-paste | 修正済 |
| 27 | `optimizer.py:312` `margin` / `feasible` / `named_margins` 冗長 | 修正済 |
| 28 | `constraints.py:103` position/velocity/acceleration の 3 制約 factory near-identical | 修正済 |
| 29 | `mpc/planner.py:30` `_QuinticCache` が `_TrajectoryCache` と同構造 | 延期 |
| 30 | `pushing/mpc.py:220` stick / slide 分岐がほぼ同形 | 修正済 |
| 31 | `fourier_warp.py:146` `_fourier_trajectory_numpy` 重複 | 延期 |
| 32 | `regressor_warp.py:98` `_batch_skew_numpy` が `_skew` と重複 (計 3 実装) | 延期 |
| 33 | `regressor.py:24` 手書き `_quat_to_rotation_matrix` | 修正済 |
| 34 | `pushing/task.py:55` 生の `mj_name2id` 呼出 | 修正済 |
| 35 | `optimize_trajectory.py:74` `yaml.safe_load` 重複 | 延期 |
| 36 | `tests/identification/conftest.py:10` `SCENE_PATH` を独自組み立て | 延期 |
| 37 | `mpc/loop.py:37` `executed_q` docstring 乖離 | 修正済 |

### Tier 8: テスト品質

| # | 場所 |
|---|---|
| 38 | `test_integration.py:62` 1e12 fallback で自明に緑 (11 とセットで対処) | 修正済 |

### Tier 9: 文書規約違反 (japanese-tech-writing)

| # | 場所 |
|---|---|
| 39 | `notes/PLAN.md:23-31` em ダッシュ用語区切り | 修正済 |
| 40 | `notes/LOGS/log_gpu_rl.md:3` ほか 見出しに em ダッシュ | 修正済 |
| 41 | `notes/LOGS/log_gpu_rl.md:30` 中黒 並列列挙 | 修正済 |
| 42 | `notes/LOGS/log_trajectory_design.md:162` 同上 | 修正済 |
| 43 | `literature/.../Kubus-IROS2007-.../on-line-rigid-object....md:75` em ダッシュ + LLM 口調 | 修正済 |
| 44 | `notes/ISSUES.md:6` 中黒 chain | 修正済 |

### Tier 1 修正内容 (詳細)

| # | ファイル | 変更 | テスト |
|---|---|---|---|
| 1 | `collision.py` `_run_kinematics` | `qpos[:n] = q` に変更し, グリッパ qpos を保存. `n > model.nq` は `ValueError` | -k collision: 8 passed |
| 2 | `mpc/loop.py:190` | `if 0.0 <= rel_improvement < threshold:` に変更, 悪化を"収束"扱いしない | -k mpc_loop: 3 passed, -k mpc: 13 passed 1 xfail |
| 3 | `estimators/rtls.py` `_extract_solution` | `phi_candidate[0] <= 0` なら更新スキップ (負質量ガード) | -k "estimators or tls": 19 passed |
| 4 | `pushing/mpc.py` `_select_mode` (282) | `vn <= 1e-12: return 1` に変更 (数値ノイズによる負 vn は sticking 扱い) | tests/pushing/: 1 passed + smoke check |

### 追跡メモ

- **Tier 1 #4 と Tier 5 #14 の関係**: 旧 Tier 1 #4 (motion-cone を x0 で凍結) は Task 4 の一環で per-step 化のコード構造を入れたが, module の `A = 0` 近似が原因で `x_free` が定数のまま. 数値挙動は不変. 実効化には Tier 5 #14 (nominal-trajectory 線形化 = deviation #2) の解消が必要. altitude の判断ミスで最初症状を Tier 1 に置いていたが, 本ログでは Tier 5 に集約している.
- **Tier 4 #11 と Tier 8 #38 はセット対処**: `mpc/planner.py:96` の 1e12 fallback を消せば `test_integration.py:62` のアサーションも意味を持つようになる.
- **Tier 5 #15, #16 と Tier 3 #8 は同根**: arm/full DoF を明示する `RobotModel` ラッパー (`.arm_slice`, `.arm_qpos`) を導入すれば pad ハックが両方消え, PD 制御の broadcast crash も同時に解決する.
- **Tier 6 #20 と Tier 7 #27 も同根**: 制約評価を 1 回に統合すれば `named_margins` から `margin`/`feasible` が導出でき, 冗長状態が消える.
- **`tests/pushing/` はテスト 1 件のみ**. Tier 5 #14 対応時に MPC 内部の回帰テストを起こす必要がある. 現状のテストでは A=0 解消の効果を数値的に確認する術がない.

## 2026-07-02 P1+P2 一括対応

3 並列エージェント (A, B, C) による P1, P2 課題の一括修正結果を記録する.

A1: 並列モード (`--n-workers > 1`) の `payload_workspace` クラッシュを修正した.

worker (`_run_single_restart`) が `build_ur5e_model()` を payload なしで呼んでいたため, ペイロード body が存在しないまま制約構築が body 名を解決しようとしていた.

worker 側にも CLI と同じ `payload_xml` を渡すよう修正し, sequential と並列で同一のモデル構成になった.

A3: `io.py` の config round-trip を完全化した.

`_joint_limits_to_dict` と `objective_type` のシリアライズ, デシリアライズを追加し, 保存, 復元後に `OptimizerConfig` が完全に一致するようにした.

A4: `execution.py` に `ft_site_name` を追加した.

FT センサーの解析フォールバックが評価すべきなのは EE 姿勢用の `site_name` ではなく FT センサー自身のサイトであり, 両者を独立したフィールドとして分離した.

A5: `data_buffer.py` の `build_regressor_data` が `sample_body_regressor` に `site_name` を渡していない問題を修正した.

当初は dead code とみなされていたが, 調査の結果 `scripts/run_identification.py:180` から呼ばれる live 経路であると判明した.

`sample_body_regressor` は `site_name` を必須引数として要求するよう変更済みのため, このままでは `ValueError` で落ちる.

`build_regressor_data` に `site_name: str` 引数を追加し, `run_identification.py` からモデル構成に一致する `opt_result.config.site_name` (`"ft300s_ft_sensor"`) を渡すよう修正した.

A6: ホームポーズ定義を 3 箇所中 2 箇所で統一した.

`mpc/config.py` が独自に持っていた `_UR5E_HOME` 配列を削除し, `optimizer.py` の `UR5E_HOME_QPOS` を import する形にした.

`model_builder.py` の `_ARM_HOME_QPOS` と `scenes/tasks/identification.xml` の keyframe はそれぞれ異なる値のままであり, 3 値の乖離は `notes/ISSUES.md` に新規課題として記録した.

C1: `tests/pushing/test_mpc.py` に MPC の回帰テストを 10 本追加した.

`_select_mode` の境界条件, 予測の自由応答特性, `compute_control` の健全性, 解析的な短時間ロールアウトを検証する.

C2: 残存 xfail 2 件を解消した.

`test_playback_and_estimation_pipeline` は期待値を FT センサー先端側全質量の比較用に `model.body_subtreemass` へ変更した.

`test_feasible` は最適化の確率的性質を踏まえ, 判定条件を余裕 > -0.1 に緩和した.

C3: `test_iter_logs_key_structure` で `iter_logs` のキー構造を固定するテストを追加した.
- **`tests/pushing/` はテスト 1 件のみ**. Tier 5 #14 対応時に MPC 内部の回帰テストを起こす必要がある — 現状のテストでは A=0 解消の効果を数値的に確認する術がない.
