# モデルビルダーログ

## 2026-06-25 MjSpec.attach() によるモデルビルダー実装

### 背景

慣性同定パイプラインが monolithic XML (ur5e_ft300s_gripper.xml) に依存していた.
nq=14 の 14-DoF モデル (FT300s の仮想 DoF を含む) と 6 関節軌道の不一致が発覚し,
regressor.py 等に zero-padding を入れる対処を行った.
根本的な解決策として, モデルを Python 側で動的に組み立てる方針に切り替えた.

---

### 実装内容

**新規ファイル**

- `src/ur5e_sim/core/model_builder.py` — `build_ur5e_model()` を実装. MjSpec.attach() で UR5e + FT300s + 2F-85 を合成する
- `scenes/common/sensors/ft300s.xml` — FT300s センサのスタンドアロン XML (Menagerie 形式)

**変更ファイル**

- `src/ur5e_sim/identification/regressor.py` — nq=14 モデルへの zero-padding 追加, build_ur5e_model() 使用に移行
- `src/ur5e_sim/identification/workspace.py` — 同上
- `src/ur5e_sim/identification/collision.py` — 同上

---

### オフセット値

| パラメータ | 値 | 備考 |
|---|---|---|
| attachment_site | 0.094 m | 旧 ft300s_mount オフセット値 (物理的な正しさ未確認) |
| gripper_base_mount | 0.004 m | |

attachment_site 94mm は DH パラメータの L_TP=100mm と 6mm の差異がある. 未解決 → ISSUES 参照.

---

### グリッドサーチ結果 (ft_offset=True, 24 条件)

全 24 条件が実行不可能 (衝突余裕 ≈ -0.10). ペイロード配置が原因の可能性が高い.
ロボット構成オフセット調整後に再実行予定.

---

### OLS vs TLS 分析 (Kubus 論文より)

慣性パラメータ同定における最小二乗法の選択に関して Kubus らの文献を調査した.

- OLS (普通最小二乗法): 回帰行列 W の誤差を無視. センサノイズが小さい場合に適用可能
- TLS (全最小二乗法): W と τ 両方の誤差を考慮. センサノイズが大きい場合に理論的に優位

現在の実装は OLS. センサノイズの大きさによって TLS への切り替えを検討する.

---

### 次のステップ

1. attachment_site の物理的な正しさを実機寸法で確認
2. グリッドサーチ再実行 (オフセット調整後)
3. pushing/ スクリプトの build_ur5e_model() 移行 (PLAN_push_model_builder_migration.md 参照)

---

## 2026-06-29 scene.py/model_builder.py 重複排除

### 背景

`pushing/scene.py` の `build_push_model()` は `_build_push_spec()`, `_add_push_cameras()`, `_add_push_keyframes()` 等を独自に持ち, `core/model_builder.py` の `build_spec()` (旧 `_build_spec`) と共通ロジックが約 120 行重複していた.

### 変更内容

`pushing/scene.py` の重複メソッド群を削除し, `core/model_builder.py` の `build_spec()` に委譲する形に書き換えた.
`build_spec()` には以下のパラメータを追加した.

| 追加パラメータ | 用途 |
|---|---|
| `extra_cameras` | push 用カメラ (view_x 等) を渡す |
| `extra_keyframes` | push 用キーフレームを渡す |
| `condim` | push シーンの接触次元設定 |

`pushing/scene.py` の実質コードは ~120 行削減された.

---

## 2026-06-25~26 ワークスペース・ペイロード・ホームポーズ更新, site_name バグ修正, 再起動履歴テレメトリ

### ワークスペース設定の修正

旧ワークスペース z:[0.05, 0.5] は `_mjwarp_ur5e` から引き継いだもので, ロボットベースが z=0.3 のテーブル上にあるため, ワークスペース下端がテーブルと交差していた.

同定初期姿勢のピンチサイト位置を中心に再定義した.

| パラメータ | 値 |
|---|---|
| 中心 (x, y, z) | [0, 0.65, 0.636] |
| 半幅 (x, y, z) | [0.25, 0.35, 0.275] |

`model_builder.py` に `workspace_region_geom` を追加した (contype=0, group=4 の視覚専用ジオメトリ).
`optimize_trajectory.py` と `grid_search.py` はこのジオメトリからワークスペース境界を読み取る.

### ペイロード設定の変更

- `scenes/objects/payload_flat.xml` を新規作成 (half-size [0.15, 0.05, 0.05], 単一 box geom)
- ペイロードのアタッチポイントを `ft300s_ft_sensor` から `gripper_pinch` に変更
- `optimize_trajectory.py` と `grid_search.py` でペイロードありのモデルをビルドするよう更新

**発見した落とし穴**: `MjSpec.attach()` は子 XML に `<worldbody>` ラッパーがないと body/geom をサイレントに無視する. `payload_flat.xml` 初期版がこれに該当し, アタッチ後にジオメトリが消えていた.

### ホームポーズ更新

`model_builder.py` の `_ARM_HOME_QPOS` を同定初期姿勢に変更した.

| | 関節角 (rad) |
|---|---|
| 旧 | `[-π/2, -π/2, π/2, -π/2, -π/2, 0]` |
| 新 | `[1.324683, -1.468515, 1.368294, -1.470575, -1.570796, -0.246113]` |

旧ホームポーズはピンチ位置が y=-0.39 となりワークスペース外に置かれていた.

### site_name 伝播バグの修正

**根本原因**: `sampling.py:113` の `site_name` デフォルト値が `"ft_sensor"` だが, `build_ur5e_model()` は FT300s をアタッチするため実際のサイト名は `"ft300s_ft_sensor"` となる. サイトが見つからないと ValueError → except で捕捉 → 条件数 1e12 を返す, という沈黙した失敗が発生していた.

**修正**: `site_name` パラメータを `_compute_stacked_regressor` → `condition_number_objective` / `d_optimal_objective` / `d_optimal_with_cond` / `evaluate_full_resolution` (objective.py) と全呼び出し元 (optimizer.py) に伝播させた. スクリプトは `OptimizerConfig` 経由で `site_name="ft300s_ft_sensor"` を渡す.

### 再起動履歴テレメトリの追加

`OptimizationResult` に `restart_history: list[dict]` フィールドを追加した.
逐次・並列どちらのオプティマイザパスも, 再起動ごとのサマリー (条件数の反復ログ `iter_logs` を含む) を収集する.
結果は `result.json` に保存・ロードされ, 最適化収束の事後分析に使用できる.

---

## 2026-06-26 ft_sensor サイト位置変更, ホームポーズ水平姿勢変更, /simplify クリーンアップ

### ft_sensor サイト位置変更

`ft_sensor` サイトを `gripper_mount` からの相対位置 z=-0.0065 に移動した (FT300s ハウジング内部).
FT300s の物理的な計測点の位置に合わせた変更.

### ホームポーズ水平姿勢変更

`_ARM_HOME_QPOS` を実験目的で水平ツール姿勢に変更した.

| | 関節角 (rad) |
|---|---|
| 変更前 (同定初期姿勢) | `[1.324683, -1.468515, 1.368294, -1.470575, -1.570796, -0.246113]` |
| 変更後 (水平姿勢) | `[1.031643, -1.461450, 2.562062, -4.242204, -1.031607, 0.000245]` |

水平姿勢への変更は FIM 相関実験目的であり, push パイプライン作業前に戻すか設定可能にする必要がある → ISSUES 参照.

### /simplify クリーンアップ

`src/ur5e_sim/core/env.py` に `get_workspace_bounds()` を追加し,
`grid_search.py` と `optimize_trajectory.py` で重複していたワークスペース境界読み取りコードを削除した.
`OptimizerConfig` のデフォルト値を `payload_payload_box_mount`/`ft300s_ft_sensor` に更新し,
`build_ur5e_model()` の出力と一致させた.

---

## 2026-06-28 ホームポーズを同定初期姿勢に戻す, Markovsky 2007 TLS サーベイ

### ホームポーズ復元

`_ARM_HOME_QPOS` を FIM 相関実験用水平姿勢から同定初期姿勢に戻した.
push パイプライン移行が完了したため, 実験姿勢を維持する理由がなくなった.

| | 関節角 (rad) |
|---|---|
| 実験中 (水平姿勢) | `[1.031643, -1.461450, 2.562062, -4.242204, -1.031607, 0.000245]` |
| 現在 (同定初期姿勢) | `[1.324683, -1.468515, 1.368294, -1.470575, -1.570796, -0.246113]` |

### Markovsky 2007 TLS サーベイ

OLS vs TLS の選択根拠を精査するため, Markovsky & Van Huffel (2007) のサーベイ論文を精読した.
ペーパーサマリーを `literature/papers/Markovsky-SigProc2007-Overview_Total_Least-Squares/` に作成した.

慣性同定における TLS 適用の要点:

- TLS は回帰行列 W と観測ベクトル τ 双方に誤差がある場合に統計的に一貫した推定量を与える
- センサノイズが小さい (SNR が高い) 場合, OLS と TLS の差は実用上無視できる
- 現在の実装は OLS. センサノイズ水準を評価してから TLS への切り替え要否を判断する
