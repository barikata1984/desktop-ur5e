# モデルビルダーログ

## 2026-06-25 — MjSpec.attach() によるモデルビルダー実装

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
