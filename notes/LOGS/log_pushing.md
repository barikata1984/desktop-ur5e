# Pushing / MPC 作業ログ

## 2026-07-01

### 論文ノートの数式変換

5本の論文サマリファイルのプレーンテキスト/バッククォート数式を LaTeX `$...$` 表記に変換した:

- Hogan-WAFR2016 (pusher-slider MPC)
- Lynch-IJRR1996 (stable pushing)
- Kubus-IROS2007 (inertial parameter identification)
- Kubus-IROS2008 (recursive TLS)
- Markovsky-SigProc2007 (TLS overview)

コード参照 (`fmincon`, `rigid_body_wrench_regressor` 等) はバッククォートのまま保持.

### Hogan 2016 論文ノートに QP コスト関数の導出を追記

追加議論セクションに, 状態コスト + 制御コストから QP 標準形 ($H$, $f$) への変換過程を記録.

### pusher-slider MPC の構造理解

セッション内で以下を確認・議論:

- 接触モード (固着/上滑り/下滑り) とモーションコーン境界 $\gamma_t, \gamma_b$
- MIQP の"混合整数"の意味 (連続変数 + 整数変数の混在)
- モードスケジュールと FOM による凸 QP への帰着 ($3^N$ → 3パターン)
- モードごとの $B_j$ 行列切替と状態の連続性
- $SE(2)$ ゴール姿勢による目標設定と receding horizon
- 名目軌道 (nominal) vs 目標軌道 (reference) の区別
- QP コスト関数: $J_\text{state} + J_\text{control}$ → $\frac{1}{2} z^\top H z + f^\top z$ への導出
