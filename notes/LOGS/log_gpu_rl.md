# GPU 加速 RL 移行 調査ログ

## 2026-06-19 — MJX/Warp ベース RL 移行可能性調査

### 調査の目的

ur5e-sim コードベース (pusher-slider 操作・慣性パラメータ同定) を GPU 加速 RL 対応に再実装できるか検討した.

---

### 現在のコードベース分析

9 ファイルにわたり 30 以上の CPU 専用 MuJoCo API 呼び出し (mj_step, mj_forward, mj_inverse, mj_kinematics 等) が存在する.
GPU カーネルは `regressor_warp.py` と `fourier_warp.py` に既存 (NumPy フォールバック付き).
JAX は未使用.

---

### バックエンド候補の評価

**MJX-JAX**

- 自動微分あり
- intvelocity アクチュエータダイナミクス非対応 (UR5e menagerie モデルが使用しているため致命的)
- コンタクト形状の静的割り当てがコンタクト多発シーンでスケールしない
- eq_active (weld 等価拘束) の既知バグあり (issue #2173)

**MJX-Warp**

- 全アクチュエータ型・全ジオメトリ型・全センサ型に対応
- 動的コンタクトスケーリング対応
- 自動微分なし (float32 精度)
- UR5e との互換性の点で最も現実的なバックエンド

**Newton 1.0 (NVIDIA GTC 2026)**

- MJWarp ベース. 操作タスクで MJX-JAX 比 475 倍高速と主張
- 2026 年時点でまだ成熟していない

---

### RL フレームワークエコシステム

| フレームワーク | バックエンド | 備考 |
|---|---|---|
| MuJoCo Playground (google-deepmind) | MJX-JAX/Warp 両対応 | 操作タスク含む, RSS 2025 Outstanding Demo |
| mjlab (2026.01) | MuJoCo Warp | Isaac Lab の manager-based API, pip install 可 |
| Brax v2+ | MJX | PPO, SAC, ARS の RL アルゴリズム実装 |

---

### 特定したブロッカー

1. **ContactSensor**: `mj_contactForce()` / `data.contact[i]` を使用. MJX-JAX に等価な API なし (MJX-Warp は対応).
2. **intvelocity アクチュエータ**: UR5e menagerie モデルが使用. MJX-JAX 非対応 (MJX-Warp は対応).
3. **等価拘束 (weld)**: グリッパの剛体接続に使用. MJX の eq_active バグ (issue #2173) が既知.

---

### 結論

**推奨スタック**: MJX-Warp (物理エンジン) + mjlab または Playground (環境) + Brax (RL アルゴリズム).

方針の根拠:
- Policy gradient 手法 (PPO/SAC) は物理シミュレータの自動微分を必要としない. ポリシーネットワークのみを微分するため, MJX-Warp の autodiff なしは問題にならない.
- MJX-Warp は UR5e の全コンポーネント (intvelocity アクチュエータ, 等価拘束, ContactSensor) に対応しており, 3 つのブロッカーすべてをクリアする.

**移行の性質**: 現在の"1 環境・CPU・逐次実行"から"数千並列環境・GPU・関数型設計"への再設計が必要.
MPC ロジックと軌道生成は再利用可能だが, シミュレーションループ層は新規設計が必要になる.

タスクの完了・課題の解消はなし. 本セッションは調査のみ.
