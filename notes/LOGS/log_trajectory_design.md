# 軌道設計ログ

## 2026-06-23 — インタラクティブ軌道設計ツール (IK コーンスイープ) の実装

### 目的

慣性同定用の励起軌道を手動設計するのではなく, UR5e の手首付近に作業面を設け, MuJoCo GUI でリアルタイムにプレビューしながら軌道を生成・エクスポートするツールを実装した.

---

### 追加ファイル

- `scripts/view_pose.py` — IK ベースのコーンスイープ軌道生成スクリプト. MuJoCo passive viewer (GUI) でリアルタイム可視化, 軌道を `_traj.json` 形式でエクスポートする
- `scenes/tasks/trajectory_design.xml` — 軌道設計専用シーン (UR5e + FT300s + Robotiq 2F-85, ペイロードなし)
- `scenes/common/robots/ur5e_ft300s_gripper_no_payload.xml` — FT300s ロボット変形版 (慣性同定ペイロードを除いた構成)

---

### 軌道パラメータ

| パラメータ | 値 |
|---|---|
| 総時間 | 20 s |
| サンプリングレート | 100 fps |
| ステップ数 | 2000 |
| 作業面中心 (TCP からの高さ) | +15 cm |
| コーン半頂角 | 30 deg (= 頂角 60 deg) |
| wrist_3 オフセット | +90 deg |

---

### 出力形式

`results/cone_sweep_traj.json` に `_traj.json` 形式で出力する.
各ステップに `t`, `q`, `dq`, `ddq` を持つオブジェクトの配列.

---

### /simplify レビューで適用したクリーンアップ

`view_pose.py` に対して /simplify レビューを実施し, 以下を修正した.

- コメント内の角度表記の誤り (45 deg → 30 deg, 90 deg → 60 deg 頂角) を修正
- `close_gripper_sim()` を `pushing.keyframe` から再利用するよう変更 (独自実装の dead code を削除)
- 意味のない `*1.0` 乗算を削除

---

### 発見した知見

**FT300s ロボット XML の body/site 命名**

FT300s あり構成のロボット XML は, FT300s なし版と body/site 名が異なる.
例: `pinch` → `gripper_pinch`, `base_mount` → `gripper_base_mount`.
FT300s あり構成を使うスクリプトでは, これらの名前を確認して参照する必要がある.

**MuJoCo passive viewer でのカスタムフレーム可視化**

`mujoco.viewer.launch_passive` の `user_scn` に対し, `mjv_connector` と `mjv_initGeom` を組み合わせることで, RGB 矢印 (XYZ 軸) などのカスタムジオメトリをリアルタイム描画できることを確認した.

---

## 2026-06-23 — グリッパ命名統一・MPC 進捗表示・軌道時間変更

### グリッパ XML 命名統一

push 版 (`ur5e_gripper.xml`) のグリッパ body/joint/site/geom 名に付いていた `gripper_` プレフィックスを除去し, FT300s 版との命名規則を揃えた.

主な変更対応表:

| 変更前 | 変更後 |
|---|---|
| `gripper` (body) | `gripper_mount` |
| `gripper_base_mount` | `base_mount` |
| `gripper_pinch` | `pinch` |
| `gripper_right_*` | `right_*` |
| `gripper_left_*` | `left_*` |
| `gripper_split` | `split` |
| `gripper_fingers` | `fingers_actuator` |

参照側ファイル (`push.xml`, `task.py`, `keyframe.py`) の名前も同時更新した.
全 94 テストがパスしたことを確認した.

変更対象ファイル:

- `scenes/common/robots/ur5e_gripper.xml`
- `scenes/tasks/push.xml`
- `src/ur5e_sim/pushing/task.py`
- `src/ur5e_sim/pushing/keyframe.py`

### MPC 進捗表示

`pushing/task.py` の MPC プッシュループに, `\r` 上書きによるリアルタイム進捗表示を追加した.
10 ステップごとに `[XX.X%]` 形式で標準出力に出力される.

### 軌道時間変更

`scripts/view_pose.py` のコーンスイープ軌道を 20 s (2000 ステップ, 100 fps) に変更した.
(上記"軌道パラメータ"表は初回記録時点で既に 20 s を反映済み)

---

## 2026-06-25~26 — グリッドサーチ実行 (ワークスペース・ペイロード更新後) と励起品質分析

### グリッドサーチ実行結果 (24 条件)

スイープ構成: ワークスペース半幅 (3) × EE 速度 (2) × dq (2) × ddq (2) = 24 条件.
ワーカー数を 14 から 24 に増加 (32 コアマシンに合わせ, 条件数にも一致).

全 24 条件で条件数は有限 (6.2–11.2) となり, site_name バグ修正の効果を確認した.
ただし全条件が infeasible (余裕 ≈ -0.0000).

| 評価軸 | 最良値 | 備考 |
|---|---|---|
| 条件数最小 | 6.234 | dq=π, ddq=4π (高速; 実機では危険) |
| 安全寄り | ≈ 10.9 | dq=π/2, ddq=2π |

違反した制約は主に EE 速度上限と payload_workspace (ペイロードがワークスペース境界をわずかに超える).

### 励起品質分析 — FIM と相関行列

条件数を補完する診断として, FIM 固有値スペクトル, パラメータ相関行列, 列ノルムを算出した.

**主な発見**:

- b_tz↔m 相関 0.98: z 方向バイアスと質量が強く結合している
- b_fx↔hy 相関 0.98: x 方向力バイアスと y 軸角運動量が結合している
- b_fy↔hx 相関 -0.97: y 方向力バイアスと x 軸角運動量が逆相関している

この結合は, 最適化後もポーズ変動が矢状面 (xz 平面) に偏っているため,
バイアスと重力ベクトルの相対的な変化が不十分であることを示す.

**文献との整合**:
Duan et al. (2022), Swevers et al. の先行研究および Cramér-Rao 理論はいずれも,
FIM 相関行列を条件数の補助診断として使うことを支持する.

### 結論

条件数を目的関数とすること自体は妥当.
問題は制約が緊すぎて十分なポーズ変動が得られないことにある.
制約の緩和 (EE 速度上限・payload_workspace 上限) か,
矢状面外への軌道変動を促すペナルティ項の追加が次の方針候補.
