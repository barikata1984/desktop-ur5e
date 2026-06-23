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
