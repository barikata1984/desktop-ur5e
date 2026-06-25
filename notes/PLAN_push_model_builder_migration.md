# Push シーンの build_ur5e_model() 移行プラン

## 背景

identification パイプラインは `build_ur5e_model()` (MjSpec.attach) でモデルを組み立てるよう移行済み.
push パイプラインはまだ `push.xml` を直接ロードしており, `model_builder.py` のオフセット上書き
(attachment_site=0.094, gripper_base_mount=0.004) が反映されない.

push シーンでは FT300s を使わないため, `build_ur5e_model()` に FT300s スキップ機能を追加する必要がある.

## 現状の push シーン構成

`scenes/tasks/push.xml` は以下を `<include>` で合成:
- `scenes/common/environment.xml` — 床, テーブル, 作業面, カメラ
- `scenes/common/robots/ur5e_gripper.xml` — UR5e + 2F-85 (FT300s なし, prefix なし)
- `scenes/objects/slider.xml` — スライダー物体

加えて push.xml 自体に:
- contact exclude 39 箇所 (ロボットリンク vs テーブル/作業面)
- contact pair 6 箇所 (スライダー摩擦, パッド摩擦)
- keyframe "ready" (qpos 21 = arm 6 + gripper 8 + slider freejoint 7)

## 現状のボディ名 (prefix なし)

push.xml の contact exclude/pair は prefix なしの名前を使用:
`gripper_mount`, `base_mount`, `gripper_base`, `right_driver`, `right_pad1`, etc.

`build_ur5e_model()` 経由だと gripper は `gripper_` prefix 付きになる:
`gripper_base_mount`, `gripper_base`, `gripper_right_driver`, `gripper_right_pad1`, etc.

注意: menagerie の 2f85.xml のボディ名は `base_mount`, `base`, `right_driver` 等.
prefix "gripper_" が付くと `gripper_base_mount`, `gripper_base`, `gripper_right_driver` になる.
push.xml の既存名 `gripper_base` は menagerie の `base` + prefix "gripper_" = `gripper_base` で一致する.
ただし `base_mount` → `gripper_base_mount`, `right_pad1` → `gripper_right_pad1` 等は変わる.

## タスク分解 (3 並列 + 1 直列)

### Task A: model_builder.py の拡張 (並列)

`_build_spec()` を拡張して FT300s なし構成をサポート.

変更内容:
1. `ft300s_xml` のデフォルトを維持しつつ `ft300s_xml=None` で FT300s スキップ
2. FT300s なしの場合: gripper を `attachment_site` に直接 attach
3. FT300s なしの場合: `ft300s_ft_sensor` 等の FT 関連 site/sensor は存在しない
4. `_build_spec` に `slider_xml: str | None = None` パラメータを追加
   - slider を worldbody に attach (freejoint 付き物体なので worldbody 直下)
5. `_build_spec` に `extra_contacts: list[dict] | None = None` パラメータを追加
   - contact exclude / pair をビルダーの外から注入できるようにする
   - または `_add_push_contacts(spec)` のようなヘルパーを別途用意
6. keyframe: FT300s なしの場合 qpos は arm 6 + gripper 8 = 14 (現状と同じ)
   slider ありの場合 qpos は 14 + freejoint 7 = 21

検証:
```python
# FT300s なし + slider なし
model, data = build_ur5e_model(ft300s_xml=None)
assert model.nq == 14  # arm 6 + gripper 8
assert model.nsensor == 0

# FT300s あり (現状)
model, data = build_ur5e_model()
assert model.nq == 14
assert model.nsensor == 2
```

### Task B: push 用ビルダー関数を作成 (並列, Task A と独立に設計可能)

`src/ur5e_sim/pushing/scene.py` (新規) に push シーン組み立て関数を作成.

```python
def build_push_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """UR5e + 2F-85 (FT300s なし) + slider + push 用 contact/keyframe."""
```

内部で:
1. `build_ur5e_model(ft300s_xml=None, slider_xml="scenes/objects/slider.xml")` を呼ぶ
   (Task A の拡張が必要. あるいは _build_spec を使って自前で slider を attach)
2. contact exclude を追加 (prefix 付きボディ名で)
3. contact pair を追加 (slider vs テーブル/パッド)
4."ready" keyframe を設定

contact exclude の名前マッピング (push.xml → build_ur5e_model 後):

| push.xml の名前 | build_ur5e_model 後の名前 |
|---|---|
| base | base (UR5e, prefix なし) |
| wrist_1_link | wrist_1_link (UR5e, prefix なし) |
| wrist_2_link | wrist_2_link (同上) |
| wrist_3_link | wrist_3_link (同上) |
| gripper_mount | ??? (FT300s なし構成では存在しない可能性) |
| base_mount | gripper_base_mount |
| gripper_base | gripper_base |
| right_driver | gripper_right_driver |
| right_coupler | gripper_right_coupler |
| right_spring_link | gripper_right_spring_link |
| right_follower | gripper_right_follower |
| right_pad | gripper_right_pad |
| left_* | gripper_left_* (同パターン) |

注意: `gripper_mount` は FT300s XML 内のボディ. FT300s なし構成では存在しない.
FT300s なしの場合, gripper は attachment_site に直接 attach されるので,
gripper のルートボディ (base_mount → gripper_base_mount) が wrist_3_link の直下に来る.
`gripper_mount` の contact exclude は不要になる.

geom 名のマッピング (contact pair 用):

| push.xml | build_ur5e_model 後 |
|---|---|
| right_pad1 | gripper_right_pad1 |
| right_pad2 | gripper_right_pad2 |
| left_pad1 | gripper_left_pad1 |
| left_pad2 | gripper_left_pad2 |
| slider_geom | slider_geom (slider XML は prefix なし or "slider_" prefix) |
| table_surface | env_table_surface |
| work_surface_geom | env_work_surface_geom |

slider の prefix と env の prefix に注意.

検証:
```python
model, data = build_push_model()
assert model.nq == 21  # arm 6 + gripper 8 + slider freejoint 7
assert model.nu == 7   # arm 6 + gripper 1
# slider が freejoint で浮いている
# ready keyframe で初期化
```

### Task C: pushing モジュールのモデルロード切替 (並列, ファイル特定済み)

以下のファイルで `from_xml_path(scene_path)` → `build_push_model()` に置換:

1. `src/ur5e_sim/pushing/task.py:81`
   - `m = mujoco.MjModel.from_xml_path(cfg.scene_path)` → `m, d = build_push_model()`
   - site/body の名前を prefix 付きに更新:
     - `"pinch"` → `"gripper_pinch"`
     - `"attachment_site"` → `"attachment_site"` (UR5e 側, 変更なし)
     - `"right_pad1"` 等 → `"gripper_right_pad1"` 等
   - keyframe 名 `"ready"` は build_push_model 内で設定

2. `src/ur5e_sim/pushing/keyframe.py:135`
   - `m = mujoco.MjModel.from_xml_path(paths.scene_path())` → `m, d = build_push_model()`
   - `"pinch"` → `"gripper_pinch"`
   - `"attachment_site"` → 変更なし

3. `src/ur5e_sim/pushing/viz/grid_video.py:55`
   - `m = mujoco.MjModel.from_xml_path(scene)` → `m, d = build_push_model()`

4. `src/ur5e_sim/pushing/config.py`
   - `scene` フィールドと `scene_path` プロパティは残すが使用箇所を削除

5. `scripts/run_push.py` — 呼び出し元の確認・更新

### Task D: テストと動作確認 (直列, A+B+C 完了後)

1. `python -m pytest tests/ -x --tb=short` — 全テスト通過確認
2. `python scripts/run_push.py` — push が動作することを確認
3. GUI でモデル表示 — ロボット構成が正しいことを目視確認
4. contact が正しく設定されている (スライダーがテーブル上で滑る, パッドでスライダーを押せる)

## 依存関係

```
Task A (model_builder 拡張)  ─┐
Task B (push scene builder)  ─┼─→ Task D (テスト)
Task C (pushing モジュール)  ─┘
```

Task A, B, C は並列実行可能. ただし B は A の API を前提とするため,
A の関数シグネチャ (ft300s_xml=None, slider 対応) を先に決めてから B を実装するか,
B が _build_spec を直接使う設計にする.

## 見積もり

| Task | 規模 | エージェント所要時間 |
|---|---|---|
| A | model_builder.py に ~30 行追加 | 10 分 |
| B | pushing/scene.py 新規 ~80 行 | 15 分 |
| C | 5 ファイル修正, 名前マッピング | 15 分 |
| D | テスト + 動作確認 | 10 分 |
| **合計** | | **A+B+C 並列 15 分 + D 10 分 ≈ 25 分** |
