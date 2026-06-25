# 未解決の技術課題

- core/ の SimRunner, UR5eRobot, Controller registry 等の高レベル抽象が production で未使用. pushing/task.py と identification/execution.py は MuJoCo を直接操作している. 将来いずれかのタスクを SimRunner に載せて妥当性を検証するか, 不要なら削除する
- `scenes/common/robots/ur5e_ft300s_gripper_no_payload.xml` は `ur5e_ft300s_gripper.xml` (345 行) のほぼ全行コピーで, ペイロード include の 1 行のみ差分. upstream の変更があっても自動追従せず, サイレントに乖離する. ロボット XML をペイロード有無で合成できる構成に変更する必要がある (命名規則は push 版との統一が完了済み; コピー構造自体は未解消)
- グリッドサーチ全 24 条件が実行不可能 (衝突余裕 ≈ -0.10). ペイロード配置が原因の可能性が高い. ロボット構成オフセット調整 (attachment_site=0.094, gripper_base_mount=0.004) 後に再実行が必要
- attachment_site 94mm vs DH パラメータ L_TP=100mm のギャップが未解消. 94mm は旧 ft300s_mount オフセット値で, 物理的な正しさが不明. 実機寸法を確認して正しい値を決定する必要がある
