# 未解決の技術課題

- core/ の SimRunner, UR5eRobot, Controller registry 等の高レベル抽象が production で未使用. pushing/task.py と identification/execution.py は MuJoCo を直接操作している. 将来いずれかのタスクを SimRunner に載せて妥当性を検証するか, 不要なら削除する
- `scenes/common/robots/ur5e_ft300s_gripper_no_payload.xml` は `ur5e_ft300s_gripper.xml` (345 行) のほぼ全行コピーで, ペイロード include の 1 行のみ差分. upstream の変更があっても自動追従せず, サイレントに乖離する. ロボット XML をペイロード有無で合成できる構成に変更する必要がある
