# 未解決の技術課題

- core/ の SimRunner, UR5eRobot, Controller registry 等の高レベル抽象が production で未使用. pushing/task.py と identification/execution.py は MuJoCo を直接操作している. 将来いずれかのタスクを SimRunner に載せて妥当性を検証するか, 不要なら削除する
- `scenes/common/robots/ur5e_ft300s_gripper_no_payload.xml` は `ur5e_ft300s_gripper.xml` (345 行) のほぼ全行コピーで, ペイロード include の 1 行のみ差分. upstream の変更があっても自動追従せず, サイレントに乖離する. ロボット XML をペイロード有無で合成できる構成に変更する必要がある (命名規則は push 版との統一が完了済み; コピー構造自体は未解消)
- グリッドサーチ全 24 条件が実行不可能 (余裕 ≈ -0.0000). 違反制約は主に EE 速度上限と payload_workspace (ペイロードがワークスペース境界を微妙にはみ出す). 条件数は全条件で有限 (6.2–11.2) になったが, 実行可能解なし
- FIM 相関行列でバイアス-重力結合が高い: b_tz↔m 相関 0.98, b_fx↔hy 0.98, b_fy↔hx -0.97. ポーズ変動が矢状面に偏り, バイアスと重力の分離が不十分. EE 速度・軌道時間・初期姿勢の変更では改善しない (根本原因: Fourier 境界が dq_max/ddq_max で決まり, dq ≤ π rad/s のハードウェア制限内では探索空間が不足). D-optimality を目的関数にすることで改善できる可能性があるが未検証
- attachment_site 94mm vs DH パラメータ L_TP=100mm のギャップが未解消. 94mm は旧 ft300s_mount オフセット値で, 物理的な正しさが不明. 実機寸法を確認して正しい値を決定する必要がある
