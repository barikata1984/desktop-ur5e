# 未解決の技術課題

- core/ の SimRunner, UR5eRobot, Controller registry 等の高レベル抽象が production で未使用. pushing/task.py と identification/execution.py は MuJoCo を直接操作している. 将来いずれかのタスクを SimRunner に載せて妥当性を検証するか, 不要なら削除する
