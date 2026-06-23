# TODO

- [x] プロジェクト基盤構築 (pyproject.toml, pixi.toml, ディレクトリ構造)
- [x] モジュラー MuJoCo シーン XML 構成
- [x] core/ 共通シミュレーション基盤の実装
- [x] pushing/ 2D push タスクコードの移行
- [x] identification/ 慣性同定コードの移行
- [x] scripts/ と configs/ の作成
- [x] テストの移行と全体検証
- [x] devcontainer postCreateCommand の `python --version` 失敗を修正
- [x] インタラクティブ軌道設計ツール (IK ベースコーンスイープ) の実装
- [ ] ロボット XML をペイロードのオプション化でリファクタリング (ur5e_ft300s_gripper_no_payload.xml の重複解消)
- [x] グリッパ XML の命名規則統一 (push 版と FT300s 版で `gripper_` プレフィックスを除去して揃える)
- [x] MPC プッシュループにリアルタイム進捗表示 (`[XX.X%]`) を追加
