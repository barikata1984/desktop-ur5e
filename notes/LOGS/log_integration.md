# 統合作業ログ

## 2026-06-19 — ur5e-sim 統合パッケージ構築完了

### 完了した作業

**1. プロジェクト基盤構築**

pyproject.toml (ur5e-sim パッケージ定義), pixi.toml (全依存集約), ディレクトリ構造 `src/ur5e_sim/{core,pushing,identification,trajectories}` を作成した.
外部アセットとして mujoco_menagerie submodule と ft300s を assets/ に配置.
scenes/, scripts/, configs/, tests/, notes/ も同時に作成した.

**2. モジュラー MuJoCo シーン XML 構成**

7 ファイルを作成した.

- scenes/common/environment.xml
- scenes/robots/ur5e_gripper.xml
- scenes/robots/ur5e_ft300s_gripper.xml
- scenes/objects/slider.xml
- scenes/objects/payload_box.xml
- scenes/tasks/push.xml
- scenes/tasks/identification.xml

両シーン (push, identification) のロード・ステップ動作を検証済み.

**3. core/ 共通シミュレーション基盤**

14 ファイルを作成した: env.py, robot.py, ik.py, sensors.py, renderer.py, types.py, runner.py, controllers/base.py, controllers/registry.py, logging/base.py, logging/npz.py, __init__.py 群.
全モジュールの import を検証済み.

**4. pushing/ 2D push タスクコード移行**

12 ファイルを移行した: mpc.py, kinematics.py, config.py, task.py, keyframe.py, io.py, paths.py, analytical/push_com_sim.py, viz/plots.py, viz/grid_video.py 等.
import パスを pusher_slider → ur5e_sim.pushing に変更した. ロジックは変更なし.

**5. identification/ 慣性同定コード移行**

22 ファイルを移行した. trajectories/ も 7 ファイルを移行.
import パスを mjwarp_ur5e → ur5e_sim に変更した. ロジックは変更なし.

**6. scripts/ と configs/ の作成**

7 スクリプトを作成した: run_push.py, run_identification.py, make_keyframe.py, analytical_push.py, render_video.py, optimize_trajectory.py, setup_assets.sh.
設定ファイルは push_default.yaml, identification_default.yaml の 2 ファイル.

**7. テスト移行・検証**

結果: 94 passed, 10 skipped (warp GPU), 14 xfailed.
ruff check / ruff format ともにクリーン.

### 設計決定

- pusher-slider のシーン (テーブル + 作業板 + カメラ) を環境ベースとした
- MuJoCo シーンは `<include>` でモジュラーに合成する設計 (environment + robot + object = task scene)
- SimRunner は `controller.compute_control(state: dict) -> ctrl(np.ndarray)` 契約とした. push の PusherSliderMPC はシグネチャが異なるため, アダプタ controller で包む設計とした
- load_model はパス必須とし, assets モジュールへの依存を切断した
- payload_box.xml は ur5e_ft300s_gripper.xml の gripper_mount 内に include することで剛体接続を保証した
- meshdir はトップレベルシーン (scenes/tasks/) からの相対パスで解決する. robot/object フラグメント単独でのロードは不可

### インフラ

リモートオリジンを git@github.com:barikata1984/desktop-ur5e.git に設定した.

### 発見された課題

なし.
