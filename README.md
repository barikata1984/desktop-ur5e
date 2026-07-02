# ur5e-sim

UR5e MuJoCo シミュレーション環境. 2D pusher-slider マニピュレーションと慣性パラメータ同定を単一パッケージで提供する.

## 構成

```
src/ur5e_sim/
├── core/               共通シミュレーション基盤 (env, robot, IK, sensors, renderer)
├── pushing/            2D push タスク (Hogan-2016 FOM MPC)
├── identification/     慣性パラメータ同定 (Newton-Euler regressor, TLS)
└── trajectories/       軌道表現 (Fourier, quintic spline)

scenes/
├── common/             環境 (テーブル, カメラ) + ロボット構成 (FT300s 有無)
├── objects/            タスク対象物 (slider, payload)
└── tasks/              合成シーン (push.xml, identification.xml)

assets/
├── mujoco_menagerie/   UR5e + Robotiq 2F-85 (git submodule)
└── ft300s/             FT300s 力覚センサ

scripts/                CLI エントリポイント
configs/                デフォルト設定 (YAML)
tests/                  テストスイート
```

## セットアップ

```bash
# サブモジュール取得
git submodule update --init

# pixi で環境構築 (コンテナ内で自動実行済み)
pixi install
```

## 使い方

### Push シミュレーション

```bash
python scripts/run_push.py
python scripts/run_push.py --push.y-goal 0.7 --mpc.v-max 0.1
```

出力: `results/<timestamp>/push_sim.mp4`, `data.npz`, `config.json`

### 慣性パラメータ同定

```bash
# 1. 励振軌道の最適化
python scripts/optimize_trajectory.py

# 2. 同定 (軌道再生 → 推定)
python scripts/run_identification.py --result-json results/excitation_result.json

# 3. 動画レンダリング
python scripts/render_excitation.py --result-json results/excitation_result.json
```

### その他

```bash
python scripts/analytical_push.py    # Stage-0 解析シミュレーション
python scripts/make_keyframe.py      # ready キーフレーム再生成
python scripts/render_video.py       # trial の 2x2 グリッド動画
```

## シーン構成

モデルは `MjSpec.attach()` によるプログラム的合成で組み立てる.
アームとグリッパは mujoco_menagerie から, FT300s とペイロード, 環境は `scenes/` 以下の断片から読み込む.

| ビルダ | ロボット | 対象物 |
|---|---|---|
| `build_push_model()` | UR5e + Robotiq 2F-85 | スライダ (80x60x30mm, 1.05kg) |
| `build_ur5e_model(payload_xml=...)` | UR5e + FT300s + Robotiq 2F-85 | ペイロード直方体 |

## テスト

```bash
python -m pytest tests/ -v
```

94 passed, 10 skipped (warp GPU), 14 xfailed

## 依存関係

- Python 3.10+
- MuJoCo >= 3.8
- NumPy, SciPy, Matplotlib, Pillow
- tyro (CLI), PyYAML, imageio[ffmpeg]
- warp-lang (オプション, GPU 高速化)
