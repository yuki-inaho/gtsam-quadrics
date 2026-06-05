# LLMオンボーディングサマリー

この文書は、新任 LLM エージェントが `gtsam-quadrics` の現在状態を誤読せずに作業へ入るための引き継ぎ資料です。特に、直近で追加した Pixi ベースのビルド環境について、何が動作確認済みで、何が未実施かを明確にします。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** `gtsam-quadrics`。GTSAM 上で constrained dual quadric landmark と 2-D bounding box factor を扱う C++ / Python binding 混在ライブラリ。
- **最終成果物:** C++ ライブラリ、GTSAM Quadrics Python extension、README に記載された再現可能な開発・ビルド手順。
- **ビジネス背景・価値:** ロボティクス / SLAM 系研究コードを、Ubuntu の `apt` 依存だけに頼らず、Pixi / conda-forge 環境で再現可能にビルドできるようにする。
- **現時点の進捗サマリ:** commit `79dc454 Add Pixi build environment` で Pixi 環境、lock file、README の Pixi 手順、CMake の Python binding ビルド調整が `origin/master` に反映済み。
- **動作判定:** 動く。現行 GTSAM submodule `69a3a75195b65356d6e56669e5199d325c7962c9`、タグ `4.1.1` 前提で、Pixi による configure/build と Python import smoke test は成功済み。

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ライン。
- GTSAM submodule は現時点では `4.1.1` のまま。最新版 GTSAM への更新はこの状態の完了条件から外している。
- Python は `>=3.9,<3.11` に制約している。理由は、同梱 GTSAM / pybind11 世代が Python 3.11 の frame API 変更に未対応だったため。
- Pixi 環境は `conda-forge`、`linux-64`、`pixi.lock` で固定する。`pixi.lock` は成果物としてバージョン管理する。
- `.pixi/` と `build/` 系の生成物はコミットしない。
- CMake では Python wrapper ビルド時に LTO を無効化している。`gtsam_py` の link が非常に重くなる問題を避けるため。
- `BUILD_DOCS` は切り替え可能にしている。GTSAM doc target との衝突を避けるため、doc 生成は明示的に管理する。
- 現行 Pixi task の `build` は `gtsam_py`、`gtsam_quadrics`、`gtsam_quadrics_py` を対象にしている。C++ examples のビルド・実行は現時点の検証済み範囲ではない。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| リポジトリ規約 | `AGENTS.md` | `/home/inaho-omen/.codex/RTK.md` を参照する入口。shell 操作では `rtk` prefix を使う。 |
| Pixi manifest | `pixi.toml` | Python / CMake / Ninja / compiler / Boost / METIS / NumPy と task 定義。 |
| Pixi lock | `pixi.lock` | 再現可能な conda-forge 環境固定。 |
| ビルド設定 | `CMakeLists.txt` | GTSAM submodule 追加、Python wrapper、LTO 抑制、doc guard などの実装箇所。 |
| Python wrapper template | `gtsam_quadrics/gtsam_quadrics.tpl` | `cstdint` include を追加済み。 |
| 利用手順 | `README.md` | Pixi install/build 手順を追加済み。 |
| submodule 定義 | `.gitmodules` | GTSAM submodule は `https://github.com/borglab/gtsam`。 |
| C++ examples | `examples/c++/` | example source は存在するが、現行 Pixi task の検証対象外。 |
| Python examples | `examples/python/` | Python example source は存在するが、現行 smoke test は import 確認まで。 |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- Pixi 環境の再現確認、`pixi.lock` の更新、依存バージョン制約の調整。
- CMake configure/build の失敗原因調査と、既存構成に沿った最小修正。
- Python binding の import smoke test。
- README / onboarding の事実ベース更新。
- GTSAM `4.1.1` 前提でのビルド互換性維持。

### 任せないタスク
- 明示要求なしに GTSAM submodule を最新版へ更新すること。
- 明示要求なしに examples を完了済み扱いにすること。
- `git reset --hard` や checkout によるユーザー変更の破棄。
- 生成物 `.pixi/`、`build/`、local tool install path のコミット。
- 未検証の動作を「確認済み」と書くこと。

## 5. インタラクション方針
- **回答スタイル:** 日本語で簡潔に、結論を先に述べる。動く / 動かないのような判定は曖昧にしない。
- **回答手順:** 現在状態、実施内容、検証結果、未実施範囲の順で説明する。
- **禁止事項・注意:** 最新 GTSAM 対応、examples 実行、full check を未実施のまま完了扱いしない。
- **秘匿情報の扱い:** ローカル path や commit hash は作業上必要な範囲で記載してよい。認証情報、token、private key は記載しない。

## 6. 試行タスク（オンボーディング演習）
1. `git status --short --branch` と `git submodule status --recursive` を確認し、GTSAM が `4.1.1` のままか説明する。
2. `pixi run configure` と `pixi run build` の task 定義を `pixi.toml` から読み、何をビルドするか説明する。
3. Python import smoke test のコマンドを確認し、`gtsam` と `gtsam_quadrics` が import できることを検証する。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 「確認済み」と書く場合は、対応する command / commit / file を併記する。
- **TBDの扱い:** GTSAM 最新化、Justfile 追加、examples build/run 対応は、要求が復活したときに別タスクとして扱う。
- **レビュー/承認フロー:** 破壊的 git 操作や submodule 大幅更新は、要求が明確な場合のみ実施する。
- **その他の運用ルール:** shell command は `rtk` prefix で実行する。手編集は `apply_patch` を使う。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:** `/home/inaho-omen/Project/gtsam-quadrics`
- **GitHub remote:** `git@github.com:yuki-inaho/gtsam-quadrics.git`
- **現在の HEAD:** `79dc454 Add Pixi build environment`
- **現在の GTSAM submodule:** `69a3a75195b65356d6e56669e5199d325c7962c9` (`4.1.1`)
- **代表的なコマンド:**
  - `rtk pixi lock`
  - `rtk pixi install --locked`
  - `rtk pixi run configure`
  - `rtk pixi run build`
  - `PYTHONPATH=build:build/gtsam/python LD_LIBRARY_PATH=build:build/gtsam/gtsam:build/gtsam/gtsam/3rdparty/metis/libmetis:$LD_LIBRARY_PATH pixi run python -c "import gtsam; import gtsam_quadrics; print(gtsam.__name__, gtsam_quadrics.__name__)"`
- **依存ライブラリ:** Python `<3.11`、CMake、Ninja、C/C++ compiler、Boost C++、METIS、NumPy、pip/setuptools/wheel、pyparsing。
- **検証済み:** `pixi lock`、`pixi install --locked`、`pixi run configure`、`pixi run build`、Python import smoke test。検証時は一時インストールした Pixi CLI `pixi 0.70.1` を使用した。
- **未検証:** `pixi run check` の full test、C++ examples の build/run、Python examples の実行、GTSAM 最新版への submodule 更新。
