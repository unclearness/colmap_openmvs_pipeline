# マネキン動画・画像からテクスチャ付きOBJ

`python mannequin_main.py INPUT OUTPUT`、またはインストール後の
`mannequin-pipeline INPUT OUTPUT` が共通の入口です。特定の動画名・カメラ姿勢・
ワールド座標のスケールには依存しません。出力先は新規または空のディレクトリを指定します。

## Windows / Ubuntu

| 工程 | Windows | Ubuntu |
|---|---|---|
| 動画のフレーム抽出 | FFmpeg | FFmpeg |
| マスク | Transformers SAM3 / CUDA | 同じ |
| SfM (`--sfm-backend auto`) | RealityScan | COLMAP 4.1.1 |
| 既存SfMの持ち込み | COLMAP text形式 | 同じ。RealityScanからのエクスポートも可 |
| 密点群 | NGC FoundationStereo dynamic 2.0 / ONNX Runtime CUDA / Open3D | 同じ |
| メッシュ | Open3D Poisson、またはTSDFメッシュ | 同じ |
| テクスチャ | OpenMVS 2.3.0、CPU | 同じバージョンのLinuxビルド、CPU |

UbuntuでRealityScanを起動する実装はありません。UbuntuのCOLMAP SfMは
RealityScanと同じアルゴリズムではありません。RealityScanのカメラを使いたい場合は
Windowsで得た`colmap/`フォルダを転送して`--sfm-root`を指定してください。
Ubuntuでの実機テストは未実施です。

OpenMVS 2.3.0をこの入口の再現設定として使い、リポジトリ全体の既定版2.4.0は変更しません。
2.4.0への自動代替はしません。CPU専用ビルドでは`--cuda-device`が存在しないため、
ヘルプの検査で省略します。CUDAを含むビルドでは`--cuda-device -2`を指定します。
根拠: [OpenMVS 2.3.0 TextureMeshのオプション定義](https://github.com/cdcseacave/openMVS/blob/v2.3.0/apps/TextureMesh/TextureMesh.cpp)。

## 環境

- 親プロセスはPython 3.11以降。WindowsのOpen3D用Pythonは3.12を指定します。
- SAM3用環境: CUDA対応PyTorch、Transformersの`Sam3Model` / `Sam3Processor`、
  huggingface-hub、Pillow、NumPy、OpenCV（headless版でも可）。
- FoundationStereo用環境: Open3D 0.19、NumPy、OpenCV、CUDA対応ONNX Runtime。
  CUDA Execution Providerが使えない場合は処理前に停止します。
- SAM3の利用同意・アクセス承認とモデルの事前取得が必要です。入口はオフラインキャッシュ
  または`--sam-model /path/to/local/snapshot`だけを使用し、自動ダウンロードしません。
  `--sam-revision COMMIT`で固定できます。
- FoundationStereo ONNXは既存の`tools/FoundationStereo-NGC-2.0/`を既定値とし、
  別配置は`--model`で指定します。SHA-256を実行前に照合します。
- FFmpeg、COLMAP 4.1.1、OpenMVS 2.3.0を事前配置します。Ubuntuでは
  `ffmpeg`、`colmap`、`TextureMesh`、`InterfaceCOLMAP`をPATHまたは各オプションで指定。
  この入口はインストール・第三者ツールのビルドを実行しません。
- Windowsでは`.venv-sam3/Scripts/python.exe`、`.venv-open3d/Scripts/python.exe`を自動検出。
  Ubuntuでは同じ環境名の`bin/python`を検出し、見つからなければ起動Pythonを使います。
  明示指定は`--sam-python`と`--stereo-python`。

各実行の`native/sam_environment.json`と`native/stereo_environment.json`にパッケージ一覧、
GPU/Provider等が残ります。環境の再構築には、成功した実行の一覧と同じバージョンを使用します。
GPU用PyTorch/ONNX RuntimeのCUDA依存は、実行先のドライバ・GPUに合ったものを用意してください。

## 実行例

Windowsの動画:

```powershell
python mannequin_main.py "C:\capture\mannequin.mp4" runs\mannequin_2fps --fps 2
```

画像フォルダ（サブフォルダも収集、JPEG/PNG/TIFF/BMP/WebP）:

```powershell
python mannequin_main.py "D:\capture\photos" runs\mannequin_photos
```

Ubuntuの動画（COLMAP SfMから）:

```bash
python mannequin_main.py /data/mannequin.mp4 runs/mannequin_linux \
  --sfm-backend colmap --fps 2 \
  --sam-python /opt/venvs/sam3/bin/python \
  --stereo-python /opt/venvs/open3d/bin/python \
  --colmap-exe /opt/colmap-4.1.1/bin/colmap \
  --openmvs-dir /opt/openmvs-2.3.0/bin
```

RealityScanのSfM結果をUbuntuへ持ち込んで密点群以降を生成:

```bash
python mannequin_main.py /data/rs/colmap/images runs/imported_rs \
  --sfm-root /data/rs/colmap --openmvs-dir /opt/openmvs-2.3.0/bin
```

採用済みメッシュとカメラでテクスチャのみ再現:

```powershell
python mannequin_main.py PATH_TO_COLMAP\images runs\retexture --sfm-root PATH_TO_COLMAP --input-mesh PATH_TO_MESH.ply
```

`--sfm-root`は`sparse/<id>/cameras.txt, images.txt, points3D.txt`を持つフォルダです。
最大登録画像数のモデルを採用します。メッシュとカメラは同じ座標系である必要があります。
持ち込み画像の名前・寸法はSfMの記録と一致させてください。

計画のみ（モデルをロードせず、成果物も作らない）:

```powershell
python mannequin_main.py "C:\capture\mannequin.mp4" runs\planned --dry-run
```

成功した設定を別入力で再利用（コマンドラインの指定が優先）:

```powershell
python mannequin_main.py "C:\capture\next.mp4" runs\next --config runs\mannequin_2fps\recipe.json --fps 4
```

`recipe.json`と参照する`stereo_overrides.json`は一緒に保存します。別PC・OSへの移動時には
`--stereo-config`と各ツール・Python・モデルのパスを実行先の配置に合わせて指定してください。

## 処理内容と調整

1. 動画から既定2 FPSで抽出。画像フォルダは名前順に全画像を使用。
   EXIF方向を画素へ反映し、最大辺1920以内・RGB・一意なPNG名へ正規化。
   元ファイルとの対応とSHA-256を保存。
2. SAM3で対象インスタンスを選択。既定プロンプトは
   `mannequin|mannequin head|cosmetology mannequin head`。
   検出しきい値は0.15、マスクしきい値は0.45。
   面積・中心との重なり・スコアで1個を選び、close/dilate後に背景を黒にします。
   マスク画像は入力画像と別フォルダに保存し、SfMへ画像として混入させません。
3. WindowsはRealityScanのBrown3・歪み事前値unknown・共有内部パラメータ・高感度設定。
   UbuntuはCOLMAPのFULL_OPENCV・共有カメラでSfM。
   COLMAPの動画はsequential、画像フォルダはexhaustiveが既定。
4. 最大コンポーネントの登録率が既定80%未満なら停止。先にFPS・撮影範囲・マスク等を確認。
   COLMAPで歪み補正画像とPINHOLEモデルを揃え、疎点群PLYを出力。
5. 歪み補正済み画像にSAM3を再適用。元画像のマスクを異なる座標の画像へ流用しません。
6. FoundationStereoで空間分散したペアを選び、ステレオ平行化、視差から深度化。
   基線比は0.003〜0.5、目標0.25、視線角差25度以内。ターンテーブル撮影に合わせた範囲です。
   全FOV、左右整合性、複数参照視点の深度・法線整合性、TSDF融合を使います。
   スケール依存のvoxel値はシーン深度に対する相対値から決めます。
7. 既定は融合点群からPoisson depth=9、密度下位1%除外、点群bboxでcrop。
   最大連結成分を保持し、元メッシュを保存したうえで25万面へ簡略化。
   非多様体の辺・頂点を整理してからテクスチャ工程へ渡します。
   整理後も多様体条件を満たさない場合は停止し、補正数を`mesh.json`に記録します。
   `--mesher tsdf`ならFoundationStereoのTSDFメッシュを使用。
8. OpenMVS 2.3.0 CPUでビュー割り当て・global/local seam leveling・アトラス生成。
   PLYの面コーナーUVを保持して標準OBJ + MTL + テクスチャへ変換します。

主要な調整:

- `--fps 4`: 動画の視点間隔を短くする。
- `--prompts "mannequin|head" --sam-threshold 0.15`: 検出しづらい後頭部など。
  低しきい値では誤検出も増えるため、保存されたマスクを確認してください。
- `--no-shared-intrinsics`: 複数カメラ、ズーム・焦点距離・画像寸法が異なる撮影。
- `--max-image-size`, `--max-pairs`, `--mesh-faces`: 計算量・形状密度。
- `--no-largest-component`: 分離した部分も保持。
- `--stereo-config SETTINGS.json`: FoundationStereoの詳細調整。
  横長画像では例えば`{"inference_width":960,"inference_height":544}`。
  高FPS等で短い基線が使える場合は`{"target_baseline_ratio":0.08,"max_baseline_ratio":0.2}`等を指定。
  入出力パス等の変更や整合性検査の無効化は受け付けません。
- `--timeout 1800`は各外部コマンドの上限で、パイプライン全体の30分完了を保証しません。

対象が剛体であること、十分な重なり・視差・全周の撮影を前提とします。
1枚の写真、未撮影の背面、激しいブラー、マネキンの変形、複数対象の同時追跡は対象外です。
フレーム数を増やすだけで未撮影面が再構築されるわけではありません。

## 再現性と過去の結果との差

テクスチャ設定は採用結果と同じです:

```text
resolution-level=0, cost-smoothness-ratio=0.1
outlier-threshold=0.006, virtual-face-images=0
global-seam-leveling=1, local-seam-leveling=1, sharpness-weight=0
```

過去の`foundationstereo_fused2_mesh.ply`を作ったメッシュ化の完全な設定記録は未発見です。
そのため、動画からの新規実行が過去OBJと同一形状になるとは主張しません。
形状まで固定した再テクスチャ検証は`--input-mesh`を使用します。
今回の新規形状生成は上記Poisson/TSDFとして方式・設定を明示しました。
FoundationStereoの左右整合性も既定で有効であり、過去の実験の無効設定とは異なります。

コマンド、設定、入力・モデル・ツール・ソースのハッシュを記録しますが、
SfMやGPU処理を含むためビット単位での同一出力は保証しません。
失敗時には`run.json`の`status=failed`、完了工程、エラーとログを残します。
途中出力への上書き・自動再開・削除は行いません。

## 出力

```text
OUTPUT/
  recipe.json                 # 次回の --config 用
  settings.json               # 解決済みツールパス等
  stereo_overrides.json       # FoundationStereo設定の再利用
  run.json                    # 状態、設定、コマンド、ハッシュ、成果物
  logs/pipeline.log
  colmap/images/
  colmap/sparse/0/             # textモデル
  sparse/sparse.ply
  mesh/mesh.obj, mesh.mtl, mesh*.png
  native/                     # マスク、SfM、深度、融合点群、元メッシュ、MVSシーン
```

密点群は`native/foundationstereo/dense/foundationstereo_fused.ply`、簡略化前の形状は
`native/geometry/mesh_full.ply`。テクスチャ画像が複数枚になった場合もOBJ/MTLから参照します。

## 検証（2026-09-08、Windows）

- 既存の採用メッシュ＋118カメラから、この入口でOBJ/MTL/アトラスを生成できることを確認。
- 元動画からSAM3、RealityScan、歪み補正、FoundationStereo、Poissonまで実行。
- 新規Poissonメッシュの非多様体接続によりOpenMVSの継ぎ目処理が異常終了するケースを検出。
  非多様体の辺186本を含む形状を整理すると、同じテクスチャ設定でOBJ生成が成功。
  整理処理を新規メッシュ生成に組み込み、回帰テストを追加。
- 検証は上記の段階実行を含みます。補正後の全工程を動画から再実行した、という意味ではありません。
- 単体テスト100件。Open3D環境では99件成功・Pillow依存1件スキップ。
  その1件はSAM3環境で成功。同名画像を含むサブフォルダ、EXIF回転、TIFF入力も検証。
- ツール診断、既存CLIのdry-run、保存recipeの再読込を確認。
- Ubuntuは実機未検証。OSによるバックエンド選択は単体テスト対象。
