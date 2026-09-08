# Multi-backend photogrammetry pipeline

マネキン動画・画像用の入口: `python mannequin_main.py INPUT OUTPUT`。
SAM3 → SfM → FoundationStereo → メッシュ化 → OpenMVS 2.3.0 CPUテクスチャを実行します。
WindowsはRealityScan、UbuntuはCOLMAP SfMまたは既存RealityScan結果の持ち込みに対応します
（Ubuntu実機テストは未実施）。[使い方・環境・再現設定](docs/mannequin_pipeline.md)。

COLMAP、OpenMVS、RealityScan、Agisoft Metashape を同じCLIから利用するWindows向け再構築パイプラインです。入力エンジン間の中間形式を無理に統一せず、すべてのバックエンドで最終的なカメラ・疎点群をCOLMAP text形式へ出力します。

## 対応範囲

| `--backend` | SfM | dense point cloud | mesh | COLMAP形式出力 |
| --- | --- | --- | --- | --- |
| `colmap` | COLMAP | COLMAP PatchMatch/Fusion | DelaunayまたはPoisson | あり |
| `openmvs` | COLMAP | OpenMVS | OpenMVS | あり |
| `realityscan` | RealityScan | 単独成果物なし | RealityScanが直接生成 | 公式export |
| `metashape` | Metashape Professional | 単独成果物なし | depth mapsから直接生成 | 公式export |

RealityScanとMetashapeは `--target sfm` または `--target mesh` のみです。`mesh` 内部でdepth mapを使用しても、dense point cloud成果物は生成しません。COLMAP形式そのものはmeshを格納できないため、meshはPLY/OBJとして併記されます。

## バージョン固定された外部ツール

- COLMAP `4.1.1`
- OpenMVS `2.4.0`
- NVIDIA NGC FoundationStereo dynamic `2.0`（実験用dense backend）

公式URL、配布variant、SHA-256、ライセンスは [`tools/manifest.json`](tools/manifest.json) に固定しています。バイナリ本体はGit管理しません。

```powershell
# 公式CUDA版をダウンロード、SHA-256検証、versioned directoryへ展開
python scripts/install_tools.py

# CPU版OpenMVS、またはno-CUDA版COLMAPを選択する例
python scripts/install_tools.py openmvs --openmvs-variant cpu
python scripts/install_tools.py colmap --colmap-variant nocuda

# 署名済みNGC ONNXを直接取得し、SHA-256検証
python scripts/install_tools.py foundationstereo
```

既定配置は `tools/COLMAP-4.1.1/`、`tools/OpenMVS-2.4.0-CUDA/` です。OpenMVS CUDA版の展開には7-Zipが必要です。既存または不完全なディレクトリをインストーラが上書きすることはありません。

## クイックスタート

Python 3.11以降を使用します。ランタイムは標準ライブラリだけで動作します。

```powershell
# インストール状況、version、edition、実行パスを確認
python -m recon_pipeline doctor

# コマンドを実行せず確認
python -m recon_pipeline run data\ET runs\et-dry `
  --backend openmvs --target mesh --preset fast --dry-run

# COLMAP SfMのみ
python -m recon_pipeline run data\ET runs\et-colmap-sfm `
  --backend colmap --target sfm

# COLMAP SfM + dense + mesh
python -m recon_pipeline run data\ET runs\et-colmap-mesh `
  --backend colmap --target mesh --preset fast --texture

# COLMAP SfM + OpenMVS dense + mesh
python -m recon_pipeline run data\ET runs\et-openmvs `
  --backend openmvs --target mesh --preset fast

# RealityScan alignment + mesh + COLMAP export
python -m recon_pipeline run data\ET runs\et-realityscan `
  --backend realityscan --target mesh --realityscan-quality normal

# Metashape Professional alignmentまで
python -m recon_pipeline run data\ET runs\et-metashape `
  --backend metashape --target sfm
```

`batch` は入力ルート直下の画像入りディレクトリを個別の出力先で処理します。

```powershell
python -m recon_pipeline batch data runs\batch-colmap `
  --backend colmap --target sfm
```

完全なオプション一覧は `python -m recon_pipeline run --help` で確認できます。

## バックエンド固有の注意

### COLMAP

- 4.1系のmatcher名 `SIFT_BRUTEFORCE` を使用します。
- `--gpu-index -1` は全GPU、`--cpu` はSfMの特徴抽出・matchingをCPUへ切り替えます。COLMAP PatchMatchはCUDA版が必要です。
- `--mesher auto` が既定です。配布物にCGAL/Delaunayが含まれればDelaunay、含まれなければPoissonを使います。明示する場合は `--mesher delaunay|poisson` です。
- `--texture` を指定すると `mesh_texturer` を実行し、標準の `mesh/<id>/textured/mesh.obj`、`mesh.mtl`、`texture.png` を生成します。COLMAP固有のper-face UV付きPLYは変換元として同じディレクトリに保持しますが、主成果物にはしません。
- 今回の公式4.1.1 CUDA配布物には `delaunay_mesher` が含まれますが、pycolmap wheelや別buildではCGAL機能が省かれる場合があります。

### OpenMVS

- `openmvs` はSfMにCOLMAPを使い、undistort後に `InterfaceCOLMAP` でOpenMVSへ渡します。
- OpenMVS 2.4ではdense cloudが `scene_dense.ply` に外出しされるため、mesh生成時に必ず `ReconstructMesh -p scene_dense.ply` を渡します。
- `--openmvs-variant auto` はCUDA版を優先します。CUDA Densifyが失敗し、公式CPU版が配置済みならCPU版へ自動フォールバックします。`cuda` を明示した場合はフォールバックしません。

### RealityScan

- 明示パス、環境変数、Epic Games manifest、既知のinstall pathの順に検出します。
- 入力をcase-insensitiveな衝突検査付きでflat化し、CLIを `appQuitOnError`、`headless`、`silent`、`stdConsole` 付きで実行します。
- `--realityscan-quality normal|high` と `--texture` を選択できます。
- 同一の固定カメラ・レンズで撮影した連番には
  `--realityscan-shared-intrinsics`を指定すると、全画像を単一の
  calibration/lens groupとしてSfMします。
- 歪みを推定する場合は、例えば
  `--realityscan-distortion-model brown3 --realityscan-distortion-prior unknown`
  を明示します。RealityScanの入力prior既定値はNo lens distortion + Approximateです。
- 公式 `exportRegistration` と同梱XMLにより、undistorted画像とCOLMAP textモデルを出力します。

### Metashape

- 自動処理はMetashape ProfessionalのPython API機能です。Standard editionでは実行前に明確なエラーを返します。
- `metashape.exe -r`、または `--metashape-python <python.exe>` で公式module入りPythonを指定できます。
- meshは `buildDepthMaps()` から `buildModel(source_data=DepthMapsData)` へ直接進み、`buildPointCloud()` は呼びません。
- `CamerasFormatColmap` の公式exporterを使い、必要に応じてpinhole変換画像も出力します。

## 出力構成

各実行には空の新規出力先が必要です。既存ディレクトリの上書き・削除はしません。

```text
OUTPUT/
  run.json                 # config、tool version、command、状態、成果物
  logs/pipeline.log
  native/                  # engine固有project/cache
  colmap/
    images/
    sparse/<model-id>/
      cameras.txt
      images.txt
      points3D.txt
  dense/<model-id>/fused.ply       # COLMAP/OpenMVSのみ
  mesh/<model-id>/...              # PLYまたはOBJ
```

COLMAP mapperが複数componentを生成した場合、すべてのSfMモデルをexportします。dense処理は登録画像3枚以上のcomponentだけが対象です。

## 検証

```powershell
python -m unittest discover -v
python -m recon_pipeline doctor --json
git diff --check
```

`data/ET` と `data/kermit` は軽量な統合テスト入力です。実ツールを使うテストでは常に `runs/` 以下の新規出力先を指定してください。

## Experimental FoundationStereo dense reconstruction

`recon_pipeline.foundation_stereo` はCOLMAP、またはRealityScanからexportしたCOLMAP text camera/pose/sparse pointを保持したまま、PatchMatch Stereoの代わりに2-view FoundationStereoを試す独立entry pointです。カメラ軌道をPCA平面へ投影して方位binごとにreferenceを均等配置し、各referenceの両側から最大2枚のsourceを選びます。その後、各ペアをOpenCVでundistort + stereo rectifyして正のdisparityに揃え、NGC ONNXでdense disparityを推論し、sparse pointでscaleを検査してから、左右整合性または別ペアとのdepth整合性でfilterし、Open3D TSDFへ融合します。

```powershell
# FoundationStereo dynamic v2.0
python scripts/install_tools.py foundationstereo

# Open3D環境へ公式ONNX Runtime CUDA/cuDNN runtimeを追加
uv pip install --python .venv-open3d\Scripts\python.exe `
  onnx==1.22.0 "onnxruntime-gpu[cuda,cudnn]==1.29.0"

# 既存COLMAP SfMを使う全周実験。LR checkを外してcoverageを確保し、
# 代わりに別viewのdepth一致を必須にする例
.venv-open3d\Scripts\python.exe -m recon_pipeline.foundation_stereo `
  runs\mannequin-colmap-textured\colmap\sparse\0 `
  runs\mannequin-video-frames\images `
  runs\foundationstereo-mannequin `
  --spatial-bins 24 --references-per-bin 2 `
  --sources-per-reference 2 --max-pairs 24 --no-lr-check `
  --colmap-fused runs\mannequin-colmap-textured\native\colmap\dense\0\fused.ply
```

動画の時間sampleを増やしてRealityScan SfMを使う例:

```powershell
ffmpeg -i video1.mp4 -vf fps=4 -q:v 2 -start_number 0 `
  runs\frames-4fps\images\v1_%05d.jpg
ffmpeg -i video2.mp4 -vf fps=4 -q:v 2 -start_number 0 `
  runs\frames-4fps\images\v2_%05d.jpg

python -m recon_pipeline run runs\frames-4fps\images runs\rs-sfm-4fps `
  --backend realityscan --target sfm --timeout 14400

.venv-open3d\Scripts\python.exe -m recon_pipeline.foundation_stereo `
  runs\rs-sfm-4fps\colmap\sparse\0 `
  runs\rs-sfm-4fps\colmap\images `
  runs\rs-foundationstereo-4fps `
  --spatial-bins 48 --references-per-bin 4 `
  --sources-per-reference 2 --max-pairs 96 --no-lr-check

# 保存済みdepthを再推論せず、3視点depth + 25度normal一致で強く再fusion
.venv-open3d\Scripts\python.exe -m recon_pipeline.foundation_stereo_refusion `
  runs\rs-foundationstereo-4fps `
  runs\rs-foundationstereo-4fps-depth-normal `
  --min-consistent-views 3 `
  --consistency-relative-tolerance 0.01 `
  --normal-consistency-angle 25
```

主な成果物:

- `pairs/*/left.png`, `right.png`: rectification確認画像
- `pairs/*/disparity.npy`, `depth.npy`: raw float32結果
- `pairs/*/valid.png`, `consistent.png`: pair内／multi-view filter mask
- `dense/foundationstereo_fused.ply`: 色付き融合点群
- `dense/hybrid_colmap_foundationstereo_fused.ply`: `--colmap-fused` 指定時の補完union
- `mesh/foundationstereo_tsdf_raw.ply`: component削除前のTSDF mesh
- `mesh/foundationstereo_tsdf.ply`: 選択した後処理後mesh
- `run.json`: pair geometry、sparse disparity残差、有効率、モデルSHA-256

`foundation_stereo_refusion` は保存済み `depth.npy` を使うためONNX推論を繰り返しません。depthから局所3D normalを推定してworld座標へ変換し、カメラ側へ向きを統一します。別viewへ再投影したdepthが3視点以上で一致し、normal差も指定角以内のpixelだけをTSDFへ再統合します。出力は `dense/foundationstereo_fused_depth_normal_consistent.ply` です。

保存depthの閾値を少し緩め、view単位のpoint-to-plane ICPをpose graphで
全体最適化する実験entry pointもあります。RealityScan poseの最小全域木を強い
priorとして残し、初期変換から大きく外れる対称形状の誤ICPは除外します。
`--optimizer leave-one-out` では、基準viewを1つ固定し、各viewを残りN-1 viewの
現在の統合点群へ逐次ICPする座標降下法に切り替えます。各更新は直ちに次の
N-1 targetへ反映されます。`--loo-tukey-k` でpoint-to-plane残差にTukey lossを
適用できます。

```powershell
.venv-open3d\Scripts\python.exe foundation_stereo_icp_refusion_main.py `
  runs\rs-foundationstereo-4fps `
  runs\rs-foundationstereo-4fps-relaxed-icp

.venv-open3d\Scripts\python.exe foundation_stereo_icp_refusion_main.py `
  runs\rs-foundationstereo-4fps `
  runs\rs-foundationstereo-4fps-relaxed-loo-icp `
  --optimizer leave-one-out --loo-iterations 3 `
  --icp-max-correspondence 0.15 --loo-target-voxel-size 0.10 `
  --loo-tukey-k 0.075 --loo-max-step-translation 0.10 `
  --loo-max-step-rotation 1.0
```

既定値は4 view、depth相対0.5%＋絶対0.5 voxel、normal 25°です。
ただしICP後は必ず同一camera投影でsurface厚みを比較してください。部分viewの
重複と対称性が強い場合、ICPは二重面を減らさず悪化させることがあります。
マネキン実験ではN-1 target自体に複数の面層が残り、leave-one-out ICPもviewごとに
異なる層へ収束して厚みを増やしました。この方式を品質検査なしで採用しないでください。

NGC `deployable_foundation_stereo_s_dynamic_v2.0` はNVIDIA Open Model Licenseの商用利用可能モデルです。入力はrectified stereoでなければならず、任意のSfM画像をそのまま2枚渡すことはできません。モデル自身はconfidenceを出さないため、この実装は異常なrectification、416 pxを超える期待disparity、depth範囲、multi-view不一致を除外します。静止物体を別時刻から撮った画像は使えますが、動く対象には同期stereoが必要です。点群／mesh PLYは未テクスチャ形状です。後段でtextureを付ける場合の主成果物は従来どおりOBJ + MTL +画像にしてください。

`doctor` は公開4バックエンドの実行エンジンだけを表示します。FoundationStereoは `scripts/install_tools.py foundationstereo` の `Ready` 表示と、各実行時のモデルSHA-256検証・`run.json` で確認します。CUDAを既定にしており、CPUへ暗黙fallbackしません。CPUを明示する場合は `--provider cpu` ですが、実用的な速度は期待できません。

## Femto Bolt / K4A RGB-D fusion

K4A-compatible MKVには、COLMAP系とは独立したOpen3D entry pointを使えます。Open3Dのframe-to-model dense SLAMでposeとTSDF meshを生成し、undistorted RGB keyframeとposeをCOLMAP textへ変換してからOpenMVS `TextureMesh`へ渡します。

```powershell
# Playback runtime。Femto Bolt device接続用途ではOrbbec wrapperも導入可能
python scripts/install_tools.py azure-kinect orbbec-k4a

# Open3D 0.19のWindows wheelはPython 3.12を使用
uv venv .venv-open3d --python 3.12
uv pip install --python .venv-open3d\Scripts\python.exe `
  open3d==0.19.0 opencv-python-headless

.venv-open3d\Scripts\python.exe -m recon_pipeline.rgbd_fusion `
  recording.mkv runs\rgbd-result `
  --frame-step 3 --texture-frame-step 15 --texture-view-count 36

# 既存trajectoryを使い、最初の約60秒からマネキン領域だけを1mmで再統合
.venv-open3d\Scripts\python.exe -m recon_pipeline.rgbd_fusion `
  recording.mkv runs\rgbd-mannequin-1mm `
  --roi-from runs\rgbd-result --roi-source-frame-max 1800 `
  --roi-min -0.22 -0.24 0.28 --roi-max 0.14 0.0 0.68 `
  --roi-voxel-size 0.001 --texture-view-count 36
```

主な成果物:

- `trajectory.json`: Open3D camera-to-world pose
- `calibration.json`: K4A raw calibrationから導出したundistorted intrinsics
- `colmap/images` と `colmap/sparse/0`: OpenMVSへ渡せるPINHOLE camera model
- `mesh/open3d_mesh.ply` または `mesh/open3d_mesh_1mm_roi.ply`: TSDFから抽出した高密度形状mesh
- `mesh/open3d_mesh_texturing.ply`: OpenMVS用の簡略mesh
- `mesh/textured/mesh.obj`、`mesh.mtl`、texture image

Open3D 0.19のdense SLAMにはloop closure/relocalizationがないため、長い周回ではdriftが残ります。OpenMVS 2.4 Windowsでは外部mesh/poseに対するseam levelingがクラッシュするケースがあるため、このentry pointはglobal/local seam levelingを無効化します。`--texture-view-count` はmesh中心を向く鮮明な画像を方向分散付きで選び、細かなpatchの増加を抑えます。OpenMVSのtextured PLYはper-face UV拡張で一般的なviewerとの互換性が低いため、主成果物は標準OBJ+MTLです。SLAM後にtexturingだけ再実行する場合は同じ出力先へ `--resume-texture` を指定します。

## RealityScan + ToF + FoundationStereo hybrid

`rgbd_hybrid_main.py` はK4A-compatible MKVの同期RGB/ToFを共通の
undistorted画面へ抽出し、RealityScan SfM、ToFによるmetric scale、
FoundationStereo depth、ToF優先のsingle-depth arbitration、TSDFを段階実行します。
RealityScan export画像の再投影は `K_export @ inv(K_factory)` でToFにも適用し、
入力画像座標のdepthをexport cameraへ直接混ぜません。

```powershell
.venv-open3d\Scripts\python.exe rgbd_hybrid_main.py recording.mkv runs\hybrid `
  --frame-step 5 --icp-refine --texture

# 長時間処理の段階確認と再開
.venv-open3d\Scripts\python.exe rgbd_hybrid_main.py recording.mkv runs\hybrid `
  --frame-step 5 --stop-after metric
.venv-open3d\Scripts\python.exe rgbd_hybrid_main.py recording.mkv runs\hybrid `
  --frame-step 5 --resume --icp-refine

# SfM poseの品質をFoundationStereo前に検査するToF-only baseline
.venv-open3d\Scripts\python.exe rgbd_hybrid_main.py recording.mkv runs\hybrid `
  --resume --tof-only --hybrid-name tof-baseline
```

hybrid arbitrationはToFが有効なpixelでは必ずToFを採用します。ToF欠損部の
stereoは、複数候補が競合せず、近傍ToF面から指定許容差内の場合だけ補完します。
RGB GrabCut、近側の肌色ToF seed、depth連結成分を交差し、背景を抑制します。
中間meshはPLYですが、`--texture` の最終成果物はOBJ + MTL + texture画像です。

1mm TSDFの前には必ず `--tof-only` を確認してください。ToF-onlyで多重面が出る場合、
原因はstereo depthではなくposeです。現在の実験用ICPは時間隣接edgeと近い既登録viewの
loop edgeを追加しますが、RealityScan componentが断片化した記録を完全に修復する
pose-graph optimizerではありません。この品質ゲートを通らない結果へtextureを実行しないでください。
