# Multi-backend photogrammetry pipeline

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

公式URL、配布variant、SHA-256、ライセンスは [`tools/manifest.json`](tools/manifest.json) に固定しています。バイナリ本体はGit管理しません。

```powershell
# 公式CUDA版をダウンロード、SHA-256検証、versioned directoryへ展開
python scripts/install_tools.py

# CPU版OpenMVS、またはno-CUDA版COLMAPを選択する例
python scripts/install_tools.py openmvs --openmvs-variant cpu
python scripts/install_tools.py colmap --colmap-variant nocuda
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
  --backend colmap --target mesh --preset fast

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
- 今回の公式4.1.1 CUDA配布物には `delaunay_mesher` が含まれますが、pycolmap wheelや別buildではCGAL機能が省かれる場合があります。

### OpenMVS

- `openmvs` はSfMにCOLMAPを使い、undistort後に `InterfaceCOLMAP` でOpenMVSへ渡します。
- OpenMVS 2.4ではdense cloudが `scene_dense.ply` に外出しされるため、mesh生成時に必ず `ReconstructMesh -p scene_dense.ply` を渡します。
- `--openmvs-variant auto` はCUDA版を優先します。CUDA Densifyが失敗し、公式CPU版が配置済みならCPU版へ自動フォールバックします。`cuda` を明示した場合はフォールバックしません。

### RealityScan

- 明示パス、環境変数、Epic Games manifest、既知のinstall pathの順に検出します。
- 入力をcase-insensitiveな衝突検査付きでflat化し、CLIを `appQuitOnError`、`headless`、`silent`、`stdConsole` 付きで実行します。
- `--realityscan-quality normal|high` と `--texture` を選択できます。
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
