# Spring-massレンダリング実行例

CUDA対応のKubric Dockerイメージを使い、次の2パターンをレンダリングする例を示す。

1. 物体をただ落とす
2. 1つのcontrol pointを持ち上げて落とす処理を繰り返す

すべてのコマンドはKubricリポジトリのルートで実行する。

## 前提条件

- Docker
- NVIDIA Container Toolkit
- 対応するNVIDIA GPU

現在のDockerfileは、デフォルトではCUDA 11.8および`sm_70`（Tesla V100）向けに
Blender CyclesとPyTorchをビルドする。他のGPUを使用する場合は、必要に応じて
`CUDA_ARCH`と`TORCH_CUDA_ARCH_LIST`のビルド引数を変更する。

## Dockerイメージのビルド

```bash
docker build \
  -f docker/BlenderCuda118Torch.Dockerfile \
  -t kubruntu-blender-torch:cu118 \
  .
```

このビルドでは、`kubricdockerhub/blender:latest`と
`kubricdockerhub/kubruntu:latest`をベースイメージとして使用する。ローカルに存在しない
場合はDockerが自動的に取得する。

必要に応じて、BlenderとPyTorchからCUDAを利用できることを確認する。

```bash
docker run --rm \
  --runtime=nvidia \
  --user "$(id -u):$(id -g)" \
  --env NVIDIA_VISIBLE_DEVICES=0 \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env KUBRIC_USE_GPU=true \
  --env KUBRIC_CYCLES_BACKEND=CUDA \
  --env PYTHONPATH=/kubric \
  --volume "$PWD:/kubric" \
  --workdir /kubric \
  kubruntu-blender-torch:cu118 \
  python3 docker/verify_cuda118.py
```

## 例1: ただ落とす

`--repeat_count=0`を指定するとcontrol pointによる持ち上げが無効になる。次の例では、
物体を離してから5秒間レンダリングする。

```bash
mkdir -p output/spring_mass_drop

docker run --rm \
  --runtime=nvidia \
  --user "$(id -u):$(id -g)" \
  --env NVIDIA_VISIBLE_DEVICES=0 \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env KUBRIC_USE_GPU=true \
  --env KUBRIC_CYCLES_BACKEND=CUDA \
  --env PYTHONPATH=/kubric \
  --volume "$PWD:/kubric" \
  --volume "$(readlink -f output):/output" \
  --workdir /kubric \
  kubruntu-blender-torch:cu118 \
  python3 examples/gso_spring_mass.py \
    --asset_id=Crosley_Alarm_Clock_Vintage_Metal \
    --particle_spacing=0.15 \
    --k_neighbors=64 \
    --spring_stiffness=20 \
    --damping=0.01 \
    --step_rate=2400 \
    --camera_count=1 \
    --randomize_cameras \
    --repeat_count=0 \
    --initial_settle_seconds=5.0 \
    --seed=42 \
    --job-dir=/output/spring_mass_drop \
    --scratch_dir=/tmp/kubric_spring_mass_drop
```

## 例2: 5回持ち上げて落とす

`--control_vertex_index`で指定したレンダーメッシュ頂点をcontrol pointとして使用する。
最初の落下後、同じ点をシミュレーション開始時の高さまでsmoothstep補間で持ち上げ、
同時にワールド座標`(x, y) = (0, 0)`へ移動する。短時間保持してから解放し、すべての
繰り返しで同じ物質点を使用する。

```bash
mkdir -p output/spring_mass_repeated_lift

docker run --rm \
  --runtime=nvidia \
  --user "$(id -u):$(id -g)" \
  --env NVIDIA_VISIBLE_DEVICES=0 \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env KUBRIC_USE_GPU=true \
  --env KUBRIC_CYCLES_BACKEND=CUDA \
  --env PYTHONPATH=/kubric \
  --volume "$PWD:/kubric" \
  --volume "$(readlink -f output):/output" \
  --workdir /kubric \
  kubruntu-blender-torch:cu118 \
  python3 examples/gso_spring_mass.py \
    --asset_id=Crosley_Alarm_Clock_Vintage_Metal \
    --particle_spacing=0.15 \
    --k_neighbors=64 \
    --spring_stiffness=20 \
    --damping=0.01 \
    --step_rate=2400 \
    --control_vertex_index=0 \
    --control_target_x=0 \
    --control_target_y=0 \
    --camera_count=4 \
    --randomize_cameras \
    --repeat_count=5 \
    --initial_settle_seconds=2.0 \
    --lift_seconds=1.0 \
    --hold_seconds=0.2 \
    --settle_seconds=2.0 \
    --seed=42 \
    --job-dir=/output/spring_mass_repeated_lift \
    --scratch_dir=/tmp/kubric_spring_mass_repeated_lift \
    --render_depth \
    --render_segmentation
```

総レンダリング時間は次の式から自動計算される。

```text
initial_settle_seconds
  + repeat_count * (lift_seconds + hold_seconds + settle_seconds)
```

`--camera_count=4 --randomize_cameras`では、同じ頂点アニメーションを4つの異なる視点から
レンダリングする。4台が近い方向へ偏らないよう、まずZ軸周りに90度間隔で配置し、その
リグ全体の方位角をランダムに回転する。その後、各カメラへ次の範囲の揺らぎを加える。

- 90度間隔からの方位角の揺らぎ: ±8度
- 原点からの水平距離: 3.8から4.2
- カメラの高さ: 2.45から2.75
- 注視点: `(0, 0, 1)`の周辺

カメラ乱数は`seed`と`asset_id`から作る。同じ組み合わせでは同じ配置を再現でき、seedまたは
GSOオブジェクトが変わるとカメラ位置と姿勢も変化する。同じオブジェクトで別のカメラ配置を
作る場合は`--seed`を変更する。実際に使用した位置、注視点、camera seedは
`metadata.json`へ保存される。

`--randomize_cameras`を指定しない場合は、次の固定配置を使用する。

```text
camera_00: ( 0.0, -4.0, 2.6)
camera_01: ( 4.0,  0.0, 2.6)
camera_02: ( 0.0,  4.0, 2.6)
camera_03: (-4.0,  0.0, 2.6)
```

床面の衝突位置と初期落下位置を画角に残しながら、従来の配置より物体が大きく写る距離に
している。結果は`<job-dir>/<asset_id>/camera_00/`から`camera_03/`へ分けて出力される。
asset IDのディレクトリは存在しない場合に自動作成される。指定できるカメラ数は1から4で、
1の場合は`<job-dir>/<asset_id>/`直下へフレームを保存する。

上記の例では、両方のコマンドに同じ
`--asset_id=Crosley_Alarm_Clock_Vintage_Metal`を指定している。別の物体を使用する場合は、
この値をGSO manifestに含まれるasset IDへ置き換える。同じseedを指定すると初期姿勢も
再現される。

このリポジトリでは`output`がリポジトリ外へのシンボリックリンクになっている場合がある。
`--volume "$(readlink -f output):/output"`はリンク先の実体をコンテナへmountするために必要で
ある。出力先にはコンテナ内の`/output/...`を指定する。

## 出力

指定した`job-dir`の下にasset IDのディレクトリが作成され、次のように出力される。

```text
<job-dir>/
└── <asset_id>/
    ├── camera_00/rgba_00000.png, ...
    ├── camera_01/rgba_00000.png, ...
    ├── camera_02/rgba_00000.png, ...
    ├── camera_03/rgba_00000.png, ...
    ├── gso_spring_mass.blend
    └── metadata.json
```

- `camera_XX/rgba_00000.png`, ...: 各カメラでレンダリングされたフレーム
- `gso_spring_mass.blend`: 頂点アニメーションを含むBlenderシーン
- `metadata.json`: アセット、粒子、ばね、control point、各カメラのメタデータ

`metadata.json`の各`cameras`要素には、次のカメラパラメータが含まれる。

- 内部パラメータ: 解像度、焦点距離、センサーサイズ、画角、主点
- Kubric正規化内部行列
- OpenCVピクセル座標系の内部行列
- カメラ位置、注視点、`wxyz`順クォータニオン
- Blender座標系のcamera-to-world / world-to-camera行列
- OpenCV座標系のworld-to-camera行列、回転行列、並進ベクトル

Blenderカメラ座標系は`+X right, +Y up, -Z forward`、OpenCVカメラ座標系は
`+X right, +Y down, +Z forward`として記録される。

ホスト側でPNG列をMP4へ変換する場合は、次のように`ffmpeg`を使用する。

```bash
ffmpeg -framerate 24 \
  -i output/spring_mass_drop/Crosley_Alarm_Clock_Vintage_Metal/camera_00/rgba_%05d.png \
  -c:v libx264 -pix_fmt yuv420p \
  output/spring_mass_drop/Crosley_Alarm_Clock_Vintage_Metal/camera_00.mp4

ffmpeg -framerate 24 \
  -i output/spring_mass_repeated_lift/Crosley_Alarm_Clock_Vintage_Metal/camera_00/rgba_%05d.png \
  -c:v libx264 -pix_fmt yuv420p \
  output/spring_mass_repeated_lift/Crosley_Alarm_Clock_Vintage_Metal/camera_00.mp4
```

`--runtime=nvidia`を利用できない新しいDocker環境では、
`--runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=0`の代わりに
`--gpus '"device=0"'`を使用できる。
