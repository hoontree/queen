# Streaming Baseline Timing Metrics

이 문서는 `train_stream.py`가 기록하는 timing metric의 의미만 정리한다.

모든 시간 단위는 **초(seconds, s)** 이다. 예를 들어 `0.0123`은 `12.3 ms`이다.

## 핵심 비교 Metric

### `OpenCV read time`

비디오 frame을 OpenCV로 읽는 시간이다.

포함 범위:
- `cv2.VideoCapture.read()` 호출 시간
- container demux
- codec decode
- OpenCV 내부 buffering
- decoded BGR frame materialization

주의:
- 순수 codec decode 시간만은 아니다.
- 하지만 streaming baseline에서 “비디오에서 frame을 얻는 실제 비용”으로 가장 직접적인 값이다.

주요 사용처:
- 비디오 decoding baseline 비교
- GOP-compressed frame representation이 decoding을 얼마나 줄이는지 비교

### `Train data loading time`

train split 기준으로, 비디오 frame을 학습 입력 tensor로 준비하는 전체 시간이다.

포함 범위:
- `OpenCV read time`
- `OpenCV convert time`
- `Tensor convert time`
- `Stack time`
- `CUDA transfer time`

주의:
- test camera loading은 포함하지 않는다.
- streaming deployment 또는 GOP update 방식과 비교할 때 가장 실용적인 input latency baseline이다.

### `e2e_latency`

steady-state 기준 end-to-end latency 추정값이다.

계산:

```text
e2e_latency = mean(Frame time) + mean(Train data loading time)
```

포함 범위:
- train frame loading/input 준비
- Gaussian update/training frame 처리 시간

주의:
- first-frame startup cost는 제외한 steady-state 값으로 보는 것이 좋다.
- rendering/output validation 비용은 별도 metric으로 분리되어 있다.

## Frame Loading Breakdown

### `OpenCV frame time`

OpenCV가 frame을 읽고 RGB로 변환하는 데 걸린 시간이다.

계산:

```text
OpenCV frame time = OpenCV read time + OpenCV convert time
```

### `OpenCV convert time`

OpenCV BGR frame을 RGB로 변환하는 시간이다.

포함 범위:
- `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)`

### `Tensor convert time`

NumPy RGB frame을 PyTorch tensor로 변환하는 시간이다.

포함 범위:
- contiguous NumPy array 준비
- `torch.from_numpy`
- channel order 변경 `(H, W, C) -> (C, H, W)`
- float 변환
- `[0, 255] -> [0, 1]` scaling

### `Stack time`

camera별 tensor를 multi-view batch tensor로 묶는 시간이다.

결과 tensor shape:

```text
(n_cams, C, H, W)
```

### `CUDA transfer time`

CPU tensor를 CUDA device로 옮기는 시간이다.

주의:
- `dataset.timed=True`일 때는 CUDA synchronize를 사용하므로 더 정확하다.
- `dataset.timed=False`일 때는 CUDA enqueue 시간에 가까워 실제 완료 시간보다 작게 보일 수 있다.

## Aggregate Loading Metrics

### `Data loading time`

train과 test split을 모두 포함한 전체 loading time이다.

계산:

```text
Data loading time =
  train load total
  + train CUDA transfer
  + test load total
  + test CUDA transfer
```

주의:
- streaming deployment baseline으로는 test loading이 섞일 수 있으므로 주의해야 한다.
- 비교 실험에서는 보통 `Train data loading time`을 우선 사용한다.

### `train_data_loading_time`

JSON summary에서 사용하는 snake_case 이름이다.

의미는 `Train data loading time`과 같다.

### `data_loading_time`

JSON summary에서 사용하는 snake_case 이름이다.

의미는 `Data loading time`과 같다.

## Training / Update Timing

### `Frame time`

한 frame에 대한 Gaussian optimization/update loop의 wall time이다.

포함 범위:
- motion estimation
- Gaussian selection
- render forward
- loss backward
- optimizer step
- densify/prune
- logging/reporting 일부

주의:
- input loading latency는 별도 `Train data loading time`으로 분리되어 있다.
- streaming 비교에서는 `Frame time` 단독보다 `Frame time + Train data loading time`을 보는 것이 낫다.

### `Rendering time`

validation/output rendering에 사용된 시간이다.

주의:
- training loss 계산을 위한 render forward와는 별도이다.
- FPS 계산에서 output rendering 성능을 보기 위한 값이다.

### `Rendering FPS`

output rendering 기준 FPS이다.

계산:

```text
Rendering FPS = Rendering frames / Rendering time
```

## Stage Timing Metrics

### `initialization`

첫 frame에서 초기화에 사용된 시간이다.

주의:
- first-frame startup 비교에는 `startup_profile`의 값을 우선 사용한다.

### `data_loading`

frame metric에 기록된 data loading stage 시간이다.

의미:
- `Data loading time`과 동일한 값
- train + test loading을 포함한다.

### `motion_estimation`

frame difference, viewspace difference 등 update mask 계산에 필요한 motion-related processing 시간이다.

### `gaussian_selection`

업데이트할 Gaussian mask/gate/update 대상 선택에 걸린 시간이다.

### `train_render_forward`

training loss 계산을 위한 render forward 시간이다.

### `loss_backward`

loss 계산과 backward pass 시간이다.

### `optimizer_step`

optimizer update step 시간이다.

### `densify_prune`

Gaussian densification/pruning 관련 시간이다.

### `rendering`

validation/output rendering 시간이다.

## First-frame Metrics

첫 frame은 steady-state와 분리해서 해석해야 한다. 첫 frame은 video loading뿐 아니라 scene/Gaussian initialization의 기준이 되기 때문이다.

### `first_frame_load_time`

첫 train frame을 학습 입력으로 준비하는 시간이다.

의미:
- first frame의 `train_data_loading_time`

### `first_frame_load_wall_time`

첫 train/test frame loading 구간의 wall time이다.

주의:
- train/test loading, Python overhead 등이 포함될 수 있다.

### `first_frame/opencv_read_time`

WandB summary에 기록되는 first-frame OpenCV read time이다.

### `first_frame/train_data_loading_time`

WandB summary에 기록되는 first-frame train input loading time이다.

## Startup Metrics

### `scene_init_time`

`Scene` 생성과 초기 Gaussian scene setup에 걸린 시간이다.

포함 가능 범위:
- camera metadata loading
- train/test/video camera construction
- initial point cloud 기반 Gaussian initialization

### `depth_init_time`

depth supervision 또는 depth initialization이 켜져 있을 때의 depth 관련 초기화 시간이다.

포함 가능 범위:
- MiDaS/depth model 준비
- first-frame depth prediction
- depth 기반 point initialization

값이 `0.0`이면 depth 관련 초기화가 비활성화된 것이다.

### `total_startup_time`

첫 frame loading부터 startup profiling 종료까지의 전체 startup time이다.

포함 범위:
- first-frame loading
- scene initialization
- optional depth initialization
- 관련 Python overhead

## Steady-state Metrics

### `steady_state_mean`

첫 frame을 제외한 frame들의 평균이다.

조건:

```text
Frame index > 1
```

이 값은 GOP-compressed Gaussian update 방식과 비교할 때 가장 중요한 summary이다.

### `steady_state_mean.opencv_read_time`

steady-state 평균 OpenCV video read/decode baseline이다.

### `steady_state_mean.train_data_loading_time`

steady-state 평균 train input preparation latency이다.

### `steady_state_mean.frame_time`

steady-state 평균 Gaussian update/training frame 처리 시간이다.

### `steady_state_mean.e2e_latency`

steady-state 평균 end-to-end latency이다.

계산:

```text
steady_state_mean.e2e_latency =
  steady_state_mean.frame_time
  + steady_state_mean.train_data_loading_time
```

## All-frame Metrics

### `all_frames_mean`

첫 frame을 포함한 전체 frame 평균이다.

주의:
- first-frame startup behavior가 섞이므로 streaming steady-state 비교에는 적합하지 않을 수 있다.
- 전체 실행 평균을 보고 싶을 때만 사용한다.

## Output Files

### `first_frame_load_profile.json`

첫 frame loading metric만 저장한다.

주요 값:
- `train_data_loading_time`
- `opencv_read_time`
- `opencv_convert_time`
- `tensor_convert_time`
- `stack_time`
- `cuda_transfer_time`
- `wall_time`

### `startup_profile.json`

startup 관련 metric을 저장한다.

주요 값:
- `first_frame_load_time`
- `first_frame_load_wall_time`
- `scene_init_time`
- `depth_init_time`
- `total_startup_time`

### `latency_metrics.json`

per-frame latency metric을 저장한다.

주요 값:
- `frame_time`
- `train_data_loading_time`
- `opencv_read_time`
- `opencv_convert_time`
- `tensor_convert_time`
- `stack_time`
- `cuda_transfer_time`
- `rendering_time`

### `streaming_baseline_summary.json`

비교 실험에 가장 직접적으로 사용할 summary 파일이다.

우선 확인할 값:
- `steady_state_mean.opencv_read_time`
- `steady_state_mean.train_data_loading_time`
- `steady_state_mean.e2e_latency`
- `first_frame.startup_profile.total_startup_time`

## Recommended Comparison Values

GOP-compressed Gaussian update 방식과 비교할 때 권장 metric은 다음 순서다.

1. `steady_state_mean.opencv_read_time`
2. `steady_state_mean.train_data_loading_time`
3. `steady_state_mean.e2e_latency`
4. `first_frame.startup_profile.total_startup_time`

해석:

- decode 비용만 비교: `steady_state_mean.opencv_read_time`
- input 준비 전체 비용 비교: `steady_state_mean.train_data_loading_time`
- streaming update 전체 latency 비교: `steady_state_mean.e2e_latency`
- startup 비용 비교: `first_frame.startup_profile.total_startup_time`
