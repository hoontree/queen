# train.py latency measure

> **버전 안내**: `main` 브랜치의 `train.py`는 **단순화된 3버킷 버전**입니다 — StageTimer 없이
> Decode / Update / Render(+E2E)만 측정하고, CUDA sync를 **프레임 경계에서만** 수행합니다
> (Update는 reconstruction 전체를 한 번의 wall-clock span으로 측정: `SpanTimer.resume()/pause()`).
> 구간별(stage) 세분화 타이밍이 필요하면 **`latency-stagewise` 브랜치**(여기 아래 3~5절에서 설명하는
> StageTimer 버전)를 사용하세요. 아래 1~2절의 버킷 정의/`--timed` 전제는 두 버전에 공통입니다.

## 1. 무엇을 측정하나 (4개 latency 버킷)

QUEEN은 프레임이 하나씩 들어오는 온라인 스트리밍 학습이라, "프레임 1장이 들어와서 → 모델이 갱신되고 → 새 뷰로 렌더될 때까지"의 시간을 단계별로 나눠 측정합니다.

| 버킷 | 의미 | 구성 구간 |
| --- | --- | --- |
| **Decode** | RGB 디코드 + GPU 업로드 + 카메라 바인딩 (데이터가 GPU에 준비될 때까지) | `next_batch_with_latency`  • `scene.updateCameraImages` |
| **Update** | 이 프레임의 reconstruction(학습) 시간. "프레임 준비됨 → G_t 완성" | motion_estimation + gaussian_selection + train_render_forward + loss_backward + optimizer_step + densify_prune |
| **Render (heldout)** | 갱신된 모델을 held-out 뷰 1장에 렌더하는 순수 rasterize 시간 (PSNR 등 평가 제외) | heldout_render |
| **E2E** | end-to-end = Decode + Update + Render | 위 3개 합 |

<aside>
🚫

**버킷에서 의도적으로 제외하는 것**: 평가용/스파이럴 렌더(`rendering`), PSNR·SSIM·LPIPS 계산, 파일 저장, 첫 프레임의 static 3DGS 초기화(별도 테이블로 보고). 즉 Update는 "방법론 자체의 갱신 비용"만 담도록 설계됨.

</aside>

## 2. 전제 조건: `--timed` 플래그

측정의 정확도를 위해 `dataset.timed`(기본 `False`)가 켜졌을 때만 다음이 동작

- **`num_workers=0`** — DataLoader를 동기 모드로. 백그라운드 워커가 디코드를 숨기지 못하게 해서, PNG 디코드 비용이 Decode 버킷 안에서 실제로 측정됨.
- **CUDA 동기화** — 각 구간 시작/끝에서 `torch.cuda.synchronize()`. CUDA 커널은 비동기라 sync 없이는 시간이 엉뚱한 구간에 잡힘.

```python
def _sync_cuda(enabled):
    """flag와 CUDA 가용성으로 가드된 torch.cuda.synchronize()."""
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
```

```python
# 측정 모드에서는 num_workers=0으로 디코드를 in-process 동기 실행시켜
# Decode 타이밍 구간 안에서 잡히게 함 (워커가 있으면 숨겨짐).
loader_workers = 0 if dataset.timed else 4
train_loader = iter(torch.utils.data.DataLoader(
    train_image_dataset, batch_size=train_image_dataset.n_cams,
    sampler=train_sampler, num_workers=loader_workers))
```

<aside>
⚠️

`--timed` 없이 돌리면 숫자는 나오지만 **의미가 없습니다**(비동기 커널 + 워커가 비용을 가림). 벤치마킹할 때는 반드시 `--timed`를 켜세요. 출력 JSON에도 `"timed": true/false`가 기록됩니다.

</aside>

## 3. 핵심 빌딩블록 (복붙용)

### 3-1. Decode 타이밍 헬퍼

배치를 하나 꺼내 GPU로 올리는 데 걸린 wall-time을 같이 반환합니다.

```python
def next_batch_with_latency(loader, sync_cuda=False):
    """DataLoader에서 배치 하나를 꺼내 CUDA로 올리고, 걸린 wall-time을 반환.
    num_workers=0(timed)에서는 실제 RGB 디코드(PIL open + ToTensor)가
    in-process 동기 실행되므로 이 구간에 포함됨.
    Returns (images_cuda, paths, elapsed_seconds)."""
    _sync_cuda(sync_cuda)
    t0 = time.perf_counter()
    images, paths = next(loader)
    images = images.cuda()
    _sync_cuda(sync_cuda)
    return images, paths, time.perf_counter() - t0
```

### 3-2. StageTimer (구간 누적 타이머)

`start(name)` / `stop()`으로 코드 구간을 감싸면, 이름별로 시간이 누적됩니다. 프레임마다 `reset()`. `sync=True`면 구간 경계에서 CUDA sync.

```python
class StageTimer:
    def __init__(self, sync):
        self.sync = sync
        self.totals = {}
        self._t0 = None
        self._cur = None
    def reset(self):
        self.totals = {n: 0.0 for n in STAGE_NAMES}
    def start(self, name):
        assert self._cur is None, f"StageTimer: {self._cur} not stopped before starting {name}"
        if self.sync:
            torch.cuda.synchronize()
        self._cur = name
        self._t0 = time.time()
    def stop(self):
        if self._cur is None:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.totals[self._cur] = self.totals.get(self._cur, 0.0) + (time.time() - self._t0)
        self._cur = None
        self._t0 = None

stage_timer = StageTimer(sync=dataset.timed)
```

### 3-3. 구간 이름과 버킷 정의

어떤 구간이 어느 버킷에 들어가는지를 선언으로 분리해 둔 게 핵심입니다. 다른 모델에 적용할 때는 **이 리스트만 본인 파이프라인에 맞게 수정**하면 됩니다.

```python
# 측정할 모든 구간
STAGE_NAMES = ["initialization", "motion_estimation", "gaussian_selection",
               "train_render_forward", "loss_backward", "optimizer_step",
               "densify_prune", "rendering", "heldout_render"]

# Update(=프레임당 reconstruction) 버킷을 구성하는 구간들.
# decode/평가렌더/heldout/저장/static init은 제외.
UPDATE_STAGES = ["motion_estimation", "gaussian_selection",
                 "train_render_forward", "loss_backward", "optimizer_step",
                 "densify_prune"]
```

## 4. 어디에 어떻게 붙였나 (이식 가이드)

학습 루프를 **프레임 루프 → (디코드) → (구간별 학습) → (heldout 렌더)** 순서로 보고, 각 위치에 타이밍 코드를 끼워 넣습니다.

### A. Decode — 데이터가 GPU에 준비되는 지점

프레임 1은 루프 진입 전에, 이후 프레임은 "이전 iteration에서 prefetch"하므로 측정값을 `pending_train_decode_time`에 담아 다음 프레임으로 넘깁니다. 그리고 카메라에 바인딩하는 `updateCameraImages` 비용도 Decode에 합산합니다.

```python
# 프레임 1 (루프 밖)
train_images, train_paths, frame1_decode_time = next_batch_with_latency(
    train_loader, sync_cuda=dataset.timed)
pending_train_decode_time = frame1_decode_time

# 프레임 루프 안: 현재 프레임의 decode_time = 직전에 prefetch된 값
decode_time = pending_train_decode_time

# 다음 프레임 데이터를 미리 로드 (= 다음 프레임의 decode를 측정)
next_train_images, next_train_paths, pending_train_decode_time = next_batch_with_latency(
    train_loader, sync_cuda=dataset.timed)

# 디코드 텐서를 카메라 객체에 바인딩하는 비용도 decode에 포함
_sync_cuda(dataset.timed)
_bind_t0 = time.time()
scene.updateCameraImages(args, train_image_data, test_image_data, frame_idx, resolution_scales=[1.0])
_sync_cuda(dataset.timed)
decode_time += time.time() - _bind_t0
```

### B. Update — 학습 루프 내부의 각 구간을 `start/stop`으로 감싸기

`stage_timer.reset()`을 프레임 시작에서 호출한 뒤, 학습의 각 단계를 감쌉니다. 예: forward / loss+backward / optimizer / densify.

```python
# (프레임 시작에서)
stage_timer.reset()

# --- forward render ---
stage_timer.start("train_render_forward")
render_pkg = render_mask(viewpoint_cam, gaussians, pipe, bg, ...)
image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
stage_timer.stop()

# --- loss + backward ---
stage_timer.start("loss_backward")
Ll1 = l1_loss(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
# ... (기타 loss 항들)
loss.backward()
stage_timer.stop()

# --- densification / pruning ---
stage_timer.start("densify_prune")
# gaussians.densify_and_prune(...) 등
stage_timer.stop()

# --- optimizer step ---
stage_timer.start("optimizer_step")
gaussians.optimizer.step()
gaussians.optimizer.zero_grad(set_to_none=True)
stage_timer.stop()
```

<aside>
💡

프레임>1에서만 도는 **motion_estimation**(gate 초기화용 gradient/frame diff)과 **gaussian_selection**(residual·optimizer 셋업, 마스크 계산)도 동일하게 `start/stop`으로 감쌌습니다. 일반 모델이라면 이 둘은 없을 수 있으니 `UPDATE_STAGES`에서 빼면 됩니다.

</aside>

### C. Render — held-out 뷰 1장에 순수 렌더

학습 루프가 끝난 뒤, 갱신된 모델 `G_t`를 평가에 쓰지 않은 뷰 1장에 rasterize. **PSNR 등 metric 계산은 일부러 제외**(그건 평가 오버헤드지 latency가 아님). `stop()`의 마지막 sync가 E2E 최종 sync 역할도 겸합니다.

```python
if test_image_dataset.n_cams > 0:
    heldout_cam = scene.getTestCameras()[0]
    with torch.no_grad():
        stage_timer.start("heldout_render")
        _ = render_mask(heldout_cam, gaussians, pipe, background,
                        image_shape=heldout_cam.original_image.shape)["render"]
        stage_timer.stop()
render_time = stage_timer.totals.get("heldout_render", 0.0)
```

### D. (1회성) Static init — 첫 프레임 초기화

첫 프레임의 3DGS 초기 학습/모델 생성은 steady-state와 성격이 다르므로 따로 잰 뒤 별도 테이블로 보고합니다.

```python
_sync_cuda(dataset.timed)
static_init_start = time.time()
gaussians = GaussianModel(...)
scene = Scene(...)
gaussians.training_setup(opt)
_sync_cuda(dataset.timed)
static_init_time = time.time() - static_init_start
```

## 5. 버킷 합산 & 출력

프레임 끝에서 구간들을 버킷으로 합산합니다.

```python
# Update = UPDATE_STAGES 합, E2E = decode + update + render
update_time = sum(stage_timer.totals.get(s, 0.0) for s in UPDATE_STAGES)
e2e_time = decode_time + update_time + render_time
latency_metrics = {
    "Decode time": round(decode_time, 6),
    "Update time": round(update_time, 6),
    "Render time (heldout)": round(render_time, 6),
    "E2E time": round(e2e_time, 6),
}
```

학습 종료 후 두 개의 테이블로 정리해 `latency.json`에 저장하고 콘솔에 출력합니다.

- **Initialization (frame 1, 1회성)**: static_init / decode / update / render / e2e
- **Steady-state (frames 2..N)**: 버킷별 mean / std / min / max (`_latency_stats`)

```python
def _latency_stats(values):
    """프레임별 latency 리스트의 count/mean/std/min/max (초)."""
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"count": n, "mean": round(mean, 6), "std": round(var ** 0.5, 6),
            "min": round(min(values), 6), "max": round(max(values), 6)}
```

생성되는 파일: `latency.json`(버킷 요약), `stage_times.json`(구간별 per-frame/total/mean), `training_metrics.json`(프레임별 전체 metric). wandb를 쓰면 `frame/latency/*`, `steady_state/latency/*`로도 로깅됩니다.

## 6. 다른 학습 코드 적용 체크리스트

<aside>
✅

1. **`--timed` 플래그** 추가 → 켜졌을 때 DataLoader `num_workers=0`, CUDA sync 활성화.
2. **`_sync_cuda`, `next_batch_with_latency`, `StageTimer`** 3개를 그대로 복사.
3. **`STAGE_NAMES` / `UPDATE_STAGES`**를 내 파이프라인 구간에 맞게 수정 (forward / loss_backward / optimizer_step은 거의 공통, densify·motion 등 방법 고유 구간만 교체).
4. 학습 루프 안에서 forward/backward/optimizer/기타 구간을 **`stage_timer.start/stop`으로 감싸기**.
5. 데이터 로드 지점을 **`next_batch_with_latency`로 교체**해 decode 측정. prefetch 구조면 값을 다음 step으로 carry-over.
6. (선택) 갱신 후 **held-out 1장 렌더**로 render latency, 첫 step은 **static init** 별도 측정.
7. 프레임/스텝 끝에서 **버킷 합산 → JSON 저장 + `_latency_stats`로 집계**.
</aside>

<aside>
📌

참고 위치(현재 코드): 헬퍼 [train.py:79](http://train.py:79)[-111](train.py#L79-L111) · StageTimer/구간정의 [train.py:272](http://train.py:272)[-307](train.py#L272-L307) · Decode [train.py:391](http://train.py:391)[-508](train.py#L391-L508) · Update 구간 감싸기 [train.py:742](http://train.py:742)[-1022](train.py#L742-L1022) · heldout render [train.py:1031](http://train.py:1031)[-1043](train.py#L1031-L1043) · 버킷 합산/출력 [train.py:1102](http://train.py:1102)[-1296](train.py#L1102-L1296)

</aside>