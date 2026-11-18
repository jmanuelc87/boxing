# EPAM-Net Architecture

Pose-driven, attention-guided two-stream network for punch recognition, implemented in `src/boxing/model.py`.

**Inputs**

| Name | Shape | Notes |
|---|---|---|
| `images` | `(B, T, 3, H, W)` | normalised RGB frames, e.g. T=16, 224x224 |
| `keypoints` | `(B, 2, T, 17, 3)` | `(x, y, score)`, x/y in [0, 1] of the crop; 2 boxers, 17 joints |

**Outputs**: `EPAMOutput(rgb_logits, pose_logits)`, each `(B, num_classes)`. At inference the two softmax scores are fused with `rgb_weight`.

## Overview

```mermaid
flowchart TD
    IMG["images<br/>(B, T, 3, 224, 224)"] --> PERM["permute to (B, 3, T, H, W)"]
    KP["keypoints<br/>(B, 2, T, 17, 3)"] --> HM["keypoints_to_heatmaps<br/>Gaussian, sigma 0.6<br/>(per joint optional)<br/>(B, 34, T, 56, 56)"]

    subgraph RGB["RGB stream"]
        PERM --> RB["X-ShiftNet-M<br/>4 stages, stem stride 2<br/>blocks 3-5-11-7"]
        RB --> RF["rgb_feats<br/>(B, 432, T, 7, 7)"]
    end

    subgraph POSE["Pose stream"]
        HM --> PB["X-ShiftNet<br/>3 stages, stem stride 1<br/>blocks 5-11-7"]
        PB --> PF["pose_feats<br/>(B, 216, T, 7, 7)"]
    end

    PF --> ALIGN["_align_pose<br/>stride frames / pool space<br/>to match RGB"]
    ALIGN --> ATT["PoseGuidedAttention<br/>(B, 1, T, 7, 7)"]

    RF --> GUIDE["guided = rgb + rgb * attention"]
    ATT --> GUIDE

    GUIDE --> RH["rgb_head<br/>ClassifierHead"]
    PF --> PH["pose_head<br/>ClassifierHead"]

    RH --> RL["rgb_logits<br/>(B, num_classes)"]
    PH --> PL["pose_logits<br/>(B, num_classes)"]

    RL --> FUSE["inference: late fusion<br/>w * softmax(rgb) + (1-w) * softmax(pose)"]
    PL --> FUSE
    FUSE --> PROB["class probabilities"]
```

## Pose heatmaps

`keypoints_to_heatmaps` renders each joint as a Gaussian at its `(x, y)` position on a 56x56 grid, scaled by its confidence. Joints with a score at or below `min_score` (0.1) give an empty map. The channels are ordered boxer 0's 17 joints, then boxer 1's.

`sigma` (the Gaussian radius, in heatmap pixels) is either one value for every joint (default 0.6) or a list of 17, one per COCO joint, so each keypoint can have its own radius. A list must have 17 positive values. Set it with `EPAMNetConfig(sigma=...)` or `--sigma` on the training script.

## X-ShiftNet backbone

Used for both streams with different depth, width and stem stride. All convolutions are 2D (`1x3x3` kernels inside `Conv3d`), and temporal mixing comes from the parameter-free Temporal Shift Module. The frame count is never reduced.

```mermaid
flowchart TD
    IN["input (B, C_in, T, H, W)"] --> STEM["Stem<br/>Conv 1x3x3 + BN + ReLU<br/>stride 1 (pose) or 2 (RGB)"]
    STEM --> S1["Stage 1..N<br/>first block stride 2, rest stride 1"]
    S1 --> C5["conv5<br/>1x1 Conv + BN + ReLU<br/>widen by 2.25x"]
    C5 --> OUT["output (B, C_out, T, 7, 7)"]
```

| Stream | Input channels | Stem stride | Stage blocks | Stage channels | Output channels |
|---|---|---|---|---|---|
| RGB | 3 | 2 | 3, 5, 11, 7 | 24, 48, 96, 192 | 432 |
| Pose | 34 | 1 | 5, 11, 7 | 24, 48, 96 | 216 |

### ShiftBlock

Inverted residual block. The shift sits on the residual branch only, so the shortcut keeps the original frames. Squeeze-and-excitation is used on every other block, starting with the first (`i % 2 == 0`, i.e. blocks 0, 2, 4, ...), and its hidden width is the block's inner width divided by 16, rounded to a multiple of 8 with a minimum of 8 (54 gives 8, 216 gives 16). The shift, SE placement and SE width follow the official EPAM-Net code, so its pretrained weights fit.

```mermaid
flowchart LR
    X["x"] --> TS["TemporalShift<br/>1/div channels take the next frame,<br/>1/div the previous frame<br/>(div 8 RGB, 4 pose)"]
    TS --> EX["1x1 expand<br/>Conv + BN + ReLU"]
    EX --> DW["1x3x3 depthwise<br/>Conv + BN<br/>stride 1 or 2"]
    DW --> SE["SqueezeExcite3d<br/>(blocks 0, 2, 4, ...)"]
    SE --> SI["SiLU"]
    SI --> PR["1x1 project<br/>Conv + BN"]
    PR --> ADD(("+"))
    X --> SC["shortcut<br/>Identity or 1x1 Conv + BN"]
    SC --> ADD
    ADD --> RE["ReLU"]
```

## Pose-guided attention

Computed from the pose features and used to re-weight the RGB features. The output lies in [0, 1].

```mermaid
flowchart TD
    PF["aligned pose_feats<br/>(B, C, T, H, W)"] --> MAX["channel max"]
    PF --> MEAN["channel mean"]
    MAX --> CAT["concat<br/>(B, 2, T, H, W)"]
    MEAN --> CAT
    CAT --> SC["Conv 1x7x7 + BN"]
    SC --> SS["sigmoid<br/>spatial map (B, 1, T, H, W)"]
    SS --> AVG["average over H, W<br/>(B, 1, T)"]
    AVG --> TC["Conv1d, kernel 7"]
    TC --> TSIG["sigmoid<br/>temporal weight (B, 1, T, 1, 1)"]
    SS --> MUL(("x"))
    TSIG --> MUL
    MUL --> ATT["attention (B, 1, T, H, W)"]
```

## Classifier head

Each stream has its own head: `Dropout(0.5)`, global average pool over (T, H, W), and one `Linear` layer to `num_classes`.

```mermaid
flowchart LR
    F["features<br/>(B, C, T, H, W)"] --> GAP["adaptive avg pool 3D"] --> DO["Dropout"] --> FC["Linear"] --> L["logits<br/>(B, num_classes)"]
```

## Temporal shift

`TemporalShift` moves the first `C // div` channels to the next frame (`out[t] = x[t + 1]`) and the next `C // div` channels to the previous frame (`out[t] = x[t - 1]`). Frames shifted past either end are zero-filled, and the rest of the channels are untouched. It has no parameters. The divisor is `rgb_shift_div` (8) for the RGB stream and `pose_shift_div` (4) for the pose stream, the values the official code uses.

## Pretrained backbones

`boxing.pretrained.load_official_backbone` loads the official EPAM-Net backbones (mmaction checkpoints, e.g. NTU RGB+D 120) into `rgb_backbone` or `pose_backbone`. It renames the keys (`layerN.i.conv1` to `stages.N-1.i.expand`, and so on) and loads strictly, so any shape or key mismatch raises. The heads and the attention block are not loaded. The official pose stem has 17 input channels (one person), so it is copied for both boxers and divided by 2. On the training script, use `--init-rgb` and `--init-pose`.

Checkpoints saved before the layout above (SE on odd blocks, a single shift fold of 8, the opposite shift direction) do not load into the current model.
