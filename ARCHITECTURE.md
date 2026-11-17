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
    KP["keypoints<br/>(B, 2, T, 17, 3)"] --> HM["keypoints_to_heatmaps<br/>Gaussian, sigma=0.6<br/>(B, 34, T, 56, 56)"]

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

Inverted residual block. The shift sits on the residual branch only, so the shortcut keeps the original frames. Squeeze-and-excitation is used on every other block (`(i + 1) % 2 == 0`).

```mermaid
flowchart LR
    X["x"] --> TS["TemporalShift<br/>1/8 channels forward,<br/>1/8 backward in time"]
    TS --> EX["1x1 expand<br/>Conv + BN + ReLU"]
    EX --> DW["1x3x3 depthwise<br/>Conv + BN<br/>stride 1 or 2"]
    DW --> SE["SqueezeExcite3d<br/>(every other block)"]
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
