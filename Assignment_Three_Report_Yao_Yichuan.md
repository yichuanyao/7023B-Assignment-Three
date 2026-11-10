# ELEC7023 Assignment 3-2 — Object Detection

| Item | Content |
|------|---------|
| Course | Edge computing device programming for AI projects |
| Student | Yao Yichuan |
| Task | Detect objects in **two images**, compare confidence thresholds **0.3** and **0.7** |
| Device | Jetson (jetson-inference `detectNet`) |
| Code | `assin3_detection.py` |
| Model | `ssd-mobilenet-v2` |

## GitHub Repository

https://github.com/yichuanyao/7023B-Assignment-Three

---

## 1. Method

**Test images (from COCO detection/segmentation benchmark):**

| Image | Content | Source |
|-------|---------|--------|
| `images/image1.jpeg` | Two cats + remotes on a sofa (COCO `000000039769`) | [COCO val2017](http://images.cocodataset.org/val2017/000000039769.jpg) |
| `images/image2.jpeg` | Kitchen scene with person & many objects (COCO `000000397133`) | [COCO val2017](http://images.cocodataset.org/val2017/000000397133.jpg) |

1. Load `ssd-mobilenet-v2` with `jetson_inference.detectNet`.
2. Change camera input to **image input** (`jetson_utils.loadImage`).
3. Run detection on **Image 1** and **Image 2**.
4. Repeat with threshold **0.3** (< 0.5) and **0.7** (> 0.5).
5. For each detection, record:  
   `ClassID, Confidence, Left, Top, Right, Bottom, Width, Height, Area, Center`.
6. Save annotated images under `results/`.

### Run command (on Jetson)

```bash
cd ~/jetson-inference/examples
python3 assin3_detection.py --image1 images/image1.jpeg --image2 images/image2.jpeg --thresholds 0.3 0.7 --outdir results
```

---

## 2. Detected Images

Insert the 4 annotated images from Jetson `results/`:

- `image1_image1_thr0p3.jpg` — Image 1 @ 0.3  
- `image1_image1_thr0p7.jpg` — Image 1 @ 0.7  
- `image2_image2_thr0p3.jpg` — Image 2 @ 0.3  
- `image2_image2_thr0p7.jpg` — Image 2 @ 0.7  

*(Copy these 4 files from Jetson into this report / PDF.)*

---

## 3. Detection Results

### Image 1 @ threshold 0.3 (5 objects)

```text
   -- Detection #1
   -- ClassID: 63
   -- Class: couch
   -- Confidence: 0.57861328125
   -- Left:    0.0
   -- Top:     0.0
   -- Right:   639.0
   -- Bottom:  479.0
   -- Width:   639.0
   -- Height:  479.0
   -- Area:    306081.0
   -- Center:  (319.5, 239.5)

   -- Detection #2
   -- ClassID: 65
   -- Class: bed
   -- Confidence: 0.444580078125
   -- Left:    2.65625
   -- Top:     6.5625
   -- Right:   638.125
   -- Bottom:  470.15625
   -- Width:   635.46875
   -- Height:  463.59375
   -- Area:    294599.34375
   -- Center:  (320.390625, 238.359375)

   -- Detection #3
   -- ClassID: 17
   -- Class: cat
   -- Confidence: 0.99072265625
   -- Left:    11.484375
   -- Top:     59.296875
   -- Right:   318.125
   -- Bottom:  471.328125
   -- Width:   306.640625
   -- Height:  412.03125
   -- Area:    126345.5234375
   -- Center:  (164.8046875, 265.3125)

   -- Detection #4
   -- ClassID: 18
   -- Class: dog
   -- Confidence: 0.55078125
   -- Left:    342.5
   -- Top:     26.71875
   -- Right:   609.375
   -- Bottom:  371.25
   -- Width:   266.875
   -- Height:  344.53125
   -- Area:    91946.78125
   -- Center:  (475.9375, 198.984375)

   -- Detection #5
   -- ClassID: 17
   -- Class: cat
   -- Confidence: 0.82421875
   -- Left:    333.125
   -- Top:     32.9296875
   -- Right:   605.625
   -- Bottom:  367.96875
   -- Width:   272.5
   -- Height:  335.0390625
   -- Area:    91298.140625
   -- Center:  (469.375, 200.44921875)
```

### Image 2 @ threshold 0.3 (4 objects)

```text
   -- Detection #1
   -- ClassID: 67
   -- Class: dining table
   -- Confidence: 0.7568359375
   -- Left:    5.3125
   -- Top:     235.39208984375
   -- Right:   328.125
   -- Bottom:  418.86865234375
   -- Width:   322.8125
   -- Height:  183.4765625
   -- Area:    59228.52734375
   -- Center:  (166.71875, 327.13037109375)

   -- Detection #2
   -- ClassID: 1
   -- Class: person
   -- Confidence: 0.9521484375
   -- Left:    377.8125
   -- Top:     66.51025390625
   -- Right:   505.9375
   -- Bottom:  349.64794921875
   -- Width:   128.125
   -- Height:  283.1376953125
   -- Area:    36277.015625
   -- Center:  (441.875, 208.0791015625)

   -- Detection #3
   -- ClassID: 62
   -- Class: chair
   -- Confidence: 0.338134765625
   -- Left:    534.6875
   -- Top:     244.14892578125
   -- Right:   619.0625
   -- Bottom:  341.30810546875
   -- Width:   84.375
   -- Height:  97.1591796875
   -- Area:    8197.8056640625
   -- Center:  (576.875, 292.728515625)

   -- Detection #4
   -- ClassID: 51
   -- Class: bowl
   -- Confidence: 0.427734375
   -- Left:    230.0
   -- Top:     267.50048828125
   -- Right:   270.0
   -- Bottom:  295.43896484375
   -- Width:   40.0
   -- Height:  27.9384765625
   -- Area:    1117.5390625
   -- Center:  (250.0, 281.4697265625)
```

### Image 1 @ threshold 0.7 (2 objects)

```text
   -- Detection #1
   -- ClassID: 17
   -- Class: cat
   -- Confidence: 0.99072265625
   -- Left:    11.484375
   -- Top:     59.296875
   -- Right:   318.125
   -- Bottom:  471.328125
   -- Width:   306.640625
   -- Height:  412.03125
   -- Area:    126345.5234375
   -- Center:  (164.8046875, 265.3125)

   -- Detection #2
   -- ClassID: 17
   -- Class: cat
   -- Confidence: 0.82421875
   -- Left:    333.125
   -- Top:     32.9296875
   -- Right:   605.625
   -- Bottom:  367.96875
   -- Width:   272.5
   -- Height:  335.0390625
   -- Area:    91298.140625
   -- Center:  (469.375, 200.44921875)
```

### Image 2 @ threshold 0.7 (2 objects)

```text
   -- Detection #1
   -- ClassID: 67
   -- Class: dining table
   -- Confidence: 0.7568359375
   -- Left:    5.3125
   -- Top:     235.39208984375
   -- Right:   328.125
   -- Bottom:  418.86865234375
   -- Width:   322.8125
   -- Height:  183.4765625
   -- Area:    59228.52734375
   -- Center:  (166.71875, 327.13037109375)

   -- Detection #2
   -- ClassID: 1
   -- Class: person
   -- Confidence: 0.9521484375
   -- Left:    377.8125
   -- Top:     66.51025390625
   -- Right:   505.9375
   -- Bottom:  349.64794921875
   -- Width:   128.125
   -- Height:  283.1376953125
   -- Area:    36277.015625
   -- Center:  (441.875, 208.0791015625)
```

### Field meanings (Lecture 4)

| Field | Meaning |
|-------|---------|
| ClassID | Predicted object category ID |
| Confidence | How certain the model is |
| Left / Top | Top-left corner of the box |
| Right / Bottom | Bottom-right corner of the box |
| Width / Height | Box size |
| Area | Width × Height |
| Center | Midpoint of the box |

---

## 4. Threshold Comparison Table

| Image | Confidence threshold | Number of detected objects | Missed or incorrect detections |
|-------|----------------------|----------------------------|--------------------------------|
| Image 1 | 0.3 | 5 | Incorrect: `bed` (sofa misclassified), `dog` (right cat misclassified as dog). Missed: two remotes |
| Image 1 | 0.7 | 2 | Incorrect low-confidence boxes removed. Still missed: two remotes |
| Image 2 | 0.3 | 4 | Weak extras: `chair` (0.34), `bowl` (0.43). Missed many kitchen items (pots/pans/utensils) |
| Image 2 | 0.7 | 2 | `chair` and `bowl` filtered out. Still missed many small kitchen objects |

---

## 5. Report Questions

### Q1. How did the results change when you increased the threshold?

When the threshold increased from **0.3** to **0.7**, the number of detections decreased:

- Image 1: **5 → 2**
- Image 2: **4 → 2**

Only high-confidence boxes remained (both cats on Image 1; person + dining table on Image 2). Low-confidence / uncertain predictions were removed, so the output looked cleaner, but some true or borderline objects disappeared.

### Q2. Were any objects detected at the lower threshold but not at the higher threshold? Why?

Yes.

- Image 1 lost: `couch` (0.58), `bed` (0.44), `dog` (0.55)  
- Image 2 lost: `chair` (0.34), `bowl` (0.43)

These scores are **between 0.3 and 0.7**, so they pass the low threshold but fail the high threshold. The network still proposes those boxes; confidence filtering discards them when the cutoff is stricter.

### Q3. Which threshold would you choose for your application scenario? Explain your choice.

**Chosen scenario:** home / kitchen object detection on an edge device (person, cat, bowl, furniture — COCO-style classes).

**Chosen threshold: 0.7**

Reason: at 0.3, Image 1 produced clear mistakes (`bed`, `dog`), and Image 2 added weak boxes. At 0.7, the remaining detections (`cat`, `cat`, `person`, `dining table`) are more reliable. For an edge demo where false alarms are more confusing than missing a few small objects, **0.7** is the better choice.

---

## 6. Submission Checklist

- [ ] `assin3_detection.py` uploaded to GitHub
- [ ] GitHub link written in this report
- [ ] Detected / annotated images included
- [ ] Detection field outputs included
- [ ] Comparison table filled
- [ ] Three questions answered
- [ ] Export this file to **PDF or DOC** and upload to **Moodle**
