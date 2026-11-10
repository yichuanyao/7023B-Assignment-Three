

import argparse
import os
import sys

try:
    import jetson_inference
    import jetson_utils
except ImportError:
    # older jetson-inference packages
    import jetson.inference as jetson_inference
    import jetson.utils as jetson_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assignment 3-2 object detection with threshold comparison"
    )
    parser.add_argument(
        "--image1",
        default="images/image1.jpg",
        help="Path to the first input image",
    )
    parser.add_argument(
        "--image2",
        default="images/image2.jpg",
        help="Path to the second input image",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.7],
        help="Confidence thresholds to compare (one < 0.5 and one > 0.5)",
    )
    parser.add_argument(
        "--model",
        default="ssd-mobilenet-v2",
        help="detectNet model name (default: ssd-mobilenet-v2)",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Directory for annotated images and summary text",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Optional: only keep detections with this ClassID",
    )
    return parser.parse_args()


def print_detection(detection, index, net):
    """Print fields required by Assignment 3-2 / Lecture 4."""
    class_name = net.GetClassDesc(detection.ClassID)
    print("   -- Detection #{}".format(index))
    print("   -- ClassID: {}".format(detection.ClassID))
    print("   -- Class: {}".format(class_name))
    print("   -- Confidence: {}".format(detection.Confidence))
    print("   -- Left:    {}".format(detection.Left))
    print("   -- Top:     {}".format(detection.Top))
    print("   -- Right:   {}".format(detection.Right))
    print("   -- Bottom:  {}".format(detection.Bottom))
    print("   -- Width:   {}".format(detection.Width))
    print("   -- Height:  {}".format(detection.Height))
    print("   -- Area:    {}".format(detection.Area))
    print("   -- Center:  ({}, {})".format(detection.Center[0], detection.Center[1]))
    print("")


def format_detection_block(detection, index, net):
    class_name = net.GetClassDesc(detection.ClassID)
    return "\n".join(
        [
            "   -- Detection #{}".format(index),
            "   -- ClassID: {}".format(detection.ClassID),
            "   -- Class: {}".format(class_name),
            "   -- Confidence: {}".format(detection.Confidence),
            "   -- Left:    {}".format(detection.Left),
            "   -- Top:     {}".format(detection.Top),
            "   -- Right:   {}".format(detection.Right),
            "   -- Bottom:  {}".format(detection.Bottom),
            "   -- Width:   {}".format(detection.Width),
            "   -- Height:  {}".format(detection.Height),
            "   -- Area:    {}".format(detection.Area),
            "   -- Center:  ({}, {})".format(detection.Center[0], detection.Center[1]),
            "",
        ]
    )


def filter_detections(detections, class_id):
    if class_id is None:
        return list(detections)
    return [d for d in detections if int(d.ClassID) == int(class_id)]


def run_detection(net, image_path, output_path, class_id=None):
    """Load one image, run Detect(), save annotated result, return detections."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Image not found: {}".format(image_path))

    img = jetson_utils.loadImage(image_path)
    detections = net.Detect(img, overlay="box,labels,conf")
    detections = filter_detections(detections, class_id)

    jetson_utils.saveImage(output_path, img)
    return detections


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    image_items = [
        ("Image 1", args.image1),
        ("Image 2", args.image2),
    ]

    for name, path in image_items:
        if not os.path.isfile(path):
            print("ERROR: {} not found: {}".format(name, path))
            print("Put two test images under ./images/ or pass --image1/--image2.")
            sys.exit(1)

    # Validate assignment requirement: one threshold < 0.5 and one > 0.5
    has_low = any(t < 0.5 for t in args.thresholds)
    has_high = any(t > 0.5 for t in args.thresholds)
    if not (has_low and has_high):
        print(
            "WARNING: assignment expects one threshold < 0.5 and one > 0.5 "
            "(e.g. 0.3 and 0.7). Current: {}".format(args.thresholds)
        )

    summary_lines = []
    summary_lines.append("Assignment 3-2 Detection Summary")
    summary_lines.append("=" * 60)
    summary_lines.append(
        "{:<10} {:<12} {:<22} {}".format(
            "Image", "Threshold", "Num detected objects", "Notes"
        )
    )
    summary_lines.append("-" * 60)

    detailed_log = []

    for threshold in args.thresholds:
        print("\n" + "=" * 60)
        print("Loading model '{}' with threshold={}".format(args.model, threshold))
        print("=" * 60)

        # Recreate the network for each threshold (matches lecture style)
        net = jetson_inference.detectNet(args.model, threshold=threshold)

        for image_name, image_path in image_items:
            base = os.path.splitext(os.path.basename(image_path))[0]
            thr_tag = str(threshold).replace(".", "p")
            out_name = "{}_{}_thr{}.jpg".format(base, image_name.replace(" ", "").lower(), thr_tag)
            out_path = os.path.join(args.outdir, out_name)

            print("\n--- {} | threshold={} | input={} ---".format(
                image_name, threshold, image_path
            ))

            detections = run_detection(net, image_path, out_path, args.class_id)
            num = len(detections)
            print("Number of detected objects: {}".format(num))
            print("Annotated image saved to: {}".format(out_path))

            block = []
            block.append("\n{} | threshold={} | file={}".format(
                image_name, threshold, image_path
            ))
            block.append("Number of detected objects: {}".format(num))
            block.append("Annotated image: {}".format(out_path))

            if num == 0:
                print("(no detections above this threshold)")
                block.append("(no detections above this threshold)")
            else:
                for i, det in enumerate(detections, start=1):
                    print_detection(det, i, net)
                    block.append(format_detection_block(det, i, net))

            detailed_log.extend(block)

            # Missed/incorrect column is filled by visual inspection in the report.
            summary_lines.append(
                "{:<10} {:<12} {:<22} {}".format(
                    image_name,
                    threshold,
                    num,
                    "Fill in after visual check",
                )
            )

    summary_text = "\n".join(summary_lines)
    detail_text = "\n".join(detailed_log)

    summary_path = os.path.join(args.outdir, "summary_table.txt")
    detail_path = os.path.join(args.outdir, "detection_details.txt")

    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    with open(detail_path, "w") as f:
        f.write(detail_text + "\n")

    print("\n" + summary_text)
    print("\nSaved summary table -> {}".format(summary_path))
    print("Saved detection details -> {}".format(detail_path))
    print(
        "\nNext: open annotated images in '{}', fill 'Missed or incorrect "
        "detections' in the report table, then answer the three questions.".format(
            args.outdir
        )
    )


if __name__ == "__main__":
    main()
