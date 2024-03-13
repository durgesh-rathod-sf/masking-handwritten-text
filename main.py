import boto3
from PIL import Image, ImageDraw
import os


def detect_handwritten_and_printed_text(image_path):
    # Initialize Textract client
    textract_client = boto3.client(
        "textract",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION"),
    )

    # Read the image
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    # Call Textract API to detect text
    response = textract_client.detect_document_text(Document={"Bytes": image_bytes})

    # Extract bounding box coordinates for handwritten text
    handwriting_text_rects = []
    printed_text_rects = []
    for item in response["Blocks"]:

        if "Text" in item:
            text_type = item.get("TextType", "")
            if text_type == "HANDWRITING" or (
                text_type == "PRINTED"
                and "Geometry" in item
                and "BoundingBox" in item["Geometry"]
            ):
                bbox = item["Geometry"]["BoundingBox"]
                width, height = Image.open(image_path).size
                left = width * bbox["Left"]
                top = height * bbox["Top"]
                right = left + (width * bbox["Width"])
                bottom = top + (height * bbox["Height"])

                if text_type == "HANDWRITING":
                    handwriting_text_rects.append((left, top, right, bottom))
                else:
                    printed_text_rects.append((left, top, right, bottom))

    return handwriting_text_rects, printed_text_rects


def is_overlap(rect1, rect2):
    """
    Check if two rectangles overlap.
    """
    return not (
        rect1[2] <= rect2[0]
        or rect1[0] >= rect2[2]
        or rect1[1] >= rect2[3]
        or rect1[3] <= rect2[1]
    )


def find_intersections_and_cropout_printed_text_rects(
    handwritten_text_rects, printed_text_rects
):
    to_be_masked_rects = []
    for to_paint_rect in handwritten_text_rects:
        entirely_overlapped = False
        for not_to_paint_rect in printed_text_rects:
            if (
                to_paint_rect[0] >= not_to_paint_rect[0]
                and to_paint_rect[1] >= not_to_paint_rect[1]
                and to_paint_rect[2] <= not_to_paint_rect[2]
                and to_paint_rect[3] <= not_to_paint_rect[3]
            ):
                entirely_overlapped = True
                break
        if entirely_overlapped:
            continue

        remaining_area = [to_paint_rect]
        for not_to_paint_rect in printed_text_rects:
            new_remaining_area = []
            for area in remaining_area:
                if is_overlap(area, not_to_paint_rect):
                    # If there's an overlap, split the area and keep the non-overlapping parts
                    left = max(area[0], not_to_paint_rect[0])
                    top = max(area[1], not_to_paint_rect[1])
                    right = min(area[2], not_to_paint_rect[2])
                    bottom = min(area[3], not_to_paint_rect[3])
                    if left < right and top < bottom:
                        # Non-overlapping area on the left
                        if area[0] < left:
                            new_remaining_area.append((area[0], area[1], left, area[3]))
                        # Non-overlapping area on the right
                        if right < area[2]:
                            new_remaining_area.append(
                                (right, area[1], area[2], area[3])
                            )
                        # Non-overlapping area on the top
                        if area[1] < top:
                            new_remaining_area.append((left, area[1], right, top))
                        # Non-overlapping area on the bottom
                        if bottom < area[3]:
                            new_remaining_area.append((left, bottom, right, area[3]))
                    else:
                        # If the cropped area is degenerate, skip it
                        new_remaining_area.append(area)
                else:
                    # If there's no overlap, keep the original area
                    new_remaining_area.append(area)
            remaining_area = new_remaining_area
        # After handling all intersections, add the remaining areas to the painted area
        to_be_masked_rects.extend(remaining_area)
    return to_be_masked_rects


def create_mask_bounding_rects(image_path, to_be_masked_rects):
    image_name = image_path.split("/")[-1]

    # Open the image
    image = Image.open(image_path)

    # Convert to grayscale
    image = image.convert("L")

    # Mask the area with white box
    draw = ImageDraw.Draw(image)
    for box in to_be_masked_rects:
        draw.rectangle(box, fill="white")

    # Save or display the image
    image.save(f"./outputs/{image_name}")


# Example usage
if __name__ == "__main__":
    image_path = "./inputs/oudwt.jpg"

    handwriting_bounding_rects, printed_text_rects = (
        detect_handwritten_and_printed_text(image_path)
    )
    to_be_masked_rects = find_intersections_and_cropout_printed_text_rects(
        handwriting_bounding_rects, printed_text_rects
    )
    create_mask_bounding_rects(image_path, to_be_masked_rects)
