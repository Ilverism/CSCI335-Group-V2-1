import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, check_imshow, non_max_suppression, apply_classifier,
                           scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    analyze = opt.img_analyze  #<-- Whether to do image analysis or not
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    #Country codes
    country_codes = ["ca", "cn", "de", "in", "jp", "pk", "ru", "kr", "gb", "us"]
    #Background categories
    categories = ["solidorgradient", "realistic", "noise"]
    #Stats dictionary for countries
    country_stats = {
        cc: {"images_processed": 0, "images_detected": 0, "detections": 0, "sum_conf": 0.0}
        for cc in country_codes
    }
    #Stats dictionary for categories
    category_stats = {
        cat: {"images_processed": 0, "images_detected": 0, "detections": 0, "sum_conf": 0.0}
        for cat in categories
    }
    #Map class names to codes
    class_name_to_code = {
        "China": "cn",
        "Japan": "jp",
        "Germany": "de",
        "India": "in",
        "Pakistan": "pk",
        "Russia": "ru",
        "South-Korea": "kr",
        "United-Kingdom": "gb",
        "United-States": "us",
        "Canada": "ca"
    }

    def get_country_from_filename(filename):
        parts = filename.lower().split('_')
        if (len(parts) > 1):
            code = parts[1]
            if code in country_codes:
                return code
        return None

    def get_category_from_filename(filename):
        parts = filename.lower().split('_')
        if (len(parts) > 2):
            cat = parts[2]
            if cat in categories:
                return cat
        return None

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        p = Path(path)
        country_code = get_country_from_filename(p.name)
        category = get_category_from_filename(p.name)

        # If analyzing and we have a country and category recognized:
        if analyze and country_code and category:
            country_stats[country_code]["images_processed"] += 1
            category_stats[category]["images_processed"] += 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        s, im0, frame = '', im0s, getattr(dataset, 'frame', 0)
        save_path = str(save_dir / p.name)
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

        found_flag_for_country = False

        for i, det in enumerate(pred):

            if len(det):

                #Rescale boxes
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    class_name = names[int(c)]
                    s += f"{n} {class_name}{'s' * (n > 1)}, "

                #Process each detection...
                for *xyxy, conf, cls_id in reversed(det):
                    cls = int(cls_id)
                    conf_val = float(conf)
                    class_name = names[cls]
                    label = f'{class_name} {conf_val:.2f}'

                    if (save_img or view_img):
                        plot_one_box(xyxy, im0, label=label, color=colors[cls], line_thickness=1)
                        print(label)

                    #If analyzing and we know this image's country and category:
                    if (analyze and country_code and category):

                        detected_code = class_name_to_code.get(class_name, None)

                        #Detected code and Country code match, increment stats
                        if detected_code == country_code:
                            country_stats[country_code]["detections"] += 1
                            country_stats[country_code]["sum_conf"] += conf_val

                            category_stats[category]["detections"] += 1
                            category_stats[category]["sum_conf"] += conf_val
                            found_flag_for_country = True

                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            else:
                # No detections
                print(f'{s}<No Flags Detected> Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        #Found at least one correct detection...
        if (analyze and country_code and category and found_flag_for_country):
            country_stats[country_code]["images_detected"] += 1
            category_stats[category]["images_detected"] += 1

        #Show images
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)

        #Save results (image)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
        print()

    #Images finished processing; if analyzing, print aggregate results by country...
    if analyze:

        print("Aggregate Results by Country:")
        for cc in country_codes:
            images_processed = country_stats[cc]["images_processed"]
            images_detected = country_stats[cc]["images_detected"]
            detections_count = country_stats[cc]["detections"]
            sum_conf = country_stats[cc]["sum_conf"]

            if images_processed > 0:
                detection_rate = (images_detected / images_processed) * 100.0
                avg_conf = (sum_conf / detections_count) * 100.0 if detections_count > 0 else 0.0
                print(f"  Country: {cc.upper()} | Images: {images_processed} | Images Detected: {images_detected} | Detection Rate: {detection_rate:.2f}% | Detections: {detections_count} | Avg. Confidence: {avg_conf:.2f}%")
            else:
                print(f"  Country: {cc.upper()} | No images processed.")

        #Print aggregate results by category
        print("\nAggregate Results by Category:")
        for cat in categories:
            images_processed = category_stats[cat]["images_processed"]
            images_detected = category_stats[cat]["images_detected"]
            detections_count = category_stats[cat]["detections"]
            sum_conf = category_stats[cat]["sum_conf"]

            if images_processed > 0:
                detection_rate = (images_detected / images_processed) * 100.0
                avg_conf = (sum_conf / detections_count) * 100.0 if detections_count > 0 else 0.0
                print(f"  Category: {cat} | Images: {images_processed} | Images Detected: {images_detected} | Detection Rate: {detection_rate:.2f}% | Detections: {detections_count} | Avg. Confidence: {avg_conf:.2f}%")
            else:
                print(f"  Category: {cat} | No images processed.")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--img-analyze', type=bool, default=False, help='report aggregate results per country')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
