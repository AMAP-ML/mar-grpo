import requests
import time
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
import torch
import numpy as np
import pickle
from collections import defaultdict
from PIL import Image, ImageOps
import json
import os
from pathlib import Path
from mmdet.apis import inference_detector, init_detector
from concurrent.futures import ThreadPoolExecutor
import open_clip
import mmdet
# from mmcv.transforms import Compose
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmengine.dataset import default_collate
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it

UTILS_DIR = Path(__file__).resolve().parent
REWARD_SERVER_DIR = UTILS_DIR / 'reward-server' / 'reward_server'
DEFAULT_CONFIG_PATH = 'configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
DEFAULT_OBJECT_DETECTOR = 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco'
MY_CONFIG_PATH = os.environ.get('GENEVAL_MMDET_CONFIG', DEFAULT_CONFIG_PATH)
MY_CKPT_PATH = os.environ.get('GENEVAL_MMDET_CKPT_DIR')
MY_CLIP_PATH = os.environ.get('GENEVAL_CLIP_CKPT')
OBJ_NAMES_PATH = os.environ.get('GENEVAL_OBJECT_NAMES', str(REWARD_SERVER_DIR / 'object_names.txt'))


def batch_inference_detector(model, image_pils, device='cuda:0'):
    cfg = model.cfg
    pipeline = cfg.data.test.pipeline

    # 替换掉从文件读取图像的部分
    for i, transform in enumerate(pipeline):
        if transform['type'] == 'LoadImageFromFile':
            pipeline[i] = dict(type='LoadImageFromWebcam')
    pipeline = Compose(pipeline)

    data_list = []
    for image_pil in image_pils:
        data = dict(img=np.array(image_pil))
        data = pipeline(data)
        data_list.append(data)

    data = collate(data_list, samples_per_gpu=len(image_pils))
    data = scatter(data, [device])[0]

    # 关闭梯度计算
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform):
        self._image = image.convert("RGB")
        self.transform=transform
        bgcolor = "#999"
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        image = image.crop(box[:4])
        return (self.transform(image), 0)


class Geneval_score:
    def __init__(self,args):
        # self.clip_path = args.geneval_clip_path
        # self.mmdet_path = args.geneval_mmdet_path
        self.THRESHOLD = 0.3
        self.COUNTING_THRESHOLD = 0.9
        self.MAX_OBJECTS = 16
        self.NMS_THRESHOLD = 1.0
        self.POSITION_THRESHOLD = 0.1
        self.only_strict=True
        self.COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
        self.COLOR_CLASSIFIERS = {}

    @property
    def __name__(self):
        return 'Geneval'
    
    def load_to_device(self, load_device):
        print(load_device)
        config_path = Path(MY_CONFIG_PATH)
        if not config_path.is_absolute():
            config_path = Path(os.path.dirname(mmdet.__file__)) / config_path

        object_detector = os.environ.get('GENEVAL_OBJECT_DETECTOR', DEFAULT_OBJECT_DETECTOR)
        if MY_CKPT_PATH is None:
            raise ValueError('Please set GENEVAL_MMDET_CKPT_DIR to the directory containing the MMDetection checkpoint.')
        ckpt_path = Path(MY_CKPT_PATH) / f"{object_detector}.pth"
        object_detector = init_detector(str(config_path), str(ckpt_path), device=load_device)

        clip_arch = "ViT-L-14"
        if MY_CLIP_PATH is None:
            raise ValueError('Please set GENEVAL_CLIP_CKPT to the OpenCLIP checkpoint used for GenEval reward.')
        clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained=MY_CLIP_PATH, device=load_device)
        tokenizer = open_clip.get_tokenizer(clip_arch)

        with open(OBJ_NAMES_PATH) as cls_file:
            classnames = [line.strip() for line in cls_file]
            
        self.object_detector=object_detector
        self.classnames=classnames
        self.clip_model=clip_model
        self.transform=transform
        self.tokenizer=tokenizer
            
    def relative_position(self, obj_a, obj_b):
        """Give position of A relative to B, factoring in object dimensions"""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        #
        revised_offset = np.maximum(np.abs(offset) - self.POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        #
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")
        return relations

    def color_classification(self, image, bboxes, classname, device, transform):
        if classname not in self.COLOR_CLASSIFIERS:
            self.COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                self.clip_model, self.tokenizer, self.COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object"
                ],
                device
            )
        clf = self.COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes, transform),
            batch_size=32, num_workers=4
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip_model, clf, dataloader, device)
            return [self.COLORS[index.item()] for index in pred.argmax(1)]

    def evaluate(self, image, objects, metadata, device):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = objects.get(classname, [])[:req['count']]
            if len(found_objects) < req['count']:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                if 'color' in req:
                    # Color check
                    device_ = str(device) if isinstance(device, torch.device) else device
                    colors = self.color_classification(image, found_objects, classname, device_, transform=self.transform)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found " +
                            f"{colors.count(req['color'])} {req['color']}; and " +
                            ", ".join(f"{colors.count(c)} {c}" for c in self.COLORS if c in colors)
                        )
                if 'position' in req and matched and req['position']!=None:
                    # Relative position check
                    expected_rel, target_group = req['position']
                    target_group=int(target_group)
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self.relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found " +
                                        f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        # Check for non-expected objects
        for req in metadata.get('exclude', []):
            classname = req['class']
            if len(objects.get(classname, [])) >= req['count']:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)


    def evaluate_reward(self,image, objects, metadata, device):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        rewards = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = objects.get(classname, [])
            rewards.append(1-abs(int(req['count']) - len(found_objects))/int(req['count']))
            if len(found_objects) != req['count']:#对应class的物体个数不对
                correct = matched = False
                reason.append(f"expected {classname}=={req['count']}, found {len(found_objects)}")
                if 'color' in req or 'position' in req:
                    rewards.append(0.0)
            else:
                if 'color' in req:
                    # Color check
                    device_ = str(device) if isinstance(device, torch.device) else device
                    colors = self.color_classification(image, found_objects, classname,device_, transform=self.transform)
                    rewards.append(1-abs(int(req['count']) - colors.count(req['color']))/int(req['count']))
                    if colors.count(req['color']) != req['count']:#对应颜色下的物体数量对不对
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found " +
                            f"{colors.count(req['color'])} {req['color']}; and " +
                            ", ".join(f"{colors.count(c)} {c}" for c in self.COLORS if c in colors)
                        )
                if 'position' in req and matched and req['position']!=None:
                    # Relative position check
                    expected_rel, target_group = req['position']
                    target_group=int(target_group)
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self.relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found " +
                                        f"{' and '.join(true_rels)} target"
                                    )
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        rewards.append(1.0)
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, "\n".join(reason)


    def _evaluate_single(self, result, image_pil, metadata, device):
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        image = ImageOps.exif_transpose(image_pil)
        detected = {}
        confidence_threshold = self.THRESHOLD if metadata['tag'] != "counting" else self.COUNTING_THRESHOLD

        for index, classname in enumerate(self.classnames):
            if bbox[index].shape[0] == 0:
                continue
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
            ordering = ordering[:self.MAX_OBJECTS].tolist()
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
                ordering = [
                    obj for obj in ordering
                    if self.NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < self.NMS_THRESHOLD
                ]
            if not detected[classname]:
                del detected[classname]

        is_strict_correct, score, reason = self.evaluate_reward(image, detected, metadata, device)
        is_correct = False if self.only_strict else self.evaluate(image, detected, metadata, device)[0]

        return {
            'tag': metadata['tag'],
            'prompt': metadata['prompt'],
            'correct': is_correct,
            'strict_correct': is_strict_correct,
            'score': score,
            'reason': reason,
            'metadata': json.dumps(metadata),
            'details': json.dumps({
                key: [box.tolist() for box, _ in value]
                for key, value in detected.items()
            })
        }


    def evaluate_image(self,image_pils, metadatas, only_strict, device):
        start=time.time()
        with torch.inference_mode():
            results = inference_detector(self.object_detector, [np.array(image_pil) for image_pil in image_pils])
            # results = batch_inference_detector(self.object_detector, image_pils, device)
        print('1 cost %.2f s'%(time.time()-start))
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._evaluate_single, result, image_pil, metadata, device)
                for result, image_pil, metadata in zip(results, image_pils, metadatas)
            ]
            ret = [f.result() for f in futures]
        print('cost %.2f s'%(time.time()-start))
        return ret

    def reformulate_metadata(self, metadatas):
        for item in metadatas:
            # 处理 "include" 字段
            if "include" in item:
                for obj in item["include"]:
                    # 处理 count
                    if "count" in obj and isinstance(obj["count"], str):
                        try:
                            obj["count"] = int(obj["count"])
                        except ValueError:
                            pass  # 非法字符串不处理

            if "exclude" in item:
                for obj in item["exclude"]:
                    # 处理 count
                    if "count" in obj and isinstance(obj["count"], str):
                        try:
                            obj["count"] = int(obj["count"])
                        except ValueError:
                            pass  # 非法字符串不处理

                    # 处理 position
                    if "position" in obj and isinstance(obj["position"], list) and len(obj["position"]) == 2:
                        if isinstance(obj["position"][1], str):
                            try:
                                obj["position"][1] = int(obj["position"][1])
                            except ValueError:
                                pass  # 非法字符串不处理
        return metadatas
        

    def __call__(self,images, prompts, metadatas):
        metadatas=self.reformulate_metadata(metadatas)
        # print(metadatas)
        device = list(self.clip_model.parameters())[0].device
        print(device)
        # del prompts
        # if isinstance(images, torch.Tensor):
        #     images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        #     images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        # images_batched = np.array_split(images, np.ceil(len(images) / self.batch_size))
        # metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / self.batch_size))
        
        # all_scores = []
        # all_rewards = []
        # all_strict_rewards = []
        # all_group_strict_rewards = []
        # all_group_rewards = []
        # for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
        #     jpeg_images = []

        #     # Compress the images using JPEG
        #     for image in image_batch:
        #         img = Image.fromarray(image)
        #         buffer = BytesIO()
        #         img.save(buffer, format="JPEG")
        #         jpeg_images.append(buffer.getvalue())

            # # format for LLaVA server
            # data = {
            #     "images": jpeg_images,
            #     "meta_datas": list(metadata_batched),
            #     "only_strict": False,
            # }
            # data_bytes = pickle.dumps(data)

        required_keys = ['single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr']
        scores = []
        strict_rewards = []
        grouped_strict_rewards = defaultdict(list)
        rewards = []
        grouped_rewards = defaultdict(list)
        results = self.evaluate_image(images, metadatas, only_strict=self.only_strict, device=device)
        for result in results:
            strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
            scores.append(result["score"])
            rewards.append(1.0 if result["correct"] else 0.0)
            tag = result["tag"]
            for key in required_keys:
                if key != tag:
                    grouped_strict_rewards[key].append(-10.0)
                    grouped_rewards[key].append(-10.0)
                else:
                    grouped_strict_rewards[tag].append(1.0 if result["strict_correct"] else 0.0)
                    grouped_rewards[tag].append(1.0 if result["correct"] else 0.0)
        print(scores)
        return scores, rewards, strict_rewards, dict(grouped_rewards), dict(grouped_strict_rewards)

        # return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict
    
    
if __name__ == "__main__":
    data_path=os.environ.get('GENEVAL_DEMO_DATA_PATH', './geneval_demo')
    image_path=os.environ.get('GENEVAL_DEMO_IMAGE_PATH', './geneval_demo/samples')
    with open(f'{data_path}/metadata.jsonl','r') as f:
        meta_data=[json.loads(line) for line in f]
        prompts=[item['prompt'] for item in meta_data]
        
    images=[]
    for item in os.listdir(image_path):
        images.append(Image.open(f'{image_path}/{item}'))
        
    Geneval=Geneval_score(None)
    Geneval.load_to_device('cuda')
    
    start=time.time()
    Geneval(images,prompts*len(images),meta_data*4)
    print('cost %.2f s'%(time.time()-start))