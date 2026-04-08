import requests
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import pickle
from collections import defaultdict

class Geneval_score:
    def __init__(self,):
        """Submits images to GenEval and computes a reward.
        """

        self.batch_size = 64
        self.url = "http://127.0.0.1:18085"
        self.sess = requests.Session()
        retries = Retry(
            total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
        )
        self.sess.mount("http://", HTTPAdapter(max_retries=retries))

    def __call__(self,images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / self.batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / self.batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = self.sess.post(self.url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict
    
    
if __name__ == "__main__":
    import json
    from PIL import Image
    import os
    data_path=os.environ.get('GENEVAL_URL_DEMO_DATA_PATH', './geneval_demo')
    with open(f'{data_path}/metadata.jsonl','r') as f:
        meta_data=[json.loads(line) for line in f]
        prompts=[item['prompt'] for item in meta_data]
        
    images=[]
    for item in os.listdir(os.path.join(data_path,'samples')):
        images.append(Image.open(f'{data_path}/samples/{item}'))
        
    Geneval=Geneval_score()
    
    Geneval(images,prompts*4,meta_data*4,only_strict=False)