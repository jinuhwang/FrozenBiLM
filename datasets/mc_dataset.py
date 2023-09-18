import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import pickle
import math
from pathlib import Path
import numpy as np
import os

import torch
import time

def add_noise_for_similarity(embedding, target_similarity):
    """
    Add noise to an embedding to achieve a specific target cosine similarity.
    
    Parameters:
    - embedding (torch.Tensor): The original embedding vector of shape (d,).
    - target_similarity (float): The desired cosine similarity after adding noise.

    Returns:
    - noisy_vector (torch.Tensor): The noisy vector of shape (d,).
    """
    # Save current RNG state
    current_rng_state = torch.get_rng_state()

    some_seed_value = int(time.time())

    # Manually set RNG state for this operation
    torch.manual_seed(some_seed_value)  # some_seed_value can be based on time, iteration number, etc.
    
    v_norm = embedding / torch.norm(embedding)
    # Generate a random vector of the same size
    r = torch.randn_like(v_norm)

    # Orthogonalize the random vector with respect to the original vector
    r_ortho = r - torch.dot(r, v_norm) * v_norm

    # Normalize the orthogonal random vector
    r_norm = r_ortho / torch.norm(r_ortho)

    target_similarity = torch.tensor(target_similarity)
    # Calculate epsilon for the desired degradation in similarity
    epsilon = torch.sqrt(2 * (1 - target_similarity))

    # Construct the noisy vector
    v_noisy = v_norm + epsilon * r_norm

    # Normalize the noisy vector
    v_noisy_norm = v_noisy / torch.norm(v_noisy)

    # Restore the previous RNG state
    torch.set_rng_state(current_rng_state)

    return v_noisy_norm

def add_noise_for_mse(embedding, target_mse):
    """
    Adds noise to the input embedding to achieve the given target MSE.

    Args:
    - embedding (torch.Tensor): The input embedding to which noise will be added.
    - target_mse (float): The desired mean squared error between the original and noisy embeddings.

    Returns:
    - noisy_embedding (torch.Tensor): The resulting noisy embedding.
    """
    # Save current RNG state
    current_rng_state = torch.get_rng_state()

    some_seed_value = int(time.time())

    # Manually set RNG state for this operation
    torch.manual_seed(some_seed_value)  # some_seed_value can be based on time, iteration number, etc.
    
    # Generate a random vector of the same size as the embedding
    noise = torch.randn_like(embedding)

    target_mse = torch.tensor(target_mse)
    # Calculate the required magnitude of the noise
    noise_magnitude = torch.sqrt(target_mse * embedding.numel())

    # Normalize the noise to have the desired magnitude
    normalized_noise = noise / torch.norm(noise) * noise_magnitude

    # Add the normalized noise to the embedding
    noisy_embedding = embedding + normalized_noise

    # Restore the previous RNG state
    torch.set_rng_state(current_rng_state)

    return noisy_embedding

class MC_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        subtitles_path,
        features_path,
        max_feats=10,
        features_dim=768,
        tokenizer=None,
        use_context=True,
        type_map=None,
        prefix="",
        suffix="",
    ):
        self.data = pd.read_csv(csv_path)
        self.data.fillna({'a0': '', 'a1': '', 'a2': '', 'a3': '', 'a4': ''}, inplace=True)

        if subtitles_path:
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            self.subs = None
        self.features_dir = Path('/mnt/ssd2/dataset/how2qa/openai_clip-vit-large-patch14') 
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.mask = tokenizer.mask_token if tokenizer is not None else None
        self.use_context = use_context
        mc = 0
        while f"a{mc}" in self.data:
            mc += 1
        self.mc = mc
        self.type_map = type_map
        self.prefix = prefix
        self.suffix = suffix

    def __len__(self):
        return len(self.data)

    def _get_subtitles(self, video_id, start, end):
        # only consider subtitles that intersec with the timestamps of the video clip
        subs_list = [
            x["text"]
            for x in self.subs[video_id]
            if x["end"] >= start and x["start"] <= end
        ]
        return " ".join(subs_list).capitalize().strip()

    def _get_text(self, subtitles, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: {question} Is it '{answer}'? {mask}{self.suffix}"
        )
        if self.use_context:
            text += f" Subtitles: {subtitles}"
        text = text.strip()
        return text

    def _get_video(self, video_id, start, end):
        if start is not None and not math.isnan(start):
            start = int(start)
            end = int(end)
        else:
            raise NotImplementedError

        features_not_loaded = False
        frame_features = []
        video_id = '_'.join(video_id.split('_')[:-2])
        for i in range(start, end):
            feature_path = self.features_dir / f'{video_id}_o_{i}.npz'
            if not feature_path.exists():
                features_not_loaded = True
                print(f'Feature {feature_path} not found, needed ({start}, {end})')
                break
            try:
                with np.load(feature_path, allow_pickle=True) as data:
                    features = th.tensor(data['embeddings'])

                    # For noise injection
                    noise_level = os.environ.get('INJECT_NOISE', None)
                    if noise_level is not None:
                        noise_level = float(noise_level)
                        if os.environ.get('INJECT_NOISE_MSE', None) is not None:
                            features = add_noise_for_mse(features, noise_level)
                        else:
                            features = add_noise_for_similarity(features, noise_level)
                    frame_features.append(features)
            except Exception as e:
                features_not_loaded = True
                print(f'Feature {feature_path} not loaded, needed ({start}, {end}): {e}')
                break

        if features_not_loaded:
            print("Using zero features")
            video_len = 1
            video = th.zeros(self.max_feats, self.features_dim)
        else:
            if len(frame_features) > self.max_feats:
                # sample frames
                frame_idxs = np.linspace(0, len(frame_features) - 1, self.max_feats)
                frame_idxs = np.round(frame_idxs).astype(int)
                frame_features = [frame_features[i] for i in frame_idxs]

            video_len = len(frame_features)
            for i in range(len(frame_features), self.max_feats):
                frame_features.append(th.zeros(self.features_dim))
            video = th.stack(frame_features)

        return video, video_len

    def __getitem__(self, idx):
        video_id = self.data["video_id"].values[idx]

        # get start, end
        start = self.data["start"].values[idx]
        end = self.data["end"].values[idx]

        # get question
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        if "type" in self.data:
            type = self.data["type"].values[idx]

        # get subs
        if self.subs:
            subs = self._get_subtitles(video_id, start, end)
        else:
            subs = ""

        # get features
        video, video_len = self._get_video(video_id, start, end)

        # get answer id
        answer_id = -1  # for hidden set testing
        if "answer_id" in self.data:
            answer_id = self.data["answer_id"].values[idx]

        text = []
        for i in range(self.mc):
            ai = self.data[f"a{i}"].values[idx].capitalize().strip()
            text.append(self._get_text(subs, ai, self.mask, question))

        qid = idx
        if "qid" in self.data:
            qid = int(self.data["qid"].values[idx])

        return {
            "video": video,
            "video_len": video_len,
            "text": text,
            "qid": qid,
            "answer_id": answer_id,
            "type": type,
        }


def mc_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = [
        [batch[i]["text"][j] for i in range(bs)] for j in range(len(batch[0]["text"]))
    ]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
        "qid": qid,
        "answer_id": answer_id,
        "type": type,
    }


def build_mc_dataset(dataset_name, split, args, tokenizer):
    type_map = None
    if dataset_name == "how2qa":
        if split == "train":
            csv_path = args.how2qa_train_csv_path
        elif split == "val":
            csv_path = args.how2qa_val_csv_path
        elif split == "test":
            csv_path = args.how2qa_val_csv_path  # eval on val public
        else:
            raise NotImplementedError
        subtitles_path = args.how2qa_subtitles_path
        features_path = args.how2qa_features_path
    elif dataset_name == "tvqa":
        if split == "train":
            csv_path = args.tvqa_train_csv_path
        elif split == "val":
            csv_path = args.tvqa_val_csv_path
        elif split == "test":
            csv_path = args.tvqa_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = args.tvqa_subtitles_path
        features_path = args.tvqa_features_path
    else:
        raise NotImplementedError
    return MC_Dataset(
        csv_path=csv_path,
        subtitles_path=subtitles_path,
        features_path=features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        tokenizer=tokenizer,
        use_context=args.use_context,
        prefix=args.prefix,
        suffix=args.suffix,
        type_map=type_map,
    )
