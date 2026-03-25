import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from ChunkedEpisodicDataset import ChunkedEpisodicDataset
import IPython
e = IPython.embed

"""class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, min_len=None):
        super(EpisodicDataset, self).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        
        # Find min episode length if not provided
        if min_len is None:
            dataset_path = os.path.join(self.dataset_dir)
            with h5py.File(dataset_path, 'r') as root:
                min_len = float('inf')
                for episode_id in episode_ids:
                    demo = root['data'][f'demo_{episode_id}']
                    min_len = min(min_len, demo['actions'].shape[0])
        self.min_len = min_len
        print(f'Dataset initialized with min episode length: {self.min_len}')
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = True # hardcode

        episode_id = self.episode_ids[index]
        # Open the single dataset file and access demo group
        dataset_path = os.path.join(self.dataset_dir)
        with h5py.File(dataset_path, 'r') as root:
            demo = root['data'][f'demo_{episode_id}']  # Access data/demo_*
            is_sim = True
            episode_len = demo['actions'].shape[0]
            
            if sample_full_episode:
                start_ts = 0
            else:
                # Adjust random sampling to account for min_len
                max_start = episode_len - self.min_len
                if max_start > 0:
                    start_ts = np.random.choice(max_start + 1)
                else:
                    start_ts = 0
            
            # get observation at start_ts only
            qpos = demo['states/articulation/robot/joint_position'][start_ts,:-1]
            qvel = demo['states/articulation/robot/joint_velocity'][start_ts,:-1]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = demo[f'obs/{cam_name}'][start_ts]
            
            # get actions and cut to min_len
            if is_sim:
                action = demo['actions'][start_ts:start_ts + self.min_len]
            else:
                action = demo['actions'][max(0, start_ts - 1):max(0, start_ts - 1) + self.min_len]
            
            action_len = action.shape[0]

        self.is_sim = is_sim
        
        # No padding needed - all episodes are cut to same length
        is_pad = np.zeros(action_len)

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad"""
from tqdm import tqdm
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size=100, context_length=1, cache_mode='none', velocity_control=False):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.chunk_size = chunk_size
        self.cache_mode = cache_mode  # 'none', 'full', 'lazy'
        self.cache = {}
        self.velocity_control = velocity_control
        self.context_length = context_length
        if cache_mode == 'full':
            self._cache_all_episodes()
        print(f'Dataset initialized with {len(self.episode_ids)} episodes')
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)
    
    def _cache_all_episodes(self):
        """Preload all episodes into memory"""
        print("Caching all episodes...")
        dataset_path = os.path.join(self.dataset_dir)
        with h5py.File(dataset_path, 'r') as root:
            for episode_id in tqdm(self.episode_ids):
                demo = root['data'][f'demo_{episode_id}']

                gripper_pos = torch.as_tensor(demo['obs/gripper_pos'][:])   # shape (T,)
                sub_lab1 = gripper_pos[:,0] > 0.03
                T = sub_lab1.shape[0]

                # 2) Detect state changes (False<->True)
                changes = torch.zeros(T, dtype=torch.bool)
                changes[1:] = sub_lab1[1:] != sub_lab1[:-1]

                # 3) Indices of the 4 transitions (guaranteed to exist)
                change_indices = torch.where(changes)[0]
                #assert len(change_indices) == 4, f"Expected 4 state changes, got {len(change_indices)}"
                change_indices = change_indices[:4]

                # 4) One-hot encode the transitions
                encode_subtask_label = np.zeros((T, 4), dtype=np.float32)

                for i, t in enumerate(change_indices):
                    encode_subtask_label[t, i] = 1.0
                self.cache[episode_id] = {
                    #'eef_pos': demo['obs/eef_pos'][()],
                    #'eef_quat': demo['obs/eef_quat'][()],
                    #'gripper_pos': demo['obs/gripper_pos'][()],
                    'qpos': demo['states/articulation/robot/joint_position'][()][:, :-1],
                    **({'qvel': demo['states/articulation/robot/joint_velocity'][()][:, :-1]} if self.velocity_control else {}),
                    'actions': demo['actions'][()],
                    'images': {cam: demo[f'obs/{cam}'][()] for cam in self.camera_names},
                    'subtask_label': encode_subtask_label
                }
    
    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        
        if self.cache_mode == 'full' and episode_id in self.cache:
            # Use cached data
            data = self.cache[episode_id]
            episode_len = data['actions'].shape[0]
            if episode_len < 50: #to sort juicer episodes
                max_start = max(0, episode_len - int(self.chunk_size/2) - self.context_length + 1)
            else:
                max_start = max(0, episode_len - int(self.chunk_size/4) - self.context_length + 1)
            start_ts = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            
            #eef_pos = data['eef_pos'][start_ts]
            #eef_quat = data['eef_quat'][start_ts]
            #gripper_pos = data['gripper_pos'][start_ts, :-1]
            #qpos = np.concatenate([eef_pos, eef_quat, gripper_pos], axis=0)
            subtask_label = data['subtask_label'][start_ts:self.context_length + start_ts]
            qpos = data['qpos'][start_ts:self.context_length + start_ts]
            if self.velocity_control:
                qvel = data['qvel'][start_ts:self.context_length + start_ts]
            #image_dict = {cam: data['images'][cam][start_ts:self.context_length + start_ts] for cam in self.camera_names}
            #action = data['actions'][start_ts+self.context_length-1:start_ts + self.chunk_size+ self.context_length-1]
            image_dict = {cam: data['images'][cam][start_ts:self.context_length + start_ts] for cam in self.camera_names}
            if start_ts + self.chunk_size+ self.context_length-1 <= episode_len:
                action = data['actions'][start_ts+self.context_length-1:start_ts + self.chunk_size+ self.context_length-1]
                is_pad = torch.zeros(action.shape[0], dtype=torch.bool)
            else:
                action = data['actions'][start_ts+self.context_length:-1]
                is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
                is_pad[action.shape[0]:] = True
                # Pad with zeros if action sequence is shorter than chunk_size
                if action.shape[0] < self.chunk_size:
                    padding = np.zeros((self.chunk_size - action.shape[0], action.shape[1]))
                    action = np.concatenate([action, padding], axis=0)
                

            self.is_sim = is_sim = True  # hardcoded as in your original code
        else:
            episode_id = self.episode_ids[index]
            dataset_path = os.path.join(self.dataset_dir)

            with h5py.File(dataset_path, 'r') as root:
                demo = root['data'][f'demo_{episode_id}']
                is_sim = True  # hardcoded as in your original code

                episode_len = demo['actions'].shape[0]
                # Random start position ensuring chunk doesn't go out of bounds
                max_start = max(0, episode_len - int(self.chunk_size/8) - self.context_length + 1)
                start_ts = np.random.randint(0, max_start + 1) if max_start > 0 else 0

                # observation at start_ts
                qpos = demo['states/articulation/robot/joint_position'][start_ts:self.context_length + start_ts, :-1]
                if self.velocity_control:
                    qvel = demo['states/articulation/robot/joint_velocity'][start_ts:self.context_length + start_ts, :-1]
                #eef_pos = demo['obs/eef_pos'][start_ts,:]
                #eef_quat = demo['obs/eef_quat'][start_ts,:]
                #gripper_pos = demo['obs/gripper_pos'][start_ts,:-1]
                #qpos = np.concatenate([eef_pos, eef_quat, gripper_pos], axis=0)
                image_dict = {
                    cam_name: demo[f'obs/{cam_name}'][start_ts:self.context_length + start_ts]
                    for cam_name in self.camera_names
                }
                gripper_pos = demo['obs/gripper_pos']   # shape (T,)
                sub_lab1 = gripper_pos[:,0] > 0.3   # shape (T,)
                T = sub_lab1.shape[0]

                changes = torch.zeros(T, dtype=torch.bool)
                changes[1:] = (sub_lab1[1:] != sub_lab1[:-1]).any(dim=1)


                # 3) Indices of the 4 transitions (guaranteed to exist)
                change_indices = torch.where(changes)[0]
                #assert len(change_indices) == 4, f"Expected 4 state changes, got {len(change_indices)}"
                change_indices = change_indices[:4]
                # 4) One-hot encode the transitions
                encode_subtask_label = np.zeros((T, 4), dtype=np.float32)

                for i, t in enumerate(change_indices):
                    encode_subtask_label[t, i] = 1.0
                subtask_label = encode_subtask_label[start_ts:self.context_length + start_ts]
                # full action sequence
                
                if is_sim:
                    if start_ts + self.chunk_size+ self.context_length-1 <= episode_len:
                        action = demo['actions'][start_ts+self.context_length-1:start_ts + self.chunk_size+ self.context_length-1]
                        is_pad = torch.zeros(action.shape[0], dtype=torch.bool)
                    else:
                        action = demo['actions'][start_ts+self.context_length:-1]

                        # Pad with zeros if action sequence is shorter than chunk_size
                        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
                        is_pad[action.shape[0]:] = True
                        if action.shape[0] < self.chunk_size:
                            padding = np.zeros((self.chunk_size - action.shape[0], action.shape[1]))
                            action = np.concatenate([action, padding], axis=0)
    

                else:
                    if max(0, start_ts - 1) + self.chunk_size+ self.context_length-1 <= episode_len:
                        action = demo['actions'][max(0, start_ts - 1)+self.context_length-1:max(0, start_ts - 1) + self.chunk_size+self.context_length-1]
                        is_pad = torch.zeros(action.shape[0], dtype=torch.bool)
                    else:
                        
                        action = demo['actions'][max(0, start_ts - 1)+self.context_length:-1]
                        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
                        is_pad[action.shape[0]:] = True
                        # Pad with zeros if action sequence is shorter than chunk_size
                        if action.shape[0] < self.chunk_size:
                            padding = np.zeros((self.chunk_size - action.shape[0], action.shape[1]))
                            action = np.concatenate([action, padding], axis=0)

                self.is_sim = is_sim

        action_len = action.shape[0]
        #is_pad = torch.zeros(action_len, dtype=torch.bool)

        # stack camera images
        all_cam_images = np.stack(
            [image_dict[cam] for cam in self.camera_names],
            axis=0
        )

        # to torch
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        if self.velocity_control:
            qvel_data = torch.from_numpy(qvel).float() 
        action_data = torch.from_numpy(action).float()
        subtask_label_data = torch.from_numpy(subtask_label).float()

        # channel last -> channel first
        image_data = torch.einsum('b k h w c ->b k c h w', image_data)

        # normalize
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        if self.velocity_control:   
            qvel_data = (qvel_data - self.norm_stats["qvel_mean"]) / self.norm_stats["qvel_std"]
            #print("Loaded data shapes:", image_data.shape, qpos_data.shape, qvel_data.shape, action_data.shape)
            return image_data, qpos_data, action_data, is_pad, qvel_data, subtask_label_data
        else:
            return image_data, qpos_data, action_data, is_pad, subtask_label_data

def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_qvel_data = []
    all_action_data = []

    dataset_path = os.path.join(dataset_dir)
    with h5py.File(dataset_path, 'r') as root:
        for episode_idx in range(num_episodes):
            print(f'Processing episode {episode_idx}/{num_episodes}')
            demo = root['data'][f'demo_{episode_idx}']

            qpos = demo['states/articulation/robot/joint_position'][()][:, :-1]   # [T_i, nq]
            qvel = demo['states/articulation/robot/joint_velocity'][()][:, :-1]
            action = demo['actions'][()]                                 # [T_i, na]

            all_qpos_data.append(torch.from_numpy(qpos))
            all_qvel_data.append(torch.from_numpy(qvel))
            all_action_data.append(torch.from_numpy(action))

    # 🔑 Concatenate over time (not stack over episodes)
    all_qpos_data = torch.cat(all_qpos_data, dim=0)      # [sum(T_i), nq]
    print(all_qpos_data.shape)
    all_action_data = torch.cat(all_action_data, dim=0)  # [sum(T_i), na]
    print(all_action_data.shape)
    all_qvel_data = torch.cat(all_qvel_data, dim=0)      # [sum(T_i), nv]

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clamp(action_std, min=1e-2)

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clamp(qpos_std, min=1e-2)

    # normalize qvel data (not used currently)
    qvel_mean = all_qvel_data.mean(dim=0, keepdim=True)
    qvel_std = all_qvel_data.std(dim=0, keepdim=True)
    qvel_std = torch.clamp(qvel_std, min=1e-2)  

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "qvel_mean": qvel_mean.numpy().squeeze(),
        "qvel_std": qvel_std.numpy().squeeze(),
        "example_qpos": qpos,  # last episode, OK for shape reference
    }

    return stats



def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,chunk_size = 100,context_length=1, velocity_control=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size=chunk_size, context_length=context_length, cache_mode='full', velocity_control=velocity_control)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size=chunk_size, context_length=context_length, cache_mode='full', velocity_control=velocity_control)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers = 16, prefetch_factor=4, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=4, persistent_workers=True)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
