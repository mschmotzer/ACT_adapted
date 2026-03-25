import numpy as np
import torch
import os
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ChunkedEpisodicDataset(Dataset):
    """
    Dataset that samples action chunks from episodes.
    
    Key differences from EpisodicDataset:
    - Pre-computes all valid chunk indices during initialization
    - Allows batch_size to be independent of number of episodes
    - Can sample multiple chunks from the same episode in one batch
    - __len__ returns total number of valid chunks, not number of episodes
    """
    
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, 
                 chunk_size=100, context_length=1, cache_mode='full', 
                 velocity_control=False, overlap_ratio=0.5):
        """
        Args:
            episode_ids: List of episode indices to use
            dataset_dir: Path to HDF5 dataset
            camera_names: List of camera names to load
            norm_stats: Normalization statistics
            chunk_size: Length of action chunks to extract
            context_length: Number of observation frames to use as context
            cache_mode: 'full' caches all data, 'none' reads from disk each time
            velocity_control: Whether to include velocity data
            overlap_ratio: Ratio of overlap between consecutive chunks (0.0 = no overlap, 0.5 = 50% overlap)
        """
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = True  # hardcoded as in original
        self.chunk_size = chunk_size
        self.context_length = context_length
        self.cache_mode = cache_mode
        self.velocity_control = velocity_control
        self.overlap_ratio = overlap_ratio
        self.cache = {}
        
        # Pre-compute all valid chunk indices
        self.chunk_indices = []  # List of (episode_id, start_ts, episode_len)
        
        print(f"Initializing ChunkedEpisodicDataset with {len(episode_ids)} episodes...")
        self._precompute_chunk_indices()
        
        if cache_mode == 'full':
            self._cache_all_episodes()
        
        print(f"Dataset initialized with {len(self.chunk_indices)} chunks from {len(episode_ids)} episodes")
    
    def _precompute_chunk_indices(self):
        """
        Pre-compute all valid chunk start indices across all episodes.
        Uses sliding window with configurable overlap.
        """
        dataset_path = os.path.join(self.dataset_dir)
        
        with h5py.File(dataset_path, 'r') as root:
            for episode_id in tqdm(self.episode_ids, desc="Computing chunk indices"):
                demo = root['data'][f'demo_{episode_id}']
                episode_len = demo['actions'].shape[0]
                
                # Calculate stride based on overlap ratio
                # overlap_ratio = 0.0 means stride = chunk_size (no overlap)
                # overlap_ratio = 0.5 means stride = chunk_size/2 (50% overlap)
                stride = max(1, int(self.chunk_size * (1 - self.overlap_ratio)))
                
                # We need at least context_length frames for observation
                # and chunk_size frames for actions
                min_required_len = self.context_length + self.chunk_size
                
                if episode_len < min_required_len:
                    # Episode too short, but we can still use it with padding
                    # Add one chunk starting at frame 0
                    self.chunk_indices.append((episode_id, 0, episode_len))
                else:
                    # Generate all valid start positions using sliding window
                    # Maximum start position ensures we have context_length frames before actions
                    max_start = episode_len - self.chunk_size - self.context_length + 1
                    
                    for start_ts in range(0, max_start, stride):
                        self.chunk_indices.append((episode_id, start_ts, episode_len))
                    
                    # Optionally add the last possible chunk if it wasn't included
                    if max_start % stride != 0:
                        self.chunk_indices.append((episode_id, max_start, episode_len))
        
        print(f"Generated {len(self.chunk_indices)} chunks with stride={int(self.chunk_size * (1 - self.overlap_ratio))}")
    
    def _cache_all_episodes(self):
        """Preload all episodes into memory"""
        print("Caching all episodes...")
        dataset_path = os.path.join(self.dataset_dir)
        
        with h5py.File(dataset_path, 'r') as root:
            for episode_id in tqdm(self.episode_ids, desc="Caching episodes"):
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


                cache_data = {
                    'qpos': demo['states/articulation/robot/joint_position'][()][:, :-1],
                    'actions': demo['actions'][()],
                    'images': {cam: demo[f'obs/{cam}'][()] for cam in self.camera_names},
                    'subtask_label': encode_subtask_label
                }
                
                if self.velocity_control:
                    cache_data['qvel'] = demo['states/articulation/robot/joint_velocity'][()][:, :-1]
                
                self.cache[episode_id] = cache_data
    
    def __len__(self):
        """Return total number of chunks, not episodes"""
        return len(self.chunk_indices)
    
    def __getitem__(self, index):
        """
        Get a single chunk.
        
        Args:
            index: Index into self.chunk_indices (NOT episode_id)
        
        Returns:
            Tuple of (image_data, qpos_data, action_data, is_pad, [qvel_data])
        """
        episode_id, start_ts, episode_len = self.chunk_indices[index]
        
        if self.cache_mode == 'full' and episode_id in self.cache:
            data = self.cache[episode_id]
            qpos = data['qpos']
            actions = data['actions']
            images = data['images']
            subtask_label = data['subtask_label']
            if self.velocity_control:
                qvel = data['qvel']
        else:
            # Load from disk
            dataset_path = os.path.join(self.dataset_dir)
            with h5py.File(dataset_path, 'r') as root:
                demo = root['data'][f'demo_{episode_id}']
                qpos = demo['states/articulation/robot/joint_position'][()][:, :-1]
                actions = demo['actions'][()]
                images = {cam: demo[f'obs/{cam}'][()] for cam in self.camera_names}
                if self.velocity_control:
                    qvel = demo['states/articulation/robot/joint_velocity'][()][:, :-1]
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
        
        # Extract context observations (context_length frames starting at start_ts)
        obs_start = start_ts
        obs_end = start_ts + self.context_length
        
        # Clamp to episode bounds
        obs_end = min(obs_end, episode_len)
        
        qpos_chunk = qpos[obs_start:obs_end]
        image_dict = {cam: images[cam][obs_start:obs_end] for cam in self.camera_names}
        subtask_label_chunk = subtask_label[obs_start:obs_end]
        if self.velocity_control:
            qvel_chunk = qvel[obs_start:obs_end]
        
        # Pad observations if necessary (episode too short)
        if qpos_chunk.shape[0] < self.context_length:
            pad_len = self.context_length - qpos_chunk.shape[0]
            qpos_chunk = np.concatenate([
                qpos_chunk,
                np.zeros((pad_len, qpos_chunk.shape[1]))
            ], axis=0)
  
            for cam in self.camera_names:
                img_shape = image_dict[cam].shape
                image_dict[cam] = np.concatenate([
                    image_dict[cam],
                    np.zeros((pad_len, *img_shape[1:]), dtype=image_dict[cam].dtype)
                ], axis=0)
            
            if self.velocity_control:
                qvel_chunk = np.concatenate([
                    qvel_chunk,
                    np.zeros((pad_len, qvel_chunk.shape[1]))
                ], axis=0)
            subtask_label_chunk = np.concatenate([
                subtask_label_chunk,
                np.zeros((pad_len, subtask_label_chunk.shape[1]))
            ], axis=0)
        
        # Extract actions (chunk_size actions starting after context)
        action_start = start_ts + self.context_length
        action_end = action_start + self.chunk_size
        
        if action_end <= episode_len:
            # Normal case: full chunk available
            action_chunk = actions[action_start:action_end,:]
            
            is_pad = np.zeros(self.chunk_size, dtype=bool)
        else:
            # Episode ends before chunk is complete: pad with zeros
            available_actions = actions[action_start:episode_len,:]
            num_available = available_actions.shape[0]
            
            is_pad = np.zeros(self.chunk_size, dtype=bool)
            is_pad[num_available:] = True
            
            padding = np.zeros((self.chunk_size - num_available, actions.shape[1]))
            action_chunk = np.concatenate([available_actions, padding], axis=0)
        # Stack camera images
        all_cam_images = np.stack([image_dict[cam] for cam in self.camera_names], axis=0)
        
        # Convert to torch tensors
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos_chunk).float()
        action_data = torch.from_numpy(action_chunk).float()
        subtask_label_data = torch.from_numpy(subtask_label_chunk).float()
        
        # Channel last -> channel first: [num_cams, context, H, W, C] -> [num_cams, context, C, H, W]
        image_data = torch.einsum('k t h w c -> k t c h w', image_data)
        
        # Normalize
        image_data = image_data / 255.0
        def normalize_simple(x, xmin, xmax):
            return 2 * (x - xmin) / (xmax - xmin) - 1


        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        #action_data = normalize_simple(action_data, torch.tensor([-0.3,0.0,0,0,0,0,0,0]), torch.tensor([0.3,0.7,0.8,1,1,1,1,1]))
        #qpos_data = normalize_simple(qpos_data, torch.tensor([-2.9,-1.8,-2.9,-3.08,-2.87,0.44,-3.05,0]), torch.tensor([2.9,1.8,2.9,-0.12,2.87,4.6,3.05,1]))   
        if self.velocity_control:
            qvel_data = torch.from_numpy(qvel_chunk).float()
            qvel_data = (qvel_data - self.norm_stats["qvel_mean"]) / self.norm_stats["qvel_std"]
            #qvel_data[:,:7] = normalize_simple(qvel_data[:,:7], torch.tensor([-2.62,-2.62,-2.62,-2.62,-5.26,-4.18,-5.26]), torch.tensor([2.62,2.62,2.62,2.62,5.26,4.18,5.26]))   
            return image_data, qpos_data, action_data, is_pad, qvel_data, subtask_label_data
        else:
            return image_data, qpos_data, action_data, is_pad,subtask_label_data


def load_data_chunked(dataset_dir, num_episodes, camera_names, batch_size_train, 
                      batch_size_val, chunk_size=100, context_length=1, 
                      velocity_control=False, overlap_ratio=0.5):
    """
    Load data using ChunkedEpisodicDataset.
    
    Args:
        overlap_ratio: Controls chunk overlap. 
                      0.0 = no overlap (stride = chunk_size)
                      0.5 = 50% overlap (stride = chunk_size/2)
                      Higher overlap = more chunks per episode
    
    Returns:
        train_dataloader, val_dataloader, norm_stats, is_sim
    """
    print(f'\nData from: {dataset_dir}\n')
    
    # Train/val split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    
    # Compute normalization stats (reuse existing function)
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    
    # Create datasets
    train_dataset = ChunkedEpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, context_length=context_length,
        cache_mode='full', velocity_control=velocity_control,
        overlap_ratio=overlap_ratio
    )
    
    val_dataset = ChunkedEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats,
        chunk_size=chunk_size, context_length=context_length,
        cache_mode='full', velocity_control=velocity_control,
        overlap_ratio=overlap_ratio
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=4, 
        prefetch_factor=4, 
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size_val, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=12, 
        prefetch_factor=4, 
        persistent_workers=True
    )
    
    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

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
    all_action_data = torch.cat(all_action_data, dim=0)  # [sum(T_i), na]
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