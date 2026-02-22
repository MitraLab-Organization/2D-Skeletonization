morse_code='DM++/Semantic_Segmentation_NMI/morse_code'

import sys
import numpy as np
sys.path.append(morse_code)
import albu_dingkang


def albu_cal(width, height, total_tiles, tile, mask_image, models_albu, norm_factor=16):
    """
    Run ALBU prediction on tiles in batches.
    Optimized for GPU throughput.
    
    Args:
        norm_factor: Normalization divisor for tile pixel values (PMD=16, STP=256)
    """
    # Initialize output array
    albu_out = np.zeros((512, 512, int(total_tiles)), dtype=np.uint8)
    
    # Identify valid tiles
    valid_indices = []
    count = 0
    tile_mapping = [] # (row, col) -> index in albu_out
    
    batch_size = 16
    
    # Collect valid tiles
    tiles_to_process = []
    indices_to_map = [] # To map result back to albu_out
    
    # Original 'count' tracks index in the INPUT 'tile' list
    # 'next_idx' tracks index in the OUTPUT 'albu_out' array
    input_tile_idx = 0
    output_idx = 0

    for row in range(0, width-511, 512):
        for column in range(0, height-511, 512):
            if np.sum(mask_image[row:row+512, column:column+512]):
                # Normalize and prepare tile (512, 512, 3)
                current_tile_raw = tile[input_tile_idx]
                tile_container = np.zeros((512, 512, 3), dtype=np.uint8)
                tile_container[:,:,0] = np.uint8(current_tile_raw // norm_factor)
                
                tiles_to_process.append(tile_container)
                indices_to_map.append(output_idx)
                
                input_tile_idx += 1
                output_idx += 1
            else:
                pass
                
    # Process in batches
    num_tiles = len(tiles_to_process)
    print(f"  Batching ALBU inference: {num_tiles} tiles, batch_size={batch_size}")
    
    for i in range(0, num_tiles, batch_size):
        batch_end = min(i + batch_size, num_tiles)
        batch_tiles = np.array(tiles_to_process[i:batch_end]) # (B, 512, 512, 3)
        batch_indices = indices_to_map[i:batch_end]
        
        # Predict batch
        # predict returns (B, 512, 512) or (512, 512) if B=1
        predictions = albu_dingkang.predict(models_albu, batch_tiles, image_type='8_bit_RGB')
        
        if len(predictions.shape) == 2:
            # Single item batch case (if batch_size=1 or last batch is 1)
            # Actually albu_dingkang.predict with my change returns (N, H, W) or (H, W) if squeezed
            # If N=1, squeeze makes it (H, W). If N > 1, (N, H, W).
            if len(batch_indices) == 1:
                idx = batch_indices[0]
                albu_out[:,:,idx] = predictions
            else:
                # Should not happen if N>1 but predictions rank 2, implying squeeze removed N...
                # Wait, if N=1, shape is (H, W). If N > 1, shape is (N, H, W).
                # My change: merged = np.squeeze(merged). 
                # If merged was (1, H, W, 1), squeeze -> (H, W). 
                # If merged was (N, H, W, 1), squeeze -> (N, H, W).
                # So if N > 1, rank is 3. If N=1, rank is 2.
                pass 
                
        if len(predictions.shape) == 3: 
             # (N, H, W)
             for k, idx in enumerate(batch_indices):
                 albu_out[:,:,idx] = predictions[k]
        elif len(predictions.shape) == 2:
             # (H, W) -> Single item batch
             idx = batch_indices[0]
             albu_out[:,:,idx] = predictions
             
    return albu_out
