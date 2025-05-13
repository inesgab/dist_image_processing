
def run_tracking(predictor, valid_masks, video_dir):
    inference_state = predictor.init_state(video_path=video_dir)
    for idx, mask in enumerate(valid_masks):
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=idx,
            mask=mask['segmentation']
        )
    video_segments = {}
    for idx, obj_ids, logits in predictor.propagate_in_video(inference_state):
        video_segments[idx] = {
            obj_id: (logits[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(obj_ids)
        }
    return video_segments