#%%
import bpy
import logging
import kubric as kb
import numpy as np
import os 
import mediapy as media
import utils as utils
from datetime import datetime
from kubric.renderer.blender import Blender as KubricRenderer
import matplotlib.pyplot as plt
from time import time

date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
logging.basicConfig(level="INFO")

nb_videos = 1
resolution = (128, 128)
samples_per_pixel = 4
fps = 6

duration = 3*60
max_num_objects = duration 
output_dir = os.path.join('output', date)
num_speed_jumps = duration // 10

def generate_video(
            output_dir, 
            seed, 
            fps, 
            duration, 
            resolution, 
            samples_per_pixel, 
            max_num_objects):
    
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed=seed)

    (camera_init_y, camera_end_y) = (-3, -3)
    (objects_init_y, objects_end_y) = (0.0, 0.0)
    light_pos = (0, -10, 10)
    noise_scales = (0.2, 0, 0.3)
    num_objects = max_num_objects

    end_x = max_num_objects

    num_frames = fps*duration

    time0 = time()

    camera_position_curve = utils.curve_and_positions_3D(
                                boundaries=[(0, end_x),
                                            (camera_init_y, camera_end_y),
                                            (0, 0)],
                                shape_func_along_axis_and_input_boundaries=(lambda x:x, (0,1)),
                                axis=1)

    camera_orientation_curve = utils.curve_and_positions_3D(
                                boundaries=[(np.pi / 2, np.pi / 2),
                                            (0, 0),
                                            (0, 0)],
                                shape_func_along_axis_and_input_boundaries=(lambda x:x, (0,1)),
                                axis=2)

    objects_position_curve = utils.curve_and_positions_3D(
                                boundaries=[(0, end_x),
                                            (objects_init_y, objects_end_y), 
                                            (0,0)],
                                shape_func_along_axis_and_input_boundaries=(lambda x:x, (0,np.pi)),
                                axis=1)



    objects_positions = utils.regular_positionning_along_curve(
                                                objects_position_curve, 
                                                num_objects,
                                                rng,
                                                noise_scales)


    objects_scales = utils.random_scales(
                                    num_objects,
                                    rng)

    
    camera_positions, speed_jumps = utils.position_along_curve_with_varying_speed(
                                            camera_position_curve, 
                                            num_frames, 
                                            num_speed_jumps)
    
    plt.plot(np.linspace(0,1,num_frames), speed_jumps)
    plt.savefig(os.path.join(output_dir, 'speed_jumps'))
    plt.close()


    camera_orientations = utils.regular_positionning_along_curve(
                                    camera_orientation_curve, 
                                    num_frames)

    utils.preview_scene_along_axis(
                                camera_position_curve, 
                                camera_positions,
                                camera_orientations,
                                objects_position_curve,
                                objects_positions,
                                objects_scales,
                                axis=1)

    plt.savefig(os.path.join(output_dir, 'preview_from_above'))
    plt.close()
    

    scene = kb.Scene(
                resolution=resolution,
                frame_end=num_frames,
                frame_rate=fps)

    renderer = KubricRenderer(scene, 
                            samples_per_pixel=samples_per_pixel)
    renderer.use_gpu = True

    # floor = kb.Cube(position=(0,0,-1), scale=(100, 100, 0.01))


    # wall = kb.Cube(position=(0,10,0), scale=(100, 0.01, 100))

    # scene += floor
    # scene += wall

    scene = utils.add_objects(scene, objects_positions, objects_scales)


    scene += kb.DirectionalLight(name="sun",
                        position=light_pos,
                        look_at=(0, 0, 0),
                        intensity=1.5)

    scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)


    scene += kb.PerspectiveCamera(
                        name="camera",
                        euler=camera_orientation_curve(0),
                        position=camera_position_curve(0))


    scene = utils.add_positions_for_camera(scene, 
                                        camera_positions)

    # --- renders the outputs
    print('Rendering frames...')
    renderer.save_state(os.path.join(output_dir, 'simulator.blend'))
    frames_dict = renderer.render()
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    print('Writing images...')
    kb.write_image_dict(frames_dict, images_dir)

    print('Computing annotations...')
    N_plus, N_minus, visibility_masks = utils.visibility_masks_from_segmentations(
                                                        frames_dict['segmentation'],
                                                        output_dir)
    print('Encoding video...')
    media.write_video(path=os.path.join(output_dir, 'video.mp4'),
                    images=frames_dict['rgba'][:,:,:,:-1],
                    fps=fps)
    

    print('Generation time', time() - time0)

for seed in range(nb_videos):
    generate_video(
            os.path.join(output_dir, str(seed)), 
            seed, 
            fps, 
            duration, 
            resolution, 
            samples_per_pixel, 
            max_num_objects)